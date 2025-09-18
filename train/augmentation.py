import torch
import torchaudio
import random
from STmodels.modules.seanet import SEANetEncoder, SEANetDecoder
from attacks import AudioEffects

def vc_simulated_augmentation(audio, sample_rate=16000, orig_audio=None):
    """
    VC-Simulated Augmentation
    Args:
        audio: Tensor [Batch, 1, T]
        sample_rate: audio sample rate
        orig_audio: original audio (for partial watermark filtering)
    Returns:
        augmented_audio: augmented audio
        pred_label: Tensor [Batch, time_steps], 1=watermarked frame, 0=non-watermarked frame
    """
    batch_size, T = audio.shape[0], audio.shape[-1]
    augmented = audio.clone()

    # 2. Completely different content (shuffle in 50ms windows, 50%)
    if random.random() < 0.5:
        win_len = int(0.05 * sample_rate)
        chunks = list(augmented.split(win_len, dim=-1))
        random.shuffle(chunks)
        augmented = torch.cat(chunks, dim=-1)

    # 3. Partial watermark filtering (replace with original audio, 50%)
    if orig_audio is not None and random.random() < 0.5:
        seg_len = int(0.05 * sample_rate)
        start = random.randint(0, T - seg_len)
        augmented[:, start:start + seg_len] = orig_audio[:, start:start + seg_len]

    # 4. Neural codec encoding & decoding (EnCodec)
    if random.random() < 0.3:
        encoder = SEANetEncoder().to(audio.device)
        decoder = SEANetDecoder().to(audio.device)
        y = encoder(augmented)
        z = decoder(y)
        augmented = z
        # Assume the watermark remains after codec by default, pred_label remains unchanged

    # 5. Audio perturbation (speed/gain/resample, 10%)
    if random.random() < 0.1:
        choice = random.choice(["speed", "filtering", "resample"])
        if choice == "speed":
            augmented = AudioEffects.speed(
                augmented,  # [B, 1, T]
                speed_range=(0.8, 1.2),
                sample_rate=sample_rate
            )
        elif choice == "filtering":
            filtering_kinds = random.choice(["lowpass", "highpass", "bandpass"])
            if filtering_kinds == "lowpass":
                cutoff = random.uniform(2000, 8000)
                augmented = AudioEffects.lowpass_filter(augmented, cutoff_freq=cutoff, sample_rate=sample_rate)
            elif filtering_kinds == "highpass":
                cutoff = random.uniform(200, 1000)
                augmented = AudioEffects.highpass_filter(augmented, cutoff_freq=cutoff, sample_rate=sample_rate)
            else:  # bandpass
                low = random.uniform(200, 1000)
                high = random.uniform(2000, 8000)
                augmented = AudioEffects.bandpass_filter(augmented, cutoff_freq_low=low, cutoff_freq_high=high,sample_rate=sample_rate)
        elif choice == "resample":
            new_sr = random.choice([8000, 22050, 32000])
            resample = torchaudio.transforms.Resample(sample_rate, new_sr).to(audio.device)
            augmented = resample(augmented)
            resample_back = torchaudio.transforms.Resample(new_sr, sample_rate).to(audio.device)
            augmented = resample_back(augmented)
        # pred_label remains unchanged

    downsampling_ratios = 320
    n_frames = augmented.shape[-1] // downsampling_ratios  # number of frames 50 steps per audio
    pred_label = torch.ones(batch_size, n_frames, device=audio.device) # default to 1（watermark frames）
    # 1. No watermark in silent frames (20%)
    if random.random() < 0.2:
        n_mask = max(1, int(0.1 * n_frames))  # mask 10% frames randomly
        mask_frames = random.sample(range(n_frames), n_mask)
        for f in mask_frames:
            start = f * downsampling_ratios
            end = start + downsampling_ratios
            augmented[:, :, start:end] = 0.0
            pred_label[:, f] = 0

    return augmented, pred_label
