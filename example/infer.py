from SBW import SBW
import torchaudio
import torch
import os
# from pathlib import Path
# import soundfile as sf
# import librosa

def hamming_distance(s1, s2, base=2):
    """
    Calculate the absolute difference between two strings interpreted in chunks of a specified base.

    Args:
        s1 (str): The first binary string.
        s2 (str): The second binary string.
        base (int): The base to interpret the binary strings (e.g., 2 for binary, 16 for hexadecimal).

    Returns:
        int: The sum of absolute differences for corresponding chunks.

    Raises:
        ValueError: If the strings are not of equal length or invalid for the given base.
    """
    if len(s1) != len(s2):
        raise ValueError("Both strings must be of equal length")

    # Determine the chunk size for the given base
    import math

    chunk_size = int(math.log(base, 2))

    if len(s1) % chunk_size != 0 or len(s2) % chunk_size != 0:
        raise ValueError(
            f"Binary strings must be a multiple of {chunk_size} bits for base {base}"
        )

    # Split binary strings into chunks
    def to_chunks(binary_str):
        return [
            binary_str[i : i + chunk_size]
            for i in range(0, len(binary_str), chunk_size)
        ]

    chunks_s1 = to_chunks(s1)
    chunks_s2 = to_chunks(s2)

    # Convert chunks to integers and calculate absolute difference
    def absolute_difference_chunks(chunk1, chunk2):
        int1 = int(chunk1, 2)
        int2 = int(chunk2, 2)
        return abs(int1 - int2)

    return sum(
        absolute_difference_chunks(c1, c2) for c1, c2 in zip(chunks_s1, chunks_s2)
    )

def random_message(nbits: int, batch_size: int) -> torch.Tensor:
    """Return random message as 0/1 tensor."""
    if nbits == 0:
        return torch.tensor([])
    return torch.randint(0, 2, (batch_size, nbits))


def string_to_message(message_str: str, batch_size: int) -> torch.Tensor:
    """
    Convert a binary string to a message tensor.

    Args:
        message_str (str): A string of '0's and '1's.
        batch_size (int): The batch size for the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, nbits) where nbits is the length of message_str.
    """
    # Convert the string to a list of integers (0 and 1)
    message_list = [int(bit) for bit in message_str]
    nbits = len(message_list)
    # Convert the list to a tensor and repeat it for the batch size
    message_tensor = torch.tensor(message_list).unsqueeze(0).repeat(batch_size, 1)
    return message_tensor


class WatermarkSolver:
    def __init__(self):
        self.device = 'cpu'
        self.sample_rate = 16000
        self.nbits = 16
        self.model = SBW()

    def load_model(self, checkpoint_dir, checkpoint_name, strict=False):
        """
        Load the model weights from a checkpoint file.
        Compatible with both 'model_state_dict' format and
        {'embedder','detectors'} format.
        """
        import os, torch

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint {checkpoint_path} not found.")
            return

        print(f"🔄 Loading model weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # -------- 1. 兼容不同的 checkpoint 格式 --------
        if "model_state_dict" in checkpoint:
            # 老版本
            model_state_dict = checkpoint["model_state_dict"]
        elif "embedder" in checkpoint and "detectors" in checkpoint:
            # 新版本：分别保存的
            model_state_dict = {}
            for k, v in checkpoint["embedder"].items():
                model_state_dict[f"msg_processor.{k}"] = v
            for k, v in checkpoint["detectors"].items():
                model_state_dict[f"detector.{k}"] = v
        else:
            # 兜底，直接认为就是 state_dict
            model_state_dict = checkpoint

        # -------- 2. 清理 "module." 前缀 --------
        new_state_dict = {}
        for k, v in model_state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        # -------- 3. 严格模式控制 --------
        if not strict:
            new_state_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in self.model.state_dict() and self.model.state_dict()[k].shape == v.shape
            }

        # -------- 4. 加载模型 --------
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        print("✅ Model state dict loaded successfully.")
        print("   Missing keys:", missing)
        print("   Unexpected keys:", unexpected)

        # -------- 5. epoch 信息（如果有的话） --------
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
            print(f"📌 Checkpoint loaded from epoch {checkpoint['epoch']}.")
        else:
            self.epoch = -1
            print("⚠️ No 'epoch' found in checkpoint.")

    def infer_for_ui(self, input_audio_path, watermark, output_audio_path="tmp"):
        message_str = watermark
        self.model.eval()

        waveform, sample_rate = torchaudio.load(input_audio_path)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        waveform = waveform[:1].to(self.device).unsqueeze(1)
        message = string_to_message(message_str=message_str, batch_size=1).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(
                waveform,
                message=message,
            )
            y_wm = output["recon_wm"]

        os.makedirs(output_audio_path, exist_ok=True)
        watermarked_audio_path = os.path.join(
            output_audio_path,
            f"{os.path.splitext(os.path.basename(input_audio_path))[0]}_watermarked.wav",
        )
        torchaudio.save(watermarked_audio_path, y_wm.squeeze(1).cpu(), self.sample_rate)
        return watermarked_audio_path

    def decode_for_ui(self, input_audio_path):
        self.model.eval()

        waveform, sample_rate = torchaudio.load(input_audio_path)

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        waveform = waveform[:1].to(self.device).unsqueeze(1)

        with torch.no_grad():
            detect_prob, detected_message_tensor, _ = self.model.detect_watermark(waveform)
            detected_id = "".join(
                map(str, detected_message_tensor.squeeze(0).cpu().numpy().tolist())
            )

        return detect_prob,detected_id

    def embed_watermark(self, waveform, message, sample_rate: int = 16000):
        message_str = message
        self.model.eval()

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        waveform = waveform[:1].to(self.device).unsqueeze(1)
        message = string_to_message(message_str=message_str, batch_size=1).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(
                waveform,
                message=message,
            )
            y_wm = output["recon_wm"]

        return y_wm

    def decode_watermark(self, waveform, sample_rate: int = 16000):
        self.model.eval()

        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        waveform = waveform[:1].to(self.device).unsqueeze(1)

        with torch.no_grad():
            detect_prob, detected_message_tensor, _ = self.model.detect_watermark(waveform)
            detected_id = "".join(
                map(str, detected_message_tensor.squeeze(0).cpu().numpy().tolist())
            )

        return detect_prob, detected_id