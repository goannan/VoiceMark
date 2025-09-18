import torch
import torchaudio
import soundfile as sf
import librosa
from pathlib import Path
import IPython.display as ipd
import matplotlib.pyplot as plt
import torch


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    兼容 torchaudio 2.9+ 的音频加载函数
    Args:
        path: 音频路径
        target_sr: 目标采样率
    Returns:
        waveform: torch.Tensor, shape [1, num_samples]
        sr: int, 采样率
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 尝试用 torchaudio.load（新版本可能需要 TorchCodec）
    try:
        waveform, sr = torchaudio.load(path)
        return waveform, sr
    except RuntimeError:
        pass  # 如果报错，则 fallback

    # fallback 1: 用 soundfile
    try:
        data, sr = sf.read(path)
        waveform = torch.from_numpy(data.T).float()  # 转为 [channels, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # 单通道
        return waveform, sr
    except Exception:
        pass

    # fallback 2: 用 librosa
    try:
        data, sr = librosa.load(path, sr=target_sr, mono=False)
        waveform = torch.from_numpy(data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {path}") from e

def plot_waveform_and_specgram(waveform, sample_rate, title):
    waveform = waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"{title} - Waveform and specgram")
    plt.show()

def play_audio(waveform, sample_rate):
    if waveform.dim() > 2:
        waveform = waveform.squeeze(0)
    waveform = waveform.detach().cpu().numpy()

    num_channels, *_ = waveform.shape
    if num_channels == 1:
        ipd.display(ipd.Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        ipd.display(ipd.Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")