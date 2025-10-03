from infer import (
    WatermarkSolver,
    # hamming_distance
)


solver = WatermarkSolver()
solver.load_model(checkpoint_dir="../", checkpoint_name="voicemark.pth", strict=True)
# solver.load_model(checkpoint_dir="../", checkpoint_name="train/Log/spt_base/WatermarkTrainer_00005000.pt", strict=True)


def embed_watermark(waveform, sample_rate: int = 16000, message: str = "1111111100000000"):
    watermarked_audio = solver.embed_watermark(waveform = waveform, message = message, sample_rate = sample_rate)
    return watermarked_audio

def decode_watermark(waveform, sample_rate: int = 16000):
    try:
        detect_prob, decoded_id = solver.decode_watermark(waveform, sample_rate)
        # if detect_prob < 1e-2:
        #     return "No matching watermark found"
        return detect_prob, decoded_id
    except ValueError as e:
        return str(e)
