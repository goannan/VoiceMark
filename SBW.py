from STmodels.model import SpeechTokenizer
from models import WMEmbedder, WMDetector
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class SBW(nn.Module):
    def __init__(self):
        super().__init__()
        self.nbits = 16
        config_path = (
            "D:/programs/VoiceMark/STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
        )
        ckpt_path = "D:/programs/VoiceMark/STmodels/pretrained_model/SpeechTokenizer.pt"
        self.st_model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        self.msg_processor = WMEmbedder(
            nbits=16,
            input_dim=1024,
            nchunk_size=4,
        )
        self.detector = WMDetector(
            1024,
            16,
            nchunk_size=4,
        )

    def detect_watermark(
        self, x: torch.Tensor, return_logits=False
    ) -> Tuple[float, torch.Tensor]:
        embedding = self.st_model.forward_feature(x)
        if return_logits:
            return self.detector(embedding)
        return self.detector.detect_watermark(embedding)

    def forward(
        self,
        speech_input: torch.Tensor,
        message: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        recon, recon_wm, acoustic, acoustic_wm = self.st_model(
            speech_input, msg_processor=self.msg_processor, message=message
        )
        wav_length = min(speech_input.size(-1), recon_wm.size(-1))
        speech_input = speech_input[..., :wav_length]
        recon = recon[..., :wav_length]
        recon_wm = recon_wm[..., :wav_length]
        return {
            "recon": recon,
            "recon_wm": recon_wm,
        }
