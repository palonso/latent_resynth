import numpy as np
import torch
from music2latent import EncoderDecoder

from . import get_device


def _patch_autocast_device(device: torch.device):
    """Monkey-patch music2latent's hardcoded device_type='cuda' in torch.autocast."""
    _original_autocast = torch.autocast

    class _PatchedAutocast(_original_autocast):
        def __init__(self, device_type, *args, **kwargs):
            if device_type == "cuda" and not torch.cuda.is_available():
                device_type = device.type
            super().__init__(device_type, *args, **kwargs)

    torch.autocast = _PatchedAutocast


class Music2LatentCodec:
    def __init__(self, cfg):
        self._device = get_device()
        _patch_autocast_device(self._device)
        self._encdec = EncoderDecoder(device=self._device)
        self._sample_rate = cfg.music2latent.sample_rate
        self._max_batch_size = cfg.music2latent.max_batch_size

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_rate(self) -> float:
        # music2latent produces ~10 Hz latent frames (1 frame per ~100ms)
        return self._sample_rate / 4410

    def encode(self, audio: np.ndarray) -> torch.Tensor:
        latent = self._encdec.encode(audio, max_batch_size=self._max_batch_size)
        # squeeze batch dim → [latent_dim, seq_len]
        return latent.squeeze(0)

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        # add batch dim → [1, latent_dim, seq_len]
        audio = self._encdec.decode(latent.unsqueeze(0), max_batch_size=self._max_batch_size)
        return audio.squeeze().cpu().numpy()
