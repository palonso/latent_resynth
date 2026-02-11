from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class AudioCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    def encode(self, audio: np.ndarray) -> torch.Tensor:
        """Encode mono audio to latent representation [latent_dim, num_frames]."""
        ...

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        """Decode latent representation to mono audio."""
        ...


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_codec(cfg) -> AudioCodec:
    if cfg.model == "music2latent":
        from .music2latent_codec import Music2LatentCodec

        return Music2LatentCodec(cfg)
    elif cfg.model == "encodec":
        from .encodec_codec import EnCodecCodec

        return EnCodecCodec(cfg)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")
