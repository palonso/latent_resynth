import numpy as np
import torch
from encodec import EncodecModel

from . import get_device


class EnCodecCodec:
    def __init__(self, cfg):
        self._device = get_device()
        self._model = EncodecModel.encodec_model_24khz()
        self._model.set_target_bandwidth(cfg.encodec.bandwidth)
        self._model.eval()
        self._model.to(self._device)
        self._sample_rate = cfg.encodec.sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_rate(self) -> float:
        # EnCodec 24kHz model: 75 Hz frame rate
        return 75.0

    def encode(self, audio: np.ndarray) -> torch.Tensor:
        # audio: 1D numpy → [1, 1, T] tensor
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            # Use encoder directly to get continuous embeddings (bypass RVQ)
            latent = self._model.encoder(wav)
        # latent: [1, latent_dim, num_frames] → [latent_dim, num_frames]
        return latent.squeeze(0)

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        # [latent_dim, num_frames] → [1, latent_dim, num_frames]
        latent = latent.unsqueeze(0).to(self._device)
        with torch.no_grad():
            audio = self._model.decoder(latent)
        # [1, 1, T] → 1D numpy
        return audio.squeeze().cpu().numpy()
