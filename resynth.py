import logging
import time
from pathlib import Path

import hydra
import librosa
import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from models import get_codec, get_device

log = logging.getLogger(__name__)


def find_audio_files(src_dir: str) -> list[Path]:
    extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg")
    files = []
    src_path = Path(src_dir)
    for ext in extensions:
        files.extend(src_path.glob(ext))
    return sorted(files)


def build_grain_database(
    latents: list[torch.Tensor], unit: int, stride: int
) -> torch.Tensor:
    """Sliding window over all source latents → [N, latent_dim, unit]."""
    grains = []
    for latent in latents:
        grains.extend(extract_grains(latent, unit, stride))
    return torch.stack(grains)


def extract_grains(latent: torch.Tensor, unit: int, stride: int) -> list[torch.Tensor]:
    """Sliding window over a single latent → list of [latent_dim, unit] tensors."""
    _, num_frames = latent.shape
    grains = []
    for start in range(0, num_frames - unit + 1, stride):
        grains.append(latent[:, start : start + unit])
    return grains


def match_grains(
    tgt_grains: list[torch.Tensor],
    src_db: torch.Tensor,
    temperature: float,
    threshold: float,
) -> list[torch.Tensor]:
    """Match each target grain to a source grain via cosine similarity with softmax sampling."""
    # Flatten source grains: [N, latent_dim * unit]
    src_flat = src_db.reshape(src_db.shape[0], -1)
    # Normalize for cosine similarity
    src_norm = F.normalize(src_flat, dim=1)

    # Stack and flatten target grains: [M, latent_dim * unit]
    tgt_stack = torch.stack(tgt_grains).reshape(len(tgt_grains), -1)
    tgt_norm = F.normalize(tgt_stack, dim=1)

    # Cosine similarity: [M, N], then convert to distance
    similarities = tgt_norm @ src_norm.T
    distances = 1.0 - similarities

    min_dists = distances.min(dim=1).values
    logits = -distances / temperature
    probs = F.softmax(logits, dim=1)

    # Sample indices from the distribution
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

    matched = []
    for i, tgt_grain in enumerate(tgt_grains):
        if min_dists[i] > threshold:
            matched.append(tgt_grain)
        else:
            matched.append(src_db[sampled_indices[i]])

    return matched


def assemble_output(
    grains: list[torch.Tensor], unit: int, stride: int, num_frames: int
) -> torch.Tensor:
    """Overlap-add with averaging when stride < unit."""
    latent_dim = grains[0].shape[0]
    output = torch.zeros(latent_dim, num_frames, device=grains[0].device)
    counts = torch.zeros(num_frames, device=grains[0].device)

    for i, grain in enumerate(grains):
        start = i * stride
        end = start + unit
        if end > num_frames:
            end = num_frames
            grain = grain[:, : end - start]
        output[:, start:end] += grain
        counts[start:end] += 1

    # Average overlapping regions
    counts = counts.clamp(min=1)
    output /= counts.unsqueeze(0)
    return output


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = get_device()
    log.info("Using device: %s", device)
    log.info("Initializing codec: %s", cfg.model)
    codec = get_codec(cfg)

    # Find and encode source files
    src_files = find_audio_files(cfg.src)
    if not src_files:
        raise FileNotFoundError(f"No audio files found in {cfg.src}")
    log.info("Found %d source files", len(src_files))

    src_latents = []
    for f in src_files:
        log.info("Encoding source: %s", f.name)
        audio, _ = librosa.load(str(f), sr=codec.sample_rate, mono=True)
        src_latents.append(codec.encode(audio))

    # Grain parameters
    unit = max(1, round(cfg.grain_size * codec.frame_rate))
    stride = max(1, round(unit * cfg.stride_factor))
    log.info("Grain unit: %d frames, stride: %d frames", unit, stride)

    # Build source grain database
    src_db = build_grain_database(src_latents, unit, stride)
    log.info("Source grain database: %d grains", src_db.shape[0])

    # Encode target
    log.info("Encoding target: %s", cfg.tgt)
    tgt_audio, _ = librosa.load(cfg.tgt, sr=codec.sample_rate, mono=True)
    tgt_duration = len(tgt_audio) / codec.sample_rate

    t0 = time.time()
    tgt_latent = codec.encode(tgt_audio)
    num_frames = tgt_latent.shape[1]

    # Extract target grains
    tgt_grains = extract_grains(tgt_latent, unit, stride)
    log.info("Target grains shape:", tgt_latent.shape, "num grains:", len(tgt_grains))

    # Match grains
    log.info(
        "Matching grains (temperature=%.4f, threshold=%.2f)",
        cfg.temperature,
        cfg.threshold,
    )
    matched = match_grains(tgt_grains, src_db, cfg.temperature, cfg.threshold)

    # Assemble output latent
    output_latent = assemble_output(matched, unit, stride, num_frames)

    # Decode
    log.info("Decoding output")
    output_audio = codec.decode(output_latent)
    t1 = time.time()

    # Save
    output_path = cfg.output
    sf.write(output_path, output_audio, codec.sample_rate)
    log.info("Saved output to %s", output_path)

    # RTF (encode target + match + decode)
    elapsed = t1 - t0
    rtf = elapsed / tgt_duration
    log.info("RTF: %.2f (%.1fs processing / %.1fs audio)", rtf, elapsed, tgt_duration)


if __name__ == "__main__":
    main()
