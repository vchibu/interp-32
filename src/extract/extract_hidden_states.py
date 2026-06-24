from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.load_wav2vec2 import Wav2Vec2Bundle


def mean_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Pool over time while respecting padding when an attention mask is present."""
    if attention_mask is None:
        return hidden_state.mean(dim=1)

    if attention_mask.shape[1] != hidden_state.shape[1]:
        return hidden_state.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


@torch.inference_mode()
def extract_utterance_embeddings(
    audio_path: str | Path,
    bundle: Wav2Vec2Bundle,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Extract hidden states and pooled layer representations for one file."""
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    inputs = bundle.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    outputs = bundle.model(**inputs, output_hidden_states=True, return_dict=True)

    pooled_layers = []
    for layer_hidden in outputs.hidden_states:
        pooled = mean_pool(layer_hidden, inputs.get("attention_mask"))
        pooled_layers.append(pooled.squeeze(0).cpu().numpy())

    # Raw CNN feature-encoder output (pre-transformer), mean-pooled over time.
    # Time length differs from the transformer hidden states, so pool unmasked.
    pooled_cnn = outputs.extract_features.mean(dim=1).squeeze(0).cpu().numpy()

    return {
        "audio_path": str(audio_path),
        "extract_features": outputs.extract_features.squeeze(0).cpu().numpy(),
        "pooled_hidden_states": np.stack(pooled_layers, axis=0),
        "pooled_cnn_features": pooled_cnn,
    }


def extract_from_manifest(
    manifest: pd.DataFrame,
    bundle: Wav2Vec2Bundle,
    output_dir: str | Path,
    sample_rate: int = 16000,
    save_frame_level: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Extract features for every row and save one .npz artifact per speaker."""
    feature_dir = Path(output_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for row in tqdm(manifest.to_dict(orient="records"), desc="Extracting hidden states"):
        audio_path = row.get("normalized_audio_path") or row["audio_path"]
        artifact_path = feature_dir / f"{row['speaker_id']}.npz"

        index_row = dict(row)
        index_row["feature_path"] = str(artifact_path)
        index_row["model_name"] = bundle.model.name_or_path
        index_row["sample_rate"] = sample_rate

        if artifact_path.exists() and not overwrite:
            with np.load(artifact_path, allow_pickle=False) as cached:
                # Recompute legacy caches that predate the pooled CNN baseline.
                if "pooled_cnn_features" in cached.files:
                    pooled_hidden_states = cached["pooled_hidden_states"]
                    index_row["n_layers"] = int(pooled_hidden_states.shape[0])
                    index_row["hidden_size"] = int(pooled_hidden_states.shape[1])
                    index_rows.append(index_row)
                    continue

        result = extract_utterance_embeddings(audio_path, bundle, sample_rate=sample_rate)

        save_payload: dict[str, Any] = {
            "pooled_hidden_states": result["pooled_hidden_states"],
            "pooled_cnn_features": result["pooled_cnn_features"],
            "audio_path": result["audio_path"],
            "speaker_id": row["speaker_id"],
            "model_name": bundle.model.name_or_path,
            "sample_rate": sample_rate,
        }
        if save_frame_level:
            save_payload["extract_features"] = result["extract_features"]
        np.savez_compressed(artifact_path, **save_payload)

        index_row["n_layers"] = int(result["pooled_hidden_states"].shape[0])
        index_row["hidden_size"] = int(result["pooled_hidden_states"].shape[1])
        index_rows.append(index_row)

    return pd.DataFrame(index_rows)
