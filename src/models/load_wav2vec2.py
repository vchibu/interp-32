from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model


@dataclass(slots=True)
class Wav2Vec2Bundle:
    feature_extractor: AutoFeatureExtractor
    model: Wav2Vec2Model
    device: torch.device


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_wav2vec2(model_name: str, device_name: str = "auto") -> Wav2Vec2Bundle:
    device = resolve_device(device_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return Wav2Vec2Bundle(
        feature_extractor=feature_extractor,
        model=model,
        device=device,
    )
