from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model, WhisperModel


@dataclass(slots=True)
class SpeechModelBundle:
    """A loaded speech encoder ready for hidden-state extraction.

    ``model_type`` is "wav2vec2" or "whisper" and drives the extraction path.
    For Whisper, ``model`` is the encoder only (the decoder is not needed for
    representation probing).
    """

    feature_extractor: AutoFeatureExtractor
    model: torch.nn.Module
    device: torch.device
    model_type: str
    model_name: str


# Backward-compatible alias for callers that still import the old name.
Wav2Vec2Bundle = SpeechModelBundle


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def detect_model_type(model_name: str, config: object | None = None) -> str:
    config_type = str(getattr(config, "model_type", "")).lower()
    if "whisper" in config_type or "whisper" in model_name.lower():
        return "whisper"
    return "wav2vec2"


def load_speech_model(model_name: str, device_name: str = "auto") -> SpeechModelBundle:
    device = resolve_device(device_name)
    config = AutoConfig.from_pretrained(model_name)
    model_type = detect_model_type(model_name, config)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    if model_type == "whisper":
        # Only the encoder produces the acoustic hidden states we probe.
        model: torch.nn.Module = WhisperModel.from_pretrained(model_name).get_encoder()
    else:
        model = Wav2Vec2Model.from_pretrained(model_name)

    model.to(device)
    model.eval()
    return SpeechModelBundle(
        feature_extractor=feature_extractor,
        model=model,
        device=device,
        model_type=model_type,
        model_name=model_name,
    )


def load_wav2vec2(model_name: str, device_name: str = "auto") -> SpeechModelBundle:
    """Backward-compatible wrapper; dispatches on the checkpoint type."""
    return load_speech_model(model_name, device_name)
