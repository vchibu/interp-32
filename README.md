# Speech Accent Archive + Wav2Vec2 Analysis

Starter scaffold for preparing the Speech Accent Archive dataset, extracting hidden states from `wav2vec2`, and analyzing internal dimensions layer by layer.

## Intended workflow

1. Put the original dataset files in `data/raw/`.
2. Build a canonical manifest with speaker metadata and file paths.
3. Normalize audio to mono 16 kHz WAV in `data/processed/audio/`.
4. Run frozen `wav2vec2` extraction and save pooled or frame-level hidden states.
5. Analyze layer-wise separability and hidden dimensions.

## CLI

The starter CLI now supports:

- `python -m src.cli scrape-archive`
- `python -m src.cli build-manifest`
- `python -m src.cli normalize-audio`
- `python -m src.cli extract-hidden-states`
- `python -m src.cli run-probes`
- `python -m src.cli summarize-accent-groups`

Example bootstrap flow:

1. `python -m src.cli scrape-archive --languages arabic english --max-speakers 20`
2. `python -m src.cli build-manifest`
3. `python -m src.cli normalize-audio`
4. `python -m src.cli extract-hidden-states`
5. `python -m src.cli run-probes`

Dataset summary example:

`python -m src.cli summarize-accent-groups --top-n 20`

## Project layout

- `configs/`: experiment and path configuration
- `data/`: raw data, processed audio, manifests
- `docs/`: dataset and experiment design notes
- `notebooks/`: exploratory analysis notebooks
- `src/data/`: dataset preparation utilities
- `src/models/`: model loading utilities
- `src/extract/`: embedding and hidden-state extraction
- `src/analysis/`: probing and dimensional analysis
- `outputs/`: extracted features, figures, metrics

## Notes on the dataset

This scaffold assumes:

- one utterance per speaker is the common case
- recordings may arrive in mixed formats and sampling rates
- metadata will need canonicalization into a single manifest table
- the primary transcript is the shared "Please call Stella" passage

## Recommended first experiment

- Use a frozen `Wav2Vec2Model`
- Extract all hidden layers with `output_hidden_states=True`
- Save utterance-level mean pooled embeddings for every layer
- Start with broad accent or native-language group labels rather than fine country labels
