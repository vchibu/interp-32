# Dataset And Experiment Plan

## Why this dataset

The Speech Accent Archive is useful here because many speakers read the same prompt. That keeps lexical content comparatively controlled, which makes hidden-state variation easier to interpret as accent, pronunciation, or speaker-related structure instead of transcript differences.

## Canonical manifest

The first real task after adding the dataset is to convert its raw metadata into one table with at least:

- `speaker_id`
- `audio_path`
- `transcript`
- `country`
- `native_language`
- `sex`
- `age`
- `accent_group`
- `duration_sec`
- `split`

## Label strategy

We should not start with ultra-fine labels such as country if many classes have only a few samples. The best first-pass options are:

- `native_language`
- a manually grouped `accent_group`
- a geographic region derived from country

Recommended default: begin with `native_language`, then add a broader grouped label if the raw class balance is poor.

## Audio preparation

Every file should be standardized to:

- mono
- `16 kHz`
- WAV

This matches the expected input path for standard `wav2vec2` feature extraction.

## Representation extraction

The initial pass should use a frozen `Wav2Vec2Model` with:

- `output_hidden_states=True`
- utterance-level mean pooling per layer
- optional frame-level storage only for a smaller subset

Artifacts to save:

- normalized manifest
- pooled hidden states per utterance
- optional conv frontend features
- run metadata including model name and config snapshot

## First analyses

The initial analysis package should answer:

1. Which layer gives the best linear separability for the chosen accent label?
2. Do utterances cluster by label in a reduced space such as PCA or UMAP?
3. Which hidden dimensions are most informative for the label?
4. How large is the within-label vs between-label distance gap?

## Practical cautions

- Many labels will be imbalanced.
- Some metadata fields may be missing or noisy.
- Speaker identity can confound accent signals.
- Frame-level hidden-state dumps can become very large quickly.

Because of that, the most stable first experiment is pooled utterance embeddings on a balanced subset of labels with enough examples.
