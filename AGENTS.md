# AGENTS.md

## Goal

Build a small, reproducible PT-BR continual-pretraining project based on
`Polygl0t/Tucano2-0.6B-Base`, using selected Portuguese corpora and exporting a
verifiable Hugging Face model artifact for downstream research.

## Project boundary

- This project owns real-data inventory, preparation, provenance, continual
  pretraining, language-quality evaluation and model export.
- Do not create runtime dependencies on a sibling project or on paths outside
  this project. External datasets and model artifacts may be selected only by
  explicit configuration.

## Rules

- Keep the implementation simple and prefer plain Python with small reusable
  modules.
- Do not add frameworks, services, databases, dashboards or abstractions unless
  required.
- Start from `Polygl0t/Tucano2-0.6B-Base` at revision
  `dad97dc864a8f9a1d240fb9351d098f3af9511d7`.
- Use full-parameter continual pretraining unless an experimental assumption is
  explicitly changed and documented.
- Keep the original tokenizer, vocabulary, token IDs and special tokens
  unchanged.
- Make every dataset size configurable and keep independent budgets and
  provenance.
- Preserve source and provenance metadata throughout processing.
- Never commit datasets, source records, derived shards, model weights, secrets,
  caches or generated checkpoints.
- Do not copy personal values, usernames, URLs or messages from source corpora
  into documentation, fixtures, logs or tests.
- Make experiments reproducible with fixed seeds and versioned config files.
- Save metrics and manifests separately from model checkpoints.
- Avoid unnecessary checkpoints and large intermediate files.
- Prefer resumable preparation and training scripts.
- Before adding a dependency, check whether the existing stack can do the job.
- Do not silently change dataset, model or training assumptions. Document
  important decisions in the README and executable configuration.

Optimize for research clarity and reproducibility, not production
infrastructure.
