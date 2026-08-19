# AGENTS.md

## Goal

Build a small, reproducible PT-BR continual-pretraining project based on
`Polygl0t/Tucano2-0.6B-Base`, using selected Portuguese corpora and exporting a
verifiable Hugging Face model artifact for downstream research.

## Project boundary

- This project owns real-data inventory, preparation, provenance, continual
  pretraining, language-quality evaluation and model export.
- Do not add federated learning, synthetic-secret generation, leakage attacks or
  privacy-defense experiments here.
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
- Use training sequences of 1,024 tokens; record that the native context length
  is 4,096.
- Keep GigaVerbo-v2 disabled until an explicit project decision re-enables it.
  If re-enabled, do not download the full dataset: use Hugging Face streaming,
  filter with `edu_int_score >= 4` and enforce a configurable size limit.
- Make every dataset size configurable and keep independent budgets and
  provenance for Adrenaline and OuterSpace.
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
- Export a complete Hugging Face directory and a
  `model_artifact_manifest.json` that follows the local artifact contract.
- Mark refined weights `internal_research_only` until corpus licenses, terms and
  permissions have been reviewed.

Optimize for research clarity and reproducibility, not production
infrastructure.
