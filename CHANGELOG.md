# Changelog

All notable changes to this repository.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Pre-arXiv preprint of the methodology paper
  *Silent Contamination in LLM Merging Evaluation* in
  `papers/eval_trust/draft.pdf`.
- `eval_trust/` audit toolkit:
  - `paired_stats.py`: McNemar exact, Wilson CI, paired bootstrap, sample-size planning.
  - `conformal_ci.py`: split-conformal accuracy intervals.
  - `t0v2/truncation_extract.py`: A_truncated channel detector.
  - `t0v2/aggregate.py`: alpha/beta/gamma routing decision aggregator.
  - `lm_eval_bridge.py`: adapter from lm-evaluation-harness samples_*.jsonl.
- `run_audit.py`: end-to-end reproducer of the case-study flip.
- `papers/eval_trust/scripts/make_figures.py`: regenerates all 3 paper figures
  from `data/`.
- `data/`: raw paired logs for the case study, anonymized marginal-protect
  history, quantization granularity summary, GSM8K dev split.
- `tests/`: 41 unit tests across paired_stats, conformal_ci, lm_eval_bridge.
- GitHub Actions CI on Python 3.10 / 3.11 / 3.12 (pytest + reproducer +
  figure regen + ruff).
- `EXAMPLES.md`: 10 copy-pasteable recipes for common audit tasks.
- `CONTRIBUTING.md`, `SECURITY.md`, issue templates.

## [0.1.0] — 2026-06-05

Initial public release. arXiv ID pending endorsement.

[Unreleased]: https://github.com/telleroutlook/evomerge-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/telleroutlook/evomerge-framework/releases/tag/v0.1.0
