# IOT_IDS

Lean, reproducible infrastructure for a tabular flow-level NIDS transfer paper centered on fixed-threshold cross-dataset evaluation under domain shift.

## Scope

The repository studies cross-domain transfer for shared-feature tabular intrusion detection using:

- IoT-23
- ToN-IoT
- UNSW-NB15

The paper-facing core is a benchmark of source-trained models whose thresholds are fixed on source validation and transferred unchanged to target domains. The repo also contains supporting diagnostics for:

- ranking failure vs. threshold drift
- oracle novelty on the claim-bearing IoT subset
- benign covariate shift
- target-free monitoring audits
- benign-only recalibration
- operating-budget robustness
- an auxiliary CORAL comparison

## Repository Layout

- `src/iot_ids/` - reusable pipeline and manuscript-asset code
- `scripts/` - executable stages for staging, benchmarking, rendering, validation, and auxiliary analyses
- `configs/` - run and render configuration
- `artifacts/` - generated benchmark outputs, tables, claims, and manuscript inputs
- `paper/` - journal and ACSAC manuscript sources, generated tables, and figures
- `docs/` - protocol notes, data contracts, ontology rules, and rebuild audit material
- `tests/` - unit and manuscript-consistency tests

## Quick Start

Create the environment:

```bash
make env
```

Common pipeline entry points:

```bash
make features
make benchmark
make render
make validate
make test
```

Build the manuscripts:

```bash
make journal
make acsac
```

## Reproducibility Rules

This repo is intentionally paper-oriented:

- no manual tables
- no prose-only empirical numbers
- thresholds selected on source validation only
- generated figures and tables flow from artifacts
- claim-bearing novelty analysis remains restricted to the IoT-only subset

## Notes

- The canonical Python package metadata lives in `pyproject.toml`.
- The main workflow commands are defined in `Makefile`.
- Manuscript metadata is still partially placeholder-driven and should be finalized before submission.
