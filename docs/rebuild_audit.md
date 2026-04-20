# Rebuild Audit

## Current Repo State
- One plain-text draft file in the repo root.
- No experiment code.
- No dataset loaders.
- No feature extraction or manifest pipeline.
- No plotting or table generation.
- No artifact store.
- No tests.
- No git metadata.

## Salvageable Material
- Operational fixed-threshold framing.
- Some references and problem motivation.
- The idea that transfer should be evaluated without target retuning.

## Off Critical Path
- Manual numeric claims in the draft.
- Existing significance section.
- Calibration-centric storyline.
- Trust or certification material as a main-paper assumption.
- Any theorem that does not survive the pilot.

## Missing for the Target Paper
- Common feature schema across `IoT-23`, `ToN-IoT`, `UNSW-NB15`.
- Deterministic split manifests.
- Ontology mapping.
- Canonical transfer benchmark pipeline.
- Oracle novelty computation.
- Frozen deployable `alpha_hat`.
- Decision-task evaluation.
- Controlled-shift experiment.
- Generated figures, tables, and claim provenance.

## Environment Reality
- `make`, `pdflatex`, and `latexmk` exist.
- Homebrew Python `3.14` is not the target interpreter.
- `/usr/bin/python3` is `3.9.6` and is the project interpreter.

## Immediate Execution Order
1. Freeze the scientific contract and pilot scope.
2. Stage lean configs, docs, and pilot-critical modules.
3. Validate dataset availability and common-feature viability.
4. Run the fast-fail pilot before broad benchmark or theorem work.
