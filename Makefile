PYTHON ?= /usr/bin/python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
ROOT_MANUSCRIPT_ARTIFACTS := manuscript.aux manuscript.bbl manuscript.blg manuscript.fdb_latexmk manuscript.fls manuscript.log manuscript.pdf manuscript.spl missfont.log

.PHONY: env stage pilot features benchmark decision render validate test fmt lint journal acsac acsac-diagnostics recalibration ranking-diagnosis source-holdout-fpr native-aux coral-aux clean-root-manuscripts verify-no-root-manuscripts submission-check

env:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

stage:
	$(PY) scripts/01_stage_datasets.py

pilot:
	$(PY) scripts/00_pilot_fast_fail.py --config configs/pilot.yaml

features:
	$(PY) scripts/10_build_common_features.py --config configs/base.yaml

benchmark:
	$(PY) scripts/20_run_transfer_benchmark.py --config configs/benchmark.yaml

decision:
	$(PY) scripts/30_run_decision_eval.py --config configs/decision.yaml

render:
	$(MAKE) ranking-diagnosis
	$(MAKE) source-holdout-fpr
	$(MAKE) native-aux
	$(MAKE) coral-aux
	$(PY) scripts/40_render_artifacts.py --config configs/render.yaml

acsac-diagnostics:
	$(PY) scripts/42_build_acsac_diagnostics.py --config configs/render.yaml

recalibration:
	$(PY) scripts/46_unlabeled_recalibration.py --settings artifacts/benchmark/settings.csv --predictions artifacts/benchmark/predictions.csv --output-dir artifacts/recalibration

ranking-diagnosis:
	$(PY) scripts/43_ranking_diagnosis.py --settings artifacts/benchmark/settings.csv --predictions artifacts/benchmark/predictions.csv --output-dir artifacts/ranking_diagnosis

source-holdout-fpr:
	$(PY) scripts/48_source_holdout_fpr.py --settings artifacts/benchmark/settings.csv --predictions artifacts/benchmark/predictions.csv --output artifacts/benchmark/source_holdout_fpr.csv

native-aux:
	$(PY) scripts/49_native_feature_auxiliary.py --config configs/render.yaml

coral-aux:
	$(PY) scripts/52_coral_auxiliary.py --config configs/render.yaml

clean-root-manuscripts:
	rm -f $(ROOT_MANUSCRIPT_ARTIFACTS)

verify-no-root-manuscripts:
	@for path in $(ROOT_MANUSCRIPT_ARTIFACTS); do \
		if [ -e "$$path" ]; then \
			echo "Unsupported root-level manuscript artifact present: $$path"; \
			echo "Canonical manuscript PDFs are paper/manuscript.pdf and paper/acsac/manuscript.pdf"; \
			exit 1; \
		fi; \
	done

journal: clean-root-manuscripts
	$(PY) scripts/42_build_acsac_diagnostics.py --config configs/render.yaml
	$(MAKE) ranking-diagnosis
	$(MAKE) source-holdout-fpr
	$(MAKE) recalibration
	$(MAKE) native-aux
	$(MAKE) coral-aux
	$(PY) scripts/41_build_manuscript_assets.py --config configs/render.yaml
	latexmk -pdf -cd -interaction=nonstopmode paper/manuscript.tex
	$(MAKE) verify-no-root-manuscripts

acsac: clean-root-manuscripts
	$(PY) scripts/42_build_acsac_diagnostics.py --config configs/render.yaml
	$(MAKE) ranking-diagnosis
	$(MAKE) source-holdout-fpr
	$(MAKE) recalibration
	$(MAKE) native-aux
	$(MAKE) coral-aux
	$(PY) scripts/41_build_manuscript_assets.py --config configs/render.yaml
	latexmk -pdf -cd -interaction=nonstopmode paper/acsac/manuscript.tex
	$(MAKE) verify-no-root-manuscripts

validate:
	$(PY) scripts/50_validate_provenance.py --config configs/render.yaml

submission-check:
	$(PY) scripts/51_validate_submission_metadata.py --manuscript paper/manuscript.tex

test:
	$(PY) -m unittest discover -s tests -p 'test_*.py'

lint:
	$(BIN)/ruff check src tests scripts
