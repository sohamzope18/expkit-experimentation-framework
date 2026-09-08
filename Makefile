.PHONY: setup test sim app lint clean
PY := .venv/bin/python

setup:
	uv venv --python 3.11
	uv pip install --python $(PY) -e ".[app,dev]"

test:
	$(PY) -m pytest

# Regenerates every artifact in results/ from a fixed seed per study.
# Fresh clone + make sim must reproduce the committed results byte for byte.
sim:
	$(PY) -m sims.study_type1
	$(PY) -m sims.study_coverage
	$(PY) -m sims.study_power
	$(PY) -m sims.study_cuped
	$(PY) -m sims.study_peeking
	$(PY) -m sims.verify_results

app:
	.venv/bin/streamlit run app/streamlit_app

lint:
	$(PY) -m ruff check . && $(PY) -m black --check .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
