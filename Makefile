.PHONY: setup notebook prepare-data

setup:
	uv sync

notebook:
	uv run marimo edit notebooks/

prepare-data:
	set -a && . ./.env.local && set +a && uv run cartridges prepare_data "$(INPUT)" "$(OUTPUT)" --model "$(MODEL)"
