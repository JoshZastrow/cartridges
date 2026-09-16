.PHONY: setup notebook

setup:
	uv sync

notebook:
	uv run marimo edit notebooks/
