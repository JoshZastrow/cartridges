## Prepare data

The `prepare-data` command runs a Prefect flow by default:

```bash
uv run cartridges prepare-data data/journals/inputs/example.pdf data/journals/output/example.md
```

Use the Anthropic agent workflow explicitly when you want the page-by-page agent loop:

```bash
uv run cartridges prepare-data data/journals/inputs/example.pdf data/journals/output/example.md --use-agent
```

The Prefect entrypoint is `src/cartridges/datagen/flows.py:prepare_data_flow`.
For local development, call it through the CLI or from Python. Once the repo is
published to GitHub, deploy it with Prefect Cloud using that entrypoint and the
GitHub source:

```bash
uvx prefect-cloud deploy src/cartridges/datagen/flows.py:prepare_data_flow \
  --from <github-account>/<repo-name> \
  --name prepare-data
```
