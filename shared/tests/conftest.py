"""A root `uv sync` installs only the cartridges package, so a root `uv run pytest`
skips these tests unless `shared` is installed (`uv sync --all-packages`)."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("shared") else ["test_*.py"]
