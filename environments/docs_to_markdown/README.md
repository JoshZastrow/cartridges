# docs-to-markdown

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `docs-to-markdown`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: <single-turn | multi-turn | tool use>
- **Output format expectations (optional)**: <e.g., plain text, XML tags, JSON schema>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run docs-to-markdown
```

Configure model and sampling:

```bash
prime eval run docs-to-markdown   -m openai/gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- Put task-owned settings under `[env.taskset]` and harness-owned settings under `[env.harness]` in TOML configs.

### Taskset Config
Document any taskset config fields and their meaning. Example:

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Harness Config
Document any harness config fields and their meaning.

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

