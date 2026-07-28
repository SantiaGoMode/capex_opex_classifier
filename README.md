# CapEx/OpEx Classifier Agent

A finance analyst AI agent that classifies Jira work items as Capital Expenditure (CapEx) or Operational Expenditure (OpEx) using Qwen 2.5 32B via Ollama.

## Project Site

Explore the [use-case overview](docs/index.html) or open the [quick-start guide](docs/guide.html).

## How It Works

The agent runs in three phases:

1. **Context Loading** — Feeds company classification standards into the model.
2. **Classification** — Batches Jira items and classifies each against the standards, returning structured JSON with confidence scores.
3. **Reasoning Report** — Generates a human-readable audit report explaining each decision.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with the `qwen2.5:32b` model pulled
- Dependencies: `ollama`, `pandas`

## Setup

```bash
# Pull the model
ollama pull qwen2.5:32b

# Install dependencies
pip install ollama pandas
```

## Usage

1. Place your company standards in `work-categories.csv` (see included example).
2. Place your Jira export in `jira-items.csv`.
3. Run the classifier:

```bash
python capex-opex.py
```

Output is written to the `output/` directory:
- `<timestamp>_classified_items.csv` — Full classification results.
- `<timestamp>_classification_reasoning.txt` — Auditor-friendly reasoning report.

## Input File Formats

### work-categories.csv

| Column | Description |
|---|---|
| Category | Classification category name |
| Classification | CapEx or OpEx |
| Jira Issue Types | Allowed issue types (comma-separated) |
| Description | Category description |
| Logic/Rule | Keywords that trigger this category |

### jira-items.csv

| Column | Description |
|---|---|
| Issue Key | Jira issue key (e.g., PROJ-101) |
| Summary | Issue summary/title |
| Issue Type | Story, Bug, Task, Epic, etc. |
| Description | (Optional) Detailed description |

## Configuration

Edit the `CONFIG` dict in `capex-opex.py` to change model, file paths, batch size, or context window.

## License

MIT
