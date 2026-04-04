# Contributing Guide

Thanks for your interest in contributing to tabra! This document explains how to participate in development.

## Project Overview

tabra is a Python econometrics and social science toolkit that aims to precisely implement commonly used computational methods in social sciences. The project uses Stata as the reference standard — Python implementations must match Stata output digit by digit.

## Claude Code Plugin Setup

This project uses [Claude Code](https://docs.anthropic.com/en/docs/claude-code) as an AI-assisted development tool. The following plugins are required:

### Install the superpowers plugin (provides TDD, brainstorming, debugging skills)

Run in Claude Code:

```
/install-plugin superpowers
```

### Install the planning-with-files plugin

```
/install-plugin planning-with-files
```

After installation, plugin config is written to `.claude/settings.json`. Project-bundled skills (`learn-stata-command`, `verify-command`, `fix-from-report`, `intuitive-test`) are in `.claude/skills/` and require no extra installation.

> If you don't use Claude Code, skip this section and follow the workflow below manually.

## Environment Setup

### Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Stata (for cross-validation)
- [stata-mcp](https://github.com/sepinetam/stata-mcp) CLI tool

### Install

```bash
git clone https://github.com/sepinetam/tabra.git
cd tabra
uv sync --dev
```

### Verify Environment

```bash
uv run pytest tests/ -v           # Run tests
uv tool install stata-mcp         # Install stata-mcp (skip if already installed)
#uv tool upgrade stata-mcp        # (optional) Upgrade stata-mcp to latest
stata-mcp --version               # Verify stata-mcp is available
uv run main.py                    # Run demo
```

## Workflow: Implementing a New Command

Using the implementation of Stata command `xxx` as an example, the full workflow is:

### Step 1: Learn the Command

In Claude Code, type:

```
/learn-stata-command xxx
```

This skill automatically: fetches help docs → retrieves ado source code → writes demo do-file → runs it → generates command breakdown doc at `docs/stata/command/xxx.md`.

**You must manually review `docs/stata/command/xxx.md` after generation** to confirm the output checklist is complete and algorithm descriptions are accurate.

> **Note: The `docs/` directory is NOT committed to git.** These docs are for local development reference only — do not `git add docs/`.

> **If using Claude Code:** These rules should be written into `CLAUDE.local.md`, and the AI will follow them automatically. When working manually, keep this in mind.

### Step 2: Brainstorm

Before writing tests, brainstorm the implementation approach:

```
/brainstorm Let's design how to implement `xxx` command, including function signature design, parameter forms, etc.
```

Discuss: module decomposition, data structure design, algorithm choices, which Stata computation steps to reference.

Then organize into a planning document:

```
/planning-with-files Organize the above development plan into a document
```

The key deliverable here is a spec document.

> **Note: Spec documents are NOT committed to git.** These are for local development reference only — do not `git add docs/specs/`.

> **If using Claude Code:** These rules should be written into `CLAUDE.local.md`, and the AI will follow them automatically. When working manually, keep this in mind.

### Step 3: Write Tests (TDD)

Write tests first, then implementation. Test file: `tests/test_xxx.py`

```python
# Tests use Stata output as reference values
def test_xxx_basic():
    result = tab.xxx(...)
    # Coefficients, SE, t-values, p-values, R², sigma_u, etc. — verify all
    assert result.coef["weight"] == pytest.approx(3.464706, abs=1e-6)
```

Run tests (they should all fail at this point):

```bash
uv run pytest tests/test_xxx.py -v
```

### Step 4: Implement

Write implementation code in the appropriate location under `src/tabra/`. Follow existing code patterns and style.

Key requirements:
- All statistics must be computed precisely — no approximations
- t-values must use `scipy.stats.t.cdf`, not the 1.96 critical value approximation
- p-values must use exact distributions, not normal approximations
- All intermediate calculations must be precise — no shortcuts

Run tests until all pass:

```bash
uv run pytest tests/test_xxx.py -v
```

### Step 5: Cross-Validate

In Claude Code, run:

```
/verify-command xxx
```

This skill automatically: writes verification code → runs both Stata and Python → compares item by item → generates verification report at `.verify/reports/xxx.md`.

Compare **all** output values: coefficients, SE, t/z, p-value, CI, R², sigma_u, sigma_e, rho, F/chi2, DoF, N obs, and every other visible numeric value.

**Precision requirement: zero tolerance.** All values must match Stata digit by digit. The only acceptable difference is last-digit floating-point rounding.

If there are discrepancies, run the fix skill:

```
/fix-from-report xxx
```

### Step 6: Update Index

After fixing, run `/verify-command xxx` again for a second verification. Once it passes, update `.verify/INDEX.md`.

## Code Standards

### File Structure

```
src/tabra/
├── core/          # Core functionality (data loading, config)
├── models/        # Model implementations
│   └── estimate/  # Estimators (OLS, IV, panel, etc.)
├── plot/          # Visualization
├── io/            # Import/export
└── utils/         # Utility functions

tests/
├── test_xxx.py    # Tests for corresponding command

docs/stata/command/
├── xxx.md         # Command breakdown doc
```

### Comment Style

- Project files (`src/`): English docstrings, Google style
- One-off scripts (`tmp/`): Use developer's preferred language
- Single-line comments: English in `src/`, developer's preferred language in scripts

### Python Execution Rules

- **Forbidden:** `python3 -c`, `python /dev/stdin <<`, `echo | python`, and any inline execution
- **Required:** Write script files and run with `uv run`
- Temporary scripts go in `tmp/` directory

### File Read/Write Rules

- **Read files** using the `Read` tool — no `cat`, `sed`, `awk`
- **Write/Edit files** using `Write` or `Edit` tool — no `sed -i`, `cat >`

### Git Standards

- Commit messages use conventional format: `feat: add xxx` / `fix: correct xxx` / `docs: update xxx`
- **Forbidden:** `git add -A` or `git add .` — must specify files explicitly
- **Forbidden:** committing `tmp/`, `.local/`, `docs/`, `main.py`
- Run full test suite before `git push`: `uv run pytest tests/ -v`
- Check `git status` before writing `git add` to see what's uncommitted
- After `git add`, review what's staged with `git diff --cached` before writing the commit message
- One feature per commit — do not mix multiple features in a single commit

## Data Management

Stata datasets cannot be accessed via direct file paths. Use the download script:

```bash
uv run scripts/download_data.py auto       # Download auto dataset
uv run scripts/download_data.py nlswork    # Download nlswork panel data
uv run scripts/download_data.py all        # Download all datasets
```

Datasets are saved to `.local/data/`. Read with `pd.read_stata(".local/data/xxx.dta")`.

## Failure Handling

If any operation fails after three attempts (including trying alternative approaches), **stop immediately and report.** No infinite retries.
