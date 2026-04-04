---
name: intuitive-test
description: Black-box usability testing from a user's perspective. Use when you want to test a module/function by writing code as a real user would — no peeking at source code, signatures, or docs. The tester only knows the basic import path and writes code purely from intuition and naming hints.
---

# Intuitive Test — User Intuition Testing

From a real user's perspective — no source code, no signatures, no docs. Write code purely from intuition and naming hints to expose anti-intuitive API designs.

## Core Rules

- **Forbidden to view source code**: Do not read any files under `src/`
- **Forbidden to view function signatures**: Do not read type hints, docstrings
- **Forbidden to view docs**: Do not read `docs/`, README
- **Only allowed to see**: `pyproject.toml` (to know project name and deps), `import` statements (to know module paths)

## Workflow

Given a test target (e.g., `tabra.data.gen`), execute:

### Step 1: Prepare Test Data

Check `.local/data/` for available data. If none:
- Use `uv run .local/data/download.py <dataset>` to download needed Stata data
- Or construct simple data with `pd.DataFrame` in the test script

### Step 2: Write Intuition Test Script

Write test code in `tmp/test_intuitive_<module>.py`.

Follow these guidelines:
- Only know `import tabra as tab` (or project name)
- Guess usage based on **intuition and naming hints**. E.g., seeing `tab.data.gen`, might write:
  - `tab.data.gen(data, "new_var = old_var * 2")`
  - `tab.data.gen(data, new_var=lambda df: df.old_var * 2)`
  - `tab.data.gen(data, {"new_var": "old_var * 2"})`
- Write each intuition attempt in a try-except block
- Print each attempt's result (success/failure/error message)

### Step 3: Run and Record

```bash
uv run tmp/test_intuitive_<module>.py
```

Record:
- Which intuitive usages succeeded
- Which failed, and what errors
- Which intuition feels most "natural"

### Step 4: Write Test Report

Output to `tmp/intuitive_report_<module>.md`:

```markdown
# Intuitive Test Report: <module>

## Target
<!-- what was tested -->

## Attempt Log
| # | Intuitive Usage | Result | Notes |
|---|----------------|--------|-------|
| 1 | `xxx` | success/failure | ... |

## Findings
- <!-- What works well in the API -->
- <!-- What is counter-intuitive -->
- <!-- Improvement suggestions -->
```

## Anti-Patterns

- **No reading source code**: Violates the "user perspective" principle
- **No `python3 -c` / `uv run python -c`**: Write script files then run
- **No `requests` for data download**: Use `uv run .local/data/download.py`
