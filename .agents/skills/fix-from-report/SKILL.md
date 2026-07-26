---
name: fix-from-report
description: Fix Python implementation issues identified by verify-command. Reads the verification report, analyzes discrepancies, applies fixes, and re-runs verification. Use when a verify-command report shows mismatches, or user says "fix report for XXX" or "patch XXX based on report".
agents: fix-analyzer fix-coder verify-runner verify-reporter
allowed-tools: Read Write Edit Grep Glob Bash(stata-mcp tool do .verify/codes/*) Bash(stata-mcp tool read-log *) Bash(uv run .verify/codes/*) Bash(uv run pytest tests/test_* -v)
---

# Fix From Report — Verification-Report-Driven Fix Workflow

Read mismatches from `.verify/reports/<command>.md`, analyze root causes, fix Python code, then re-verify.

## Workflow

### Step 1: Read Report

Read `.verify/reports/<command>.md`, confirm all ❌ items.

If report does not exist or all ✅ → stop, nothing to fix.

### Step 2: Analyze (fix-analyzer)

Use Agent tool to spawn `fix-analyzer`:

```
Agent(subagent_type="fix-analyzer", prompt="Read mismatches from .verify/reports/<command>.md, combine with docs/stata/command/<command>.md and src/ source code, analyze root causes and output fix suggestions to .verify/fixes/<command>.md")
```

Wait for fix-analyzer to complete the fix suggestion document.

### Step 3: Fix (fix-coder)

Use Agent tool to spawn `fix-coder`:

```
Agent(subagent_type="fix-coder", prompt="Read fix suggestions from .verify/fixes/<command>.md, modify corresponding code in src/. Only change what needs changing, no extra refactoring.")
```

Wait for fix-coder to complete code modifications.

### Step 4: Re-verify

Spawn two agents in sequence:

```
Agent(subagent_type="verify-runner", prompt="Run .verify/codes/<command>.do and .verify/codes/<command>.py, collect raw output")
```

```
Agent(subagent_type="verify-reporter", prompt="Compare the following Stata and Python outputs item by item, generate new report overwriting .verify/reports/<command>.md")
```

### Step 5: Run Tests

```bash
uv run pytest tests/test_<command>.py -v
```

Ensure fixes did not break existing tests.

### Step 6: Report

Summarize:
- What was fixed (reference fix suggestions)
- New report results (all ✅ or not)
- Whether tests pass

## File Conventions

- Fix suggestions: `.verify/fixes/<command>.md`
- Re-verification report: `.verify/reports/<command>.md` (overwrites)

## Anti-Patterns

- **`stata-mcp` is a CLI tool, not an MCP server**: Verify with `stata-mcp --version` or `uvx stata-mcp --version`. All operations via `stata-mcp tool <subcommand>`
- **No `python3 -c`**: Write script files then run. Also forbidden: `python /dev/stdin <<`, `echo | python`, etc.
- **No over-modification**: Only fix mismatches flagged in the report — no refactoring, no new features
- **No approximate computation**: t-values cannot use 1.96 critical value approximation. p-values cannot use normal approximation instead of t distribution. Must use exact distributions (`scipy.stats.t`, `scipy.stats.chi2`, `scipy.stats.f`, etc.). All values must match Stata exactly, zero tolerance. Only acceptable difference is last-digit FP rounding drift, but should be minimized.
- **No skipping re-verification**: Must re-run verification after fixing
- **No skipping tests**: Must run the corresponding single test file after fixing
- **Do not touch Fix Checklist**: The `## Fix Checklist` section in reports is written by the verify workflow. The fix workflow must **never modify it under any circumstances** — no reading, checking, adding, deleting, or editing.
- **No output redirection**: Only use standard execution commands — no `> file`, `2>&1`, `| tee`, etc. (see verify-command skill's "Standard Execution")

## Example

See [references/](references/) for a complete reghdfe example:
- [references/example_reghdfe_fix.md](references/example_reghdfe_fix.md) — Fix suggestion document (with root cause analysis, code examples, priority ordering)
