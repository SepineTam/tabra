---
name: verify-command
description: Cross-validate a Stata command against its Python implementation by running identical parameter combinations in both environments and comparing results. Use when user says "verify command XXX" or "validate XXX" or wants to check that Python output matches Stata output for a specific command.
agents: verify-runner verify-reporter
allowed-tools: Read Write Edit Grep Glob Bash(stata-mcp tool do .verify/codes/*) Bash(stata-mcp tool read-log *) Bash(uv run .verify/codes/*)
---

# Verify Command — Stata vs Python Cross-Validation

Run identical data and parameter combinations in both Stata and Python, compare results item by item.

## Workflow

Given one or more command names (e.g., `reghdfe`, `xtreg`), execute:

### Step 1: Understand the Command

Read `docs/stata/command/<command>.md` to identify:
- All sub-commands / variants (e.g., `xtreg` fe/be/re/mle/pa)
- All key options (e.g., `vce(robust)`, `vce(cluster ...)`, `absorb(...)`)
- What output values to compare — **must cross-check every single line of Stata's raw output, no omissions**

**Comparison Checklist (must cover all)**:

| Category | Specific Metrics |
|----------|-----------------|
| Regression table | Every coef, SE, t/z, p-value, CI (including `_cons`) |
| Goodness of fit | R² (within/between/overall), Adj R², root MSE |
| Variance decomposition | sigma_u, sigma_e, rho (panel models), corr(u_i, Xb) |
| Sample info | N obs, N groups, group variable name, Obs per group (min/avg/max) |
| Test statistics | F/chi2, Prob > F/chi2, DoF (model/resid) |
| Special tests | F test u_i=0 (FE), Hausman (RE vs FE), Breusch-Pagan LM (RE), Wald (BE) |
| Log-likelihood | log-likelihood (if Stata prints it) |
| Other | Any numeric value Stata prints — **not a single one can be missed** |

**Principle: Whatever Stata prints, Python must verify. When in doubt, compare.**

### Step 2: Enumerate Parameter Combinations

List all parameter combinations to verify. Ensure coverage:

| Dimension | Example |
|-----------|---------|
| Sub-commands/variants | `fe`, `be`, `re` |
| VCE type | `conventional`, `robust`, `cluster(var)` |
| Key options | `absorb()` with different dimension combinations |
| Edge cases | No fixed effects, single-dim FE, multi-dim FE |

Output a clear table of all combinations to verify.

### Step 3: Write Verification Files

Write two files to `.verify/codes/` directory:

**`.verify/codes/<command>.do`** — Stata side:
```stata
sysuse auto, clear
* or other suitable dataset

* --- Case 1: basic ---
<command> price weight mpg, <options>
* save key results to scalar/matrix

* --- Case 2: variant_x ---
<command> price weight mpg, <options>
* ...

* finally print all values to compare
```

**`.verify/codes/<command>.py`** — Python side:
```python
import pandas as pd
import tabra as tab

df = pd.read_stata(".local/data/<dataset>.dta")

# --- Case 1: basic ---
result = tab.<command>(...)
print(f"Case 1: coef = {result.coef}, se = {result.se}, ...")

# --- Case 2: variant_x ---
# ...
```

Each case must print the **exact same metrics** for easy comparison.

### Step 4: Run Both via Agent

Use the Agent tool to spawn `verify-runner`:

```
Agent(subagent_type="verify-runner", prompt="Run .verify/codes/<command>.do and .verify/codes/<command>.py, collect raw output")
```

Wait for verify-runner to return raw Stata and Python output.

### Step 5: Compare & Report via Agent

Use the Agent tool to spawn `verify-reporter`:

```
Agent(subagent_type="verify-reporter", prompt="Compare the following Stata and Python outputs item by item, generate report at .verify/reports/<command>.md")
```

Tolerance standards:

**Precision principle: All values must match Stata exactly, zero tolerance.**
- t/z values must use exact distributions (`scipy.stats.t.cdf`/`scipy.stats.norm.cdf`), no 1.96 critical value approximation
- p-values must use exact distribution calculations, no normal approximation for t distribution
- F/chi2 tests, sigma_u/sigma_e/rho, etc. — same rule
- Every value must match Stata digit by digit
- The only acceptable difference is floating-point rounding at the last decimal place (e.g., last digit off by 1), but this is still a bug to fix if possible
- "Close enough" is not acceptable

**Integer metrics must match exactly (zero tolerance):**
- DoF (model / resid)
- N obs
- N groups
- Group variable name

**Floating-point metrics: must match digit by digit.** Only mark ⚠️ when both sides use same-precision FP arithmetic and the last-digit drift is purely due to rounding order — and must note the reason in the report and try to eliminate it.

Report structure:

```markdown
# <command> Verification Report

## Summary

- Verified X parameter combinations, Y metric comparisons total
- ✅ Passed: Z | ❌ Mismatched: W
- **If any mismatches, list them immediately here** (param name + Stata value + Python value + diff)

## Detailed Results

### Case 1: <parameter combination description>
| Metric | Stata | Python | Match? |
|--------|-------|--------|--------|
| coef_weight | 1.234 | 1.234 | ✅ |
| se_weight | 0.567 | 0.568 | ❌ (diff=0.001) |

### Case 2: <parameter combination description>
...
```

**Key**: The summary section must flag all mismatches immediately — don't make the user scroll to find problems.

Report must end with a **Fix Checklist**, listing each mismatch as an item for `fix-from-report` skill:

```markdown
## Fix Checklist (for fix-from-report)

- [ ] Case X: <metric_name> — Stata: <value>, Python: <value>, diff=<diff>
- [ ] Case Y: <metric_name> — ...
```

**Checkmark rule**: If re-verifying (old report exists), only check `[x]` when the item passes (✅) after re-verification. Otherwise keep `[ ]`.

Note: This checklist is written by `verify-reporter`. **Only the verify workflow can update the checklist.**

## File Conventions

- Verification directory: `.verify/`
- Code files: `.verify/codes/<command>.do`, `.verify/codes/<command>.py`
- Report files: `.verify/reports/<command>.md`
- Prefer `sysuse auto` data (download via `uv run .local/data/download.py auto` to `.local/data/`)
- If command requires panel data, use `webuse nlswork` or other suitable dataset

## Anti-Patterns (forbidden)

- **`stata-mcp` is a CLI tool, not an MCP server**: Verify with `stata-mcp --version` or `uvx stata-mcp --version`. All operations via `stata-mcp tool <subcommand>`
- **No `python3 -c`**: Write script files then run. Also forbidden: `python /dev/stdin <<`, `echo | python`, etc.
- **No `requests` for data download**: Use `uv run .local/data/download.py <dataset>`
- **No guessing Stata data paths**: Must use the download script
- **No skipping parameter combinations**: Must cover all reasonable combinations, not just happy path
- **No output redirection**: See Standard Execution below — do not add `> file`, `2>&1`, `| tee`, etc.

## Standard Execution

**Stata**:
```bash
stata-mcp tool do .verify/codes/<command>.do
```
Why: `stata-mcp tool do` automatically generates a `.log` file in `stata-mcp-folder/stata-mcp-log/`. Read logs later with `stata-mcp tool read-log <path>`. Redirection causes output loss or file conflicts.

**Python**:
```bash
uv run .verify/codes/<command>.py
```
Why: `uv run` stdout returns directly to the conversation context. No need to store separately. Redirection is pointless and may cause encoding issues.

Both commands **must use exactly the syntax above** — no extra parameters or shell operators.

**Tests**:
```bash
uv run pytest tests/test_<command>.py -v
```
Run the corresponding test file after verification to ensure nothing is broken.

## Update Index

After verification, **must update** `.verify/INDEX.md`:
- If new command: add a row to the table
- If re-verification: update status (✅ all passed / ❌ mismatches exist)
- Keep table sorted alphabetically

## Example

See [references/](references/) for a complete reghdfe example:
- [references/example_reghdfe_codes.do](references/example_reghdfe_codes.do) — Stata verification code
- [references/example_reghdfe_codes.py](references/example_reghdfe_codes.py) — Python verification code
- [references/example_reghdfe_report.md](references/example_reghdfe_report.md) — Verification report
