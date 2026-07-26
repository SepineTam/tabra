---
name: learn-stata-command
description: Learn and document a Stata command for Python econometric implementation reference. Use when user says "learn stata command XXX" or wants to understand a Stata command's internals for replicating in Python. Workflow: fetch help via Stata-MCP, write demo do-file, run it, and produce a command breakdown doc.
context: fork
agent: stata-dev
disable-model-invocation: true
allowed-tools: Read Write Edit Grep Glob WebSearch WebFetch Bash(stata-mcp tool*)
---

# Learn Stata Command

Systematic workflow to study a Stata command and produce reference docs for Python implementation.

## Prerequisites

- Stata-MCP tool must be available. If absent, stop and ask the user.
- Project directories: `stata-mcp-folder/stata-mcp-tmp/` (help cache), `stata-mcp-folder/stata-mcp-dofile/` (do-files), `stata-mcp-folder/stata-mcp-log/` (logs), `docs/stata/command/` (breakdown docs).

## Workflow

Given a command name (e.g., `xtreg`, `reghdfe`, `summarize`), execute these steps in order.

### Step 1: Fetch Help

Fetch help via Stata-MCP:

```bash
stata-mcp tool help <command>
```

This writes `help__<command>.txt` to `stata-mcp-folder/stata-mcp-tmp/`. Read it to understand:
- Syntax and sub-commands (e.g., `xtreg ... , fe / be / re / mle / pa`)
- Options and their meanings
- Key formulas / methods described

### Step 2: Fetch Ado Source Code

Many Stata commands (especially user-contributed ones like `reghdfe`) have viewable `.ado` source files. Use the script to fetch:

```bash
uv run scripts/fetch_ado.py <command>
```

The script uses `findfile` + `cp` to copy the ado file to `.verify/ado_plus/<command>.ado`.

If the script errors (command may be `built-in`), it means the command is compiled C code — skip this step.
If successful, Read `.verify/ado_plus/<command>.ado` to understand:
- Internal algorithm flow (e.g., MAP iteration, FE absorption details)
- Option processing logic
- Intermediate variable computation

This information is critical for Python replication — help docs are rarely detailed enough.

### Step 3: Write Demo Do-File(s)

Based on the help output, write one or more do-files to `stata-mcp-folder/stata-mcp-dofile/`. Each do-file tests one sub-command or major option variant.

Do-file conventions:
- Use `sysuse auto, clear` as the dataset unless the command requires panel data (then use a panel setup like `webuse nlswork, clear` + `xtset`).
- Filename pattern: `<command>_<variant>_demo.do` (e.g., `xtreg_fe_demo.do`, `xtreg_re_demo.do`).
- Keep each do-file minimal: load data, set up if needed, run the command.

### Step 4: Run Do-Files

Execute each do-file via Stata-MCP:

```bash
stata-mcp tool do stata-mcp-folder/stata-mcp-dofile/<filename>.do
```

Read the log output to verify the command ran correctly and note the output format.

### Step 5: Write Command Breakdown Doc

Create `docs/stata/command/<command>.md` with the following sections:

```markdown
# <command> -- <brief description>

## Syntax
<!-- from help output -->

## Sub-commands / Variants
<!-- list each variant with its purpose -->

## Key Options
<!-- most important options -->

## Algorithm / Estimation Method
<!-- the math under the hood, from help or Stata manual -->

## Output Elements (Complete Output Checklist)

<!-- List every single numeric item in Stata's output, one per line, no omissions -->
<!-- Reference the actual log output from Step 4 to fill this in -->

### Header section
- Number of obs
- Group variable / Panel variable
- Number of groups
- ... (anything Stata prints in the header)

### Regression table
- Every coef, SE, t/z, p-value, CI (including `_cons`)

### Goodness of fit
- R² (within/between/overall), Adj R², root MSE

### Variance decomposition (panel models)
- sigma_u, sigma_e, rho
- corr(u_i, Xb) (if present)

### Test statistics
- F/chi2, Prob > F/chi2
- F test u_i=0 (FE), Hausman (RE vs FE), Breusch-Pagan LM (RE)
- Any other tests Stata prints

### Other
- log-likelihood (if present)
- Any numeric value in Stata's output

## Implementation Notes
<!-- hints for Python replication -->
```

**Critical**: The `Output Elements` section must be extracted from **actual log output in Step 4** — not from memory. Whatever Stata prints, list it. Missing any item will cause missed detections during verification.

## Example

See [references/example_reghdfe.md](references/example_reghdfe.md) for a complete breakdown of `reghdfe`. Use it as the template for future command breakdown docs.

## Important Notes

- Do NOT ask the user for Stata code — use Stata-MCP exclusively.
- If a command has many variants (like `xtreg`), cover each significant variant with its own do-file.
- The breakdown doc should be concise. Target implementation-relevant details, not a copy of the help file.

## Anti-Patterns (forbidden)

- **No `python3 -c "xxx"`**: Write script files then run with `uv run`.
- **No `requests.get()` for data download**: Stata datasets use `uv run scripts/download_data.py <dataset>` to download to `.local/data/`. Read with `pd.read_stata(".local/data/<dataset>.dta")`.
- **No guessing Stata data paths**: Cannot directly access `sysuse` / `webuse` file paths — must use the download script.

## Stata-MCP Available Tools

**`stata-mcp` is a CLI tool, not an MCP server.** Verify with: `stata-mcp --version` or `uvx stata-mcp --version`.

```
stata-mcp tool help <command>    # View command help, cached to stata-mcp-folder/stata-mcp-tmp/
stata-mcp tool do <file.do>      # Execute do-file, logs saved to stata-mcp-folder/stata-mcp-log/
stata-mcp tool data-info <path>  # View dataset metadata (variable names, types, etc.)
stata-mcp tool read-log <path>   # Read Stata log
stata-mcp tool ado-install <pkg> # Install ado package
```
