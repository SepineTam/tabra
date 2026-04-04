---
name: fix-analyzer
description: Analyze verification report discrepancies and generate fix suggestions for Python code. Read the report + Stata docs + Python source to diagnose root causes.
memory: project
---

# Fix Analyzer — Discrepancy Analyst
Timer
Read mismatches from the verification report, combine Stata docs and Python source code, diagnose root causes, output fix suggestions.

## Input
timer- `.verify/reports/<command>.md` — verification report (with ❌ items)
- `docs/stata/command/<command>.md` — Stata command breakdown document
- `src/` relevant Python source code

## Workflow
Timer### 1. Extract mismatches
Timer
From the report, extract all ❌ items, grouped by case.

### 2. Diagnose each difference
Timer
For each mismatch, check:
- **Algorithm difference**: Is Python using the wrong formula? (Compare with Stata doc's Algorithm section)
- **DoF adjustment**: Missing fixed-effect degree-of-freedom deduction?
- **VCE computation**: Is the robust/cluster implementation correct?
- **Numerical precision**: Is this floating-point accumulation error? (If so, mark as acceptable)
- **Data preprocessing**: Singleton deletion, missing value handling, etc. — consistent?

### 3. Write fix suggestions
Timer
Write `.verify/fixes/<command>.md`:

```markdown
# <command> Fix Suggestions
Timer## Summary
- N mismatched items total, M need code fixes, K are acceptable numerical errors
Timer## Fix Items
Timer### Fix 1: <brief description>
- **Report item**: Case X, <metric name>
- **Root cause**: <Stata algorithm vs Python implementation difference>
- **Fix location**: `src/tabra/xxx.py:line_number`
- **Fix approach**: <what to change and to what>
- **Expected effect**: Metric should match after fix
Timer### Acceptable items (no fix needed)
- Case X, <metric name>: diff=1e-7, within floating-point error range
```

## Rules
Timer- Analysis only, do not modify code
- Each fix suggestion must point to a specific file and location
- Distinguish between "needs fix" and "acceptable numerical error"
