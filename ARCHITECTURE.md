# Tabra Architecture

Tabra is one Python package with strong internal module boundaries. The public
workflow in `README.md` is the compatibility contract: data is loaded into one
analysis session and commands are reached through `tab.data`, `tab.est`, and
`tab.plot`.

## Command flow

Every result-producing command follows the same sequence:

1. The public accessor validates command-specific state and builds a Stata-like
   command label.
2. A computational model receives a DataFrame and explicit arguments.
3. `TabraData._finalize_result()` applies style and display settings, records the
   model and result, and appends the command to `tab.results.history`.
4. Result objects handle rendering, saving, and result-specific operations.

Plotting is a consumer of data and results. Estimators must not depend on the
plotting implementation.

## Module boundaries

```text
src/tabra/
├── api/estimation/       # Public tab.est commands grouped by model family
├── core/                 # Analysis session, configuration, result lifecycle
├── data/                 # Reusable data-processing components
├── models/estimate/      # Numerical estimator implementations
├── ops/                  # Low-level statistical and linear algebra operations
├── results/              # Result containers and display contracts
├── plot/                 # Plotting commands and figure wrappers
└── io/                   # Data import and export
```

New estimation commands belong to one of the domain accessors:

- `linear.py`: OLS, HDFE, and quantile models;
- `panel.py`: panel estimators;
- `discrete.py`: binary, ordered, multinomial, and generalized linear models;
- `limited.py`: censoring and sample-selection models;
- `instrumental.py`: linear and nonlinear instrumental-variable models.

Causal estimators will use a separate `tab.causal` accessor when the first
method is implemented. Difference-in-differences, event studies, regression
discontinuity, matching, and synthetic control should not be added to the
general `tab.est` accessor.

## Decisions and trade-offs

### One distribution, modular internals

Tabra remains one installable package. Multiple distributions would add release
and dependency-management costs before the contributor base needs them.

### Preserve the README API

Internal modules may move, but calls documented in `README.md` remain stable.
Compatibility imports should be retained when a previously importable internal
path is moved.

### One result lifecycle

All command result classes inherit `BaseResult`. Model-specific statistics may
remain specialized, while style, command display, saving, and session history
use a common contract.

### Restricted data expressions

Data expressions are parsed by `ExpressionEngine`; arbitrary Python evaluation
is not allowed. This supports common arithmetic and approved math functions
while preventing expressions from executing unrelated code.

### Numerical accuracy before abstraction

Shared covariance, optimization, and linear-algebra policies should move into
`inference/` and `ops/` only when existing Stata parity tests prove that the
extraction does not change results.

## Adding a command

A new command should require:

1. one domain accessor method;
2. one estimator module or a focused extension of an existing family;
3. one `BaseResult` implementation or an existing compatible result type;
4. unit tests, Stata parity fixtures, README or API documentation;
5. a full test-suite pass.

Command code must not duplicate result storage, style application, or display
control; those responsibilities belong to the shared session lifecycle.
