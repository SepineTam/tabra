#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : coefplot.py

"""Coefficient plot for regression results."""

import numpy as np
from scipy import stats

from tabra.core.errors import (
    InvalidLevelError,
    NoCommonVarsError,
    NoResultError,
    ResultTypeError,
)


def _resolve_template(template):
    """Resolve template, falling back to DEFAULT."""
    if template is not None:
        return template
    from tabra.plot.templates import DEFAULT
    return DEFAULT


def _model_offsets(n_models, m_idx):
    """Compute offset for multi-model positioning."""
    if n_models == 1:
        return 0.0
    total = 0.6
    step = total / n_models
    return -total / 2 + step * (m_idx + 0.5)


def _compute_ci(coef, se, level=0.95, df=None):
    """Compute confidence interval bounds.

    Args:
        coef: Coefficient estimate.
        se: Standard error.
        level: Confidence level (0-1).
        df: Degrees of freedom for t-distribution. None = use z.

    Returns:
        tuple: (ci_lo, ci_hi)
    """
    if df is not None and df < 120:
        crit = stats.t.ppf((1 + level) / 2, df)
    else:
        crit = stats.norm.ppf((1 + level) / 2)
    return coef - crit * se, coef + crit * se


def _is_multi_equation(result):
    """Check if result has multiple equations."""
    return (isinstance(getattr(result, "coef", None), dict)
            or hasattr(result, "outcome_coef"))


def _extract_coefs(result, level=0.95):
    """Extract coefficient data from a result object.

    Args:
        result: A tabra result object.
        level: Confidence level.

    Returns:
        list[dict]: Each dict has 'label' (str) and 'items' (list[dict]).
            Each item: {'name': str, 'coef': float, 'ci_lo': float, 'ci_hi': float}.
    """
    if _is_multi_equation(result):
        return _extract_multi(result, level)

    coef_arr = np.asarray(result.coef)
    se_arr = np.asarray(result.std_err)
    names = list(result.var_names)
    df = getattr(result, "df_resid", None)

    items = []
    for i, name in enumerate(names):
        ci_lo, ci_hi = _compute_ci(coef_arr[i], se_arr[i], level, df)
        items.append({
            "name": name,
            "coef": float(coef_arr[i]),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
        })
    return [{"label": "coefficients", "items": items}]


def _extract_multi(result, level=0.95):
    """Extract coefficients from multi-equation results."""
    # MLogit: coef is dict {category: array}
    if hasattr(result, "categories"):
        series_list = []
        for cat in result.categories:
            coef_arr = np.asarray(result.coef[cat])
            se_arr = np.asarray(result.std_err[cat])
            names = list(result.var_names)
            items = []
            for i, name in enumerate(names):
                ci_lo, ci_hi = _compute_ci(coef_arr[i], se_arr[i], level)
                items.append({
                    "name": name,
                    "coef": float(coef_arr[i]),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                })
            series_list.append({"label": str(cat), "items": items})
        return series_list

    # Heckman: separate outcome and selection
    if hasattr(result, "outcome_coef"):
        series_list = []
        for eq_label, coef_attr, se_attr, names_attr in [
            ("outcome", "outcome_coef", "outcome_se", "outcome_var_names"),
            ("selection", "select_coef", "select_se", "select_var_names"),
        ]:
            coef_arr = np.asarray(getattr(result, coef_attr))
            se_arr = np.asarray(getattr(result, se_attr))
            names = list(getattr(result, names_attr))
            items = []
            for i, name in enumerate(names):
                ci_lo, ci_hi = _compute_ci(coef_arr[i], se_arr[i], level)
                items.append({
                    "name": name,
                    "coef": float(coef_arr[i]),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                })
            series_list.append({"label": eq_label, "items": items})
        return series_list

    raise ResultTypeError(
        f"Unsupported result type: {type(result).__name__}"
    )


def _apply_filter(items, keep=None, drop=None, sort=None, rename=None):
    """Filter, sort, and rename coefficient items.

    Args:
        items: list[dict] with 'name', 'coef', 'ci_lo', 'ci_hi'.
        keep: Only keep these variable names.
        drop: Remove these variable names.
        sort: 'ascending', 'descending', or None.
        rename: dict mapping old names to new names.

    Returns:
        list[dict]: Filtered items.
    """
    result = list(items)

    if keep is not None:
        keep_set = set(keep)
        result = [it for it in result if it["name"] in keep_set]

    if drop is not None:
        drop_set = set(drop)
        result = [it for it in result if it["name"] not in drop_set]

    if sort == "ascending":
        result.sort(key=lambda it: it["coef"])
    elif sort == "descending":
        result.sort(key=lambda it: it["coef"], reverse=True)

    if rename is not None:
        for it in result:
            if it["name"] in rename:
                it["name"] = rename[it["name"]]

    return result


def _render_horizontal(ax, series_list, labels, xline, ci_style, tmpl):
    """Render horizontal coefplot (variable names on Y-axis).

    Args:
        ax: matplotlib Axes.
        series_list: list of list[dict] (one per model).
        labels: Model labels for legend.
        xline: Reference line x position.
        ci_style: 'spike' or 'area'.
        tmpl: PlotTemplateBase instance.
    """
    colors = list(tmpl.color_cycle)
    n_models = len(series_list)
    all_names = [it["name"] for it in series_list[0]]
    n_vars = len(all_names)
    y_pos = list(range(n_vars))

    if xline is not None:
        ax.axvline(x=xline, color="gray", linewidth=0.5, linestyle="--")

    for m_idx, items in enumerate(series_list):
        color = colors[m_idx % len(colors)]
        label = labels[m_idx] if labels and m_idx < len(labels) else None
        offset = _model_offsets(n_models, m_idx)
        ys = [y + offset for y in y_pos]
        coefs = [it["coef"] for it in items]
        ci_los = [it["ci_lo"] for it in items]
        ci_his = [it["ci_hi"] for it in items]

        if ci_style == "spike":
            for i in range(n_vars):
                ax.plot([ci_los[i], ci_his[i]], [ys[i], ys[i]],
                        color=color, linewidth=1.0)
        else:
            height = 0.6 / n_models if n_models > 1 else 0.5
            for i in range(n_vars):
                ax.barh(ys[i], ci_his[i] - ci_los[i],
                        left=ci_los[i], height=height,
                        color=color, alpha=0.25)

        ax.scatter(coefs, ys, color=color, s=tmpl.marker_size,
                   zorder=3, label=label)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_names)
    ax.invert_yaxis()

    if labels and n_models > 1:
        ax.legend(fontsize=tmpl.legend_size)


def _render_vertical(ax, series_list, labels, xline, ci_style, tmpl):
    """Render vertical coefplot (variable names on X-axis)."""
    colors = list(tmpl.color_cycle)
    n_models = len(series_list)
    all_names = [it["name"] for it in series_list[0]]
    n_vars = len(all_names)
    x_pos = list(range(n_vars))

    if xline is not None:
        ax.axhline(y=xline, color="gray", linewidth=0.5, linestyle="--")

    for m_idx, items in enumerate(series_list):
        color = colors[m_idx % len(colors)]
        label = labels[m_idx] if labels and m_idx < len(labels) else None
        offset = _model_offsets(n_models, m_idx)
        xs = [x + offset for x in x_pos]
        coefs = [it["coef"] for it in items]
        ci_los = [it["ci_lo"] for it in items]
        ci_his = [it["ci_hi"] for it in items]

        if ci_style == "spike":
            for i in range(n_vars):
                ax.plot([xs[i], xs[i]], [ci_los[i], ci_his[i]],
                        color=color, linewidth=1.0)
        else:
            width = 0.6 / n_models if n_models > 1 else 0.5
            for i in range(n_vars):
                ax.bar(xs[i], ci_his[i] - ci_los[i],
                       bottom=ci_los[i], width=width,
                       color=color, alpha=0.25)

        ax.scatter(xs, coefs, color=color, s=tmpl.marker_size,
                   zorder=3, label=label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_names, rotation=45, ha="right")

    if labels and n_models > 1:
        ax.legend(fontsize=tmpl.legend_size)


def plot_coefplot(result_or_list=None, *,
                  keep=None, drop=None,
                  sort=None, rename=None,
                  vertical=False, xline=0, level=0.95,
                  ci_style='spike',
                  labels=None,
                  title=None, xtitle=None, ytitle=None,
                  template=None):
    """Plot regression coefficients with confidence intervals.

    Args:
        result_or_list: A result object, list of result objects, or None.
        keep: Only plot these variable names.
        drop: Exclude these variable names.
        sort: 'ascending', 'descending', or None.
        rename: Dict mapping old names to new names.
        vertical: If True, use vertical layout.
        xline: Reference line position (default 0).
        level: Confidence level (0-1).
        ci_style: 'spike' (line segments) or 'area' (shaded bars).
        labels: Model labels for multi-model comparison.
        title: Plot title.
        xtitle: X-axis label.
        ytitle: Y-axis label.
        template: PlotTemplateBase instance.

    Returns:
        TabraFigure: Wrapped matplotlib figure.

    Raises:
        NoResultError: If result is None.
        InvalidLevelError: If level is not in (0, 1).
        NoCommonVarsError: If multi-model has no common variables.
    """
    import matplotlib.pyplot as plt
    from tabra.plot import TabraFigure

    if result_or_list is None:
        raise NoResultError(
            "Run an estimation first before calling coefplot()"
        )

    if not (0 < level < 1):
        raise InvalidLevelError(
            f"level must be between 0 and 1, got {level}"
        )

    # Normalize to list
    if not isinstance(result_or_list, list):
        result_or_list = [result_or_list]

    tmpl = _resolve_template(template)
    tmpl.apply()

    # Extract coefficients for each result
    all_series = []
    for r in result_or_list:
        extracted = _extract_coefs(r, level=level)
        all_series.append(extracted)

    # Compute common var names across models for each equation slot
    n_eqs = len(all_series[0])
    groups = []
    for eq_idx in range(n_eqs):
        # Collect items from each model for this equation
        model_items = []
        common_names = None
        for model_series in all_series:
            items = _apply_filter(
                model_series[eq_idx]["items"],
                keep=keep, drop=drop, sort=sort, rename=rename,
            )
            names = set(it["name"] for it in items)
            if common_names is None:
                common_names = names
            else:
                common_names = common_names & names
            model_items.append(items)

        if common_names is not None and len(common_names) == 0 and len(result_or_list) > 1:
            raise NoCommonVarsError(
                "Models share no common variables for comparison"
            )

        # Filter to common names for multi-model
        if len(result_or_list) > 1 and common_names is not None:
            model_items = [
                [it for it in items if it["name"] in common_names]
                for items in model_items
            ]

        eq_label = all_series[0][eq_idx]["label"]
        groups.append({"label": eq_label, "model_items": model_items})

    # Create subplots
    n_subplots = len(groups)
    fig, axes = plt.subplots(
        1, n_subplots,
        figsize=(tmpl.fig_width * max(n_subplots, 1), tmpl.fig_height),
        dpi=tmpl.dpi,
    )
    if n_subplots == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        group = groups[idx]
        if vertical:
            _render_vertical(ax, group["model_items"], labels,
                             xline, ci_style, tmpl)
        else:
            _render_horizontal(ax, group["model_items"], labels,
                               xline, ci_style, tmpl)

        if title and n_subplots == 1:
            ax.set_title(title, fontsize=tmpl.title_size)
        elif n_subplots > 1:
            ax.set_title(group["label"], fontsize=tmpl.title_size)

        if not vertical:
            if xtitle:
                ax.set_xlabel(xtitle, fontsize=tmpl.label_size)
            if ytitle:
                ax.set_ylabel(ytitle, fontsize=tmpl.label_size)
        else:
            if xtitle:
                ax.set_xlabel(xtitle, fontsize=tmpl.label_size)
            if ytitle:
                ax.set_ylabel(ytitle, fontsize=tmpl.label_size)

        # Spine settings
        if not tmpl.spine_top:
            ax.spines["top"].set_visible(False)
        if not tmpl.spine_right:
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return TabraFigure(fig)
