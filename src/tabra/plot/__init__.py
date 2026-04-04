#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabra.plot.fig_setting import PlotKind


class TabraFigure:
    """Wrapper around matplotlib Figure with user-friendly API."""

    def __init__(self, fig, tabra=None):
        self._fig = fig
        self._tabra = tabra

    def save(self, filename: str, dpi: int = 300,
             formats: list = None, **kwargs):
        """Save the figure to one or more files.

        Args:
            filename (str): Output file path. If no extension and formats
                is None, defaults to ``.png``.
            dpi (int): Resolution in dots per inch. Default 300.
            formats (list): List of format extensions, e.g. ``["png", "pdf"]``.
                If filename has no extension, replaces the implicit default.
                If filename has an extension and formats is given, saves as
                ``name.ext.fmt`` for each format.
            **kwargs: Additional keyword arguments passed to ``savefig``.

        Returns:
            TabraFigure: Returns self for method chaining.
        """
        paths = self._resolve_paths(filename, formats)
        for p in paths:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            self._fig.savefig(p, dpi=dpi, bbox_inches="tight", **kwargs)
        return self

    def _resolve_paths(self, filename, formats):
        """Resolve filename into one or more output paths."""
        # Resolve save_base: instance config > global > None
        base = None
        if self._tabra is not None:
            base = getattr(self._tabra._config, "_figure_save_base", None)
        if base is None:
            from tabra.core.config import _global_figure_save_base
            base = _global_figure_save_base

        p = Path(filename)
        is_abs = p.is_absolute()
        if base and not is_abs:
            p = Path(base) / p

        suffix = p.suffix  # e.g. ".png" or ""

        if formats is None:
            # No formats specified
            if suffix == "":
                # No extension → default to .png
                return [str(p.with_suffix(".png"))]
            else:
                return [str(p)]
        else:
            # Formats specified
            if suffix == "":
                # No extension → generate one file per format
                return [str(p.with_suffix(f".{fmt}")) for fmt in formats]
            else:
                # Has extension → name.ext.fmt for each format
                return [str(p).rstrip(suffix) + suffix + f".{fmt}"
                        if fmt != suffix.lstrip(".")
                        else str(p)
                        for fmt in formats]

    def show(self):
        """Display the figure.

        Returns:
            TabraFigure: Returns self for method chaining.
        """
        plt.show()
        return self

    def close(self):
        """Close the figure and free memory.

        Returns:
            TabraFigure: Returns self for method chaining.
        """
        plt.close(self._fig)
        return self

    @property
    def figure(self):
        return self._fig


class PlotOps:
    def __init__(self, tabra):
        self._tabra = tabra

    @property
    def _df(self):
        return self._tabra._df

    def scatter(self, y: str, x: str, title: str = None,
                xtitle: str = None, ytitle: str = None,
                template=None,
                fig_setting=None):
        """Draw a scatter plot.

        Args:
            y (str): Variable name for the y-axis.
            x (str): Variable name for the x-axis.
            title (str): Plot title.
            xtitle (str): X-axis label. Defaults to x variable name.
            ytitle (str): Y-axis label. Defaults to y variable name.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        ax.scatter(self._df[x], self._df[y],
                   s=template.marker_size, c=template.primary_color)
        ax.set_xlabel(xtitle if xtitle is not None else x)
        ax.set_ylabel(ytitle if ytitle is not None else y)
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def _apply_template(self, template, ax):
        """Apply spine and style settings to an axes."""
        if not template.spine_top:
            ax.spines["top"].set_visible(False)
        if not template.spine_right:
            ax.spines["right"].set_visible(False)

    def _make_fig(self, template):
        """Create a figure with template sizing."""
        fig, ax = plt.subplots(
            figsize=(template.fig_width, template.fig_height),
            dpi=template.dpi,
        )
        return fig, ax

    def hist(self, var: str, bins: int = 30, title: str = None,
             xtitle: str = None, ytitle: str = None,
             template=None, density: bool = False,
             fig_setting=None):
        """Draw a histogram.

        Args:
            var (str): Variable name to plot.
            bins (int): Number of bins. Default 30.
            title (str): Plot title.
            xtitle (str): X-axis label. Defaults to var name.
            ytitle (str): Y-axis label. Defaults to "Frequency" or "Density".
            template (PlotTemplate): Plot template to use.
            density (bool): If True, normalize to density. Default False.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        ax.hist(self._df[var].dropna(), bins=bins, density=density,
                color=template.primary_color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(xtitle if xtitle is not None else var)
        ax.set_ylabel(ytitle if ytitle is not None else ("Density" if density else "Frequency"))
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def bar(self, var: str, by: str = None, stat: str = "mean",
            title: str = None, xtitle: str = None, ytitle: str = None,
            template=None, fig_setting=None):
        """Draw a bar plot.

        If ``by`` is None, plots value counts of ``var``.
        If ``by`` is given, plots ``stat`` (mean/sum/count) of ``var``
        grouped by ``by``.

        Args:
            var (str): Variable name to plot.
            by (str): Grouping variable. If None, plots value counts.
            stat (str): Aggregation stat when by is given: "mean", "sum", or
                "count". Default "mean".
            title (str): Plot title.
            xtitle (str): X-axis label.
            ytitle (str): Y-axis label.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)

        if by is not None:
            agg = self._df.groupby(by)[var].agg(stat).sort_values()
            ax.bar(agg.index.astype(str), agg.values,
                   color=template.primary_color, edgecolor="white", linewidth=0.5)
            ax.set_xlabel(xtitle if xtitle is not None else by)
            ax.set_ylabel(ytitle if ytitle is not None else f"{stat} of {var}")
        else:
            counts = self._df[var].value_counts().sort_index()
            ax.bar(counts.index.astype(str), counts.values,
                   color=template.primary_color, edgecolor="white", linewidth=0.5)
            ax.set_xlabel(xtitle if xtitle is not None else var)
            ax.set_ylabel(ytitle if ytitle is not None else "Count")

        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def lfit(self, y: str, x: str, title: str = None,
             xtitle: str = None, ytitle: str = None,
             template=None, fig_setting=None):
        """Draw a linear fit plot (OLS regression line).

        Args:
            y (str): Variable name for the y-axis.
            x (str): Variable name for the x-axis.
            title (str): Plot title.
            xtitle (str): X-axis label. Defaults to x variable name.
            ytitle (str): Y-axis label. Defaults to y variable name.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)

        xv = self._df[x].dropna().values.astype(float)
        yv = self._df[y].dropna().values.astype(float)
        mask = self._df[x].dropna().index.intersection(self._df[y].dropna().index)
        xv = self._df.loc[mask, x].values.astype(float)
        yv = self._df.loc[mask, y].values.astype(float)

        x_mean = xv.mean()
        ssx = ((xv - x_mean) ** 2).sum()
        slope = ((xv - x_mean) * (yv - yv.mean())).sum() / ssx
        intercept = yv.mean() - slope * x_mean

        order = np.argsort(xv)
        xs = xv[order]
        ys = intercept + slope * xs
        ax.plot(xs, ys, color=template.primary_color)

        ax.set_xlabel(xtitle if xtitle is not None else x)
        ax.set_ylabel(ytitle if ytitle is not None else y)
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def lfitci(self, y: str, x: str, title: str = None,
               xtitle: str = None, ytitle: str = None,
               level: float = 0.95,
               template=None, fig_setting=None):
        """Draw a linear fit plot with confidence interval.

        Args:
            y (str): Variable name for the y-axis.
            x (str): Variable name for the x-axis.
            title (str): Plot title.
            xtitle (str): X-axis label. Defaults to x variable name.
            ytitle (str): Y-axis label. Defaults to y variable name.
            level (float): Confidence level. Default 0.95.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        from scipy.stats import t as t_dist

        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)

        mask = self._df[x].dropna().index.intersection(self._df[y].dropna().index)
        xv = self._df.loc[mask, x].values.astype(float)
        yv = self._df.loc[mask, y].values.astype(float)

        n = len(xv)
        x_mean = xv.mean()
        ssx = ((xv - x_mean) ** 2).sum()
        slope = ((xv - x_mean) * (yv - yv.mean())).sum() / ssx
        intercept = yv.mean() - slope * x_mean

        order = np.argsort(xv)
        xs = xv[order]
        ys = intercept + slope * xs

        y_hat = intercept + slope * xv
        mse = ((yv - y_hat) ** 2).sum() / max(n - 2, 1)
        se = np.sqrt(mse * (1 / n + (xs - x_mean) ** 2 / ssx))
        t_val = t_dist.ppf((1 + level) / 2, df=max(n - 2, 1))

        ax.fill_between(xs, ys - t_val * se, ys + t_val * se,
                        color=template.primary_color, alpha=0.15)
        ax.plot(xs, ys, color=template.primary_color)

        ax.set_xlabel(xtitle if xtitle is not None else x)
        ax.set_ylabel(ytitle if ytitle is not None else y)
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def line(self, y: str, x: str = None, by: str = None,
             title: str = None, xtitle: str = None, ytitle: str = None,
             template=None, fig_setting=None):
        """Draw a line plot.

        If ``x`` is None, uses the DataFrame index as x-axis.
        If ``by`` is given, draws one line per group.

        Args:
            y (str): Variable name for the y-axis.
            x (str): Variable name for the x-axis. Defaults to index.
            by (str): Grouping variable. If None, draws a single line.
            title (str): Plot title.
            xtitle (str): X-axis label.
            ytitle (str): Y-axis label. Defaults to y variable name.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        colors = list(template.color_cycle)

        if by is not None:
            groups = self._df[by].unique()
            for i, g in enumerate(sorted(groups)):
                mask = self._df[by] == g
                x_data = self._df.loc[mask, x] if x else self._df.loc[mask].index
                y_data = self._df.loc[mask, y]
                ax.plot(x_data, y_data, color=colors[i % len(colors)], label=g)
            ax.legend(fontsize=template.legend_size)
            ax.set_xlabel(xtitle if xtitle is not None else (x if x else "Index"))
        else:
            x_data = self._df[x] if x else self._df.index
            ax.plot(x_data, self._df[y], color=template.primary_color)
            ax.set_xlabel(xtitle if xtitle is not None else (x if x else "Index"))

        ax.set_ylabel(ytitle if ytitle is not None else y)
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def pie(self, var: str, title: str = None,
             template=None, fig_setting=None):
        """Draw a pie chart of value counts for a variable.

        Args:
            var (str): Variable name to plot.
            title (str): Plot title.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        colors = list(template.color_cycle)

        counts = self._df[var].value_counts()
        ax.pie(counts.values, labels=counts.index.astype(str),
               colors=colors[:len(counts)], autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": template.tick_size})
        ax.set_aspect("equal")
        if title is not None:
            ax.set_title(title)
        return TabraFigure(fig, tabra=self._tabra)

    def violin(self, var: str, by: str = None, title: str = None,
               xtitle: str = None, ytitle: str = None,
               template=None, fig_setting=None):
        """Draw a violin plot.

        Args:
            var (str): Variable name to plot.
            by (str): Grouping variable. If None, draws a single violin.
            title (str): Plot title.
            xtitle (str): X-axis label.
            ytitle (str): Y-axis label. Defaults to var name.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        import seaborn as sns

        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)

        data = self._df[[var] + ([by] if by else [])].dropna()

        if by is not None:
            sns.violinplot(data=data, x=by, y=var, ax=ax,
                           color=template.primary_color, alpha=0.7)
            ax.set_xlabel(xtitle if xtitle is not None else by)
        else:
            sns.violinplot(data=data, y=var, ax=ax,
                           color=template.primary_color, alpha=0.7)
            ax.set_xlabel(xtitle)

        ax.set_ylabel(ytitle if ytitle is not None else var)
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def box(self, var, by: str = None, title: str = None,
               xtitle: str = None, ytitle: str = None,
               template=None, fig_setting=None):
        """Draw a box plot.

        Args:
            var (str | list[str]): Variable name(s) to plot.
            by (str): Grouping variable. If None, draws without grouping.
            title (str): Plot title.
            xtitle (str): X-axis label.
            ytitle (str): Y-axis label.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        colors = list(template.color_cycle)

        vars_list = [var] if isinstance(var, str) else var

        if by is None:
            # No grouping: side by side boxes for each var
            data = [self._df[v].dropna().values for v in vars_list]
            bp = ax.boxplot(data, labels=vars_list, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
        else:
            # Grouped by 'by'
            groups = sorted(self._df[by].dropna().unique())
            if len(vars_list) == 1:
                all_data = [self._df[self._df[by] == g][vars_list[0]].dropna().values
                              for g in groups]
                bp = ax.boxplot(all_data,
                                labels=[str(g) for g in groups],
                                patch_artist=True)
                for patch, color in zip(bp['boxes'], colors[:len(groups)]):
                    patch.set_facecolor(color)
            else:
                # Multiple vars + grouping: nested
                all_data = []
                tick_labels = []
                positions = []
                pos = 1
                for g in groups:
                    for v in vars_list:
                        subset = self._df[self._df[by] == g]
                        all_data.append(subset[v].dropna().values)
                        tick_labels.append(str(g))
                        positions.append(pos)
                        pos += 1
                    pos += 0.5
                bp = ax.boxplot(all_data, labels=tick_labels,
                                positions=positions, patch_artist=True,
                                widths=0.6)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i % len(vars_list)])

        ax.set_xlabel(xtitle if xtitle is not None else (by if by is not None else ", ".join(vars_list)))
        ax.set_ylabel(ytitle if ytitle is not None else ", ".join(vars_list))
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def mix(self, layers: list, title: str = None,
            xtitle: str = None, ytitle: str = None,
            template=None, fig_setting=None):
        """Overlay multiple plot layers on a single axes (like Stata twoway).

        Args:
            layers (list): List of ``{PlotKind: {kwargs}}`` dicts.
            title (str): Plot title.
            xtitle (str): X-axis label.
            ytitle (str): Y-axis label.
            template (PlotTemplate): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.

        Raises:
            ValueError: If layers is empty or contains incompatible types.
        """
        if not layers:
            raise ValueError("layers must not be empty")

        kinds = [next(iter(d)) for d in layers]
        has_pie = any(k == PlotKind.pie for k in kinds)
        if has_pie and len(layers) > 1:
            raise ValueError("pie cannot be mixed with other plot types")

        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        colors = list(template.color_cycle)

        for layer in layers:
            kind = next(iter(layer))
            kwargs = dict(layer[kind])
            color = kwargs.pop("color", None) or template.primary_color

            if kind == PlotKind.scatter:
                ax.scatter(self._df[kwargs["x"]], self._df[kwargs["y"]],
                           s=template.marker_size, c=color)
            elif kind == PlotKind.line:
                x_data = self._df[kwargs["x"]] if "x" in kwargs else self._df.index
                ax.plot(x_data, self._df[kwargs["y"]], color=color)
            elif kind == PlotKind.bar:
                by = kwargs.get("by")
                var = kwargs["var"]
                if by:
                    agg = self._df.groupby(by)[var].agg(
                        kwargs.get("stat", "mean")).sort_values()
                    ax.bar(agg.index.astype(str), agg.values,
                           color=color, edgecolor="white", linewidth=0.5)
                else:
                    counts = self._df[var].value_counts().sort_index()
                    ax.bar(counts.index.astype(str), counts.values,
                           color=color, edgecolor="white", linewidth=0.5)
            elif kind == PlotKind.hist:
                ax.hist(self._df[kwargs["var"]].dropna(),
                        bins=kwargs.get("bins", 30),
                        density=kwargs.get("density", False),
                        color=color, edgecolor="white", linewidth=0.5,
                        alpha=0.5)
            elif kind == PlotKind.pie:
                counts = self._df[kwargs["var"]].value_counts()
                ax.pie(counts.values, labels=counts.index.astype(str),
                       colors=colors[:len(counts)], autopct="%1.1f%%",
                       startangle=90)
                ax.set_aspect("equal")
            elif kind in (PlotKind.lfit, PlotKind.lfitci):
                x_col = self._df[kwargs["x"]].dropna()
                y_col = self._df[kwargs["y"]].dropna()
                mask = x_col.index.intersection(y_col.index)
                xv = x_col.loc[mask].values.astype(float)
                yv = y_col.loc[mask].values.astype(float)
                # OLS: y = intercept + slope * x
                n = len(xv)
                x_mean = xv.mean()
                ssx = ((xv - x_mean) ** 2).sum()
                slope = ((xv - x_mean) * (yv - yv.mean())).sum() / ssx
                intercept = yv.mean() - slope * x_mean
                # sort for clean line
                order = np.argsort(xv)
                xs = xv[order]
                ys = intercept + slope * xs
                if kind == PlotKind.lfitci:
                    # CI: se(y_hat) = sqrt(MSE * (1/n + (x-x_mean)^2/SSx))
                    y_hat = intercept + slope * xv
                    mse = ((yv - y_hat) ** 2).sum() / max(n - 2, 1)
                    se = np.sqrt(mse * (1 / n + (xs - x_mean) ** 2 / ssx))
                    t_val = 1.96  # approx 95% CI
                    ax.fill_between(xs, ys - t_val * se, ys + t_val * se,
                                    color=color, alpha=0.15)
                ax.plot(xs, ys, color=color)

        if title is not None:
            ax.set_title(title)
        if xtitle is not None:
            ax.set_xlabel(xtitle)
        if ytitle is not None:
            ax.set_ylabel(ytitle)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

    def coefplot(self, result=None, **kwargs):
        """Plot regression coefficients with confidence intervals.

        Args:
            result: Result object, list of results, or None (uses latest).
            **kwargs: Passed to coefplot.plot_coefplot().

        Returns:
            TabraFigure: Wrapped matplotlib figure.
        """
        from tabra.plot.coefplot import plot_coefplot
        if result is None:
            result = self._tabra._result
        return plot_coefplot(result, **kwargs)

    def kdensity(self, var, by: str = None,
                 bw=None, kernel: str = "gaussian",
                 title: str = None, xtitle: str = None, ytitle: str = None,
                 template=None, fig_setting=None):
        """Draw a kernel density estimation plot.

        Args:
            var (str | list[str]): Variable name(s) to plot.
            by (str): Grouping variable. If None, draws without grouping.
            bw (float): Bandwidth. None = auto (Silverman's rule).
            kernel (str): Kernel type. Default 'gaussian'.
            title (str): Plot title.
            xtitle (str): X-axis label.
            ytitle (str): Y-axis label. Defaults to "Density".
            template (PlotTemplateBase): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        from scipy.stats import gaussian_kde

        template = template or self._tabra._config.plot_template
        template.apply()

        vars_list = [var] if isinstance(var, str) else var

        # Collect all combinations of (var, group)
        plot_specs = []
        if by is not None:
            groups = self._df[by].dropna().unique()
            for v in vars_list:
                for g in groups:
                    plot_specs.append((v, g))
        else:
            for v in vars_list:
                plot_specs.append((v, None))

        n_plots = len(plot_specs)
        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(template.fig_width * n_cols, template.fig_height * n_rows),
            dpi=template.dpi,
            squeeze=False,
        )
        if n_plots == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        colors = list(template.color_cycle)

        for idx, (v, group) in enumerate(plot_specs):
            ax = axes_flat[idx]
            if by is not None and group is not None:
                data = self._df.loc[self._df[by] == group, v].dropna().values
            else:
                data = self._df[v].dropna().values

            kde = gaussian_kde(data, bw_method=bw)
            x_grid = np.linspace(data.min(), data.max(), 200)
            density = kde(x_grid)

            color = colors[idx % len(colors)]
            ax.plot(x_grid, density, color=color, linewidth=template.line_width)
            ax.fill_between(x_grid, density, alpha=0.2, color=color)

            xlabel = xtitle if xtitle is not None else v
            ylabel = ytitle if ytitle is not None else "Density"
            ax.set_xlabel(xlabel, fontsize=template.label_size)
            ax.set_ylabel(ylabel, fontsize=template.label_size)

            if by is not None and group is not None:
                ax.set_title(f"{v} | {by}={group}", fontsize=template.title_size)
            elif title is not None and n_plots == 1:
                ax.set_title(title, fontsize=template.title_size)

            if not template.spine_top:
                ax.spines["top"].set_visible(False)
            if not template.spine_right:
                ax.spines["right"].set_visible(False)

        # Hide unused axes
        for idx in range(n_plots, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.tight_layout()
        return TabraFigure(fig, tabra=self._tabra)

    def heatmap(self, data=None, *, var_names=None,
                annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1.0, vmax=1.0,
                title=None, template=None, fig_setting=None):
        """Draw a heatmap from a correlation matrix or result.

        Args:
            data: CorrResult, DataFrame, ndarray, or None (uses latest result).
            var_names (list[str]): Variable labels for ndarray input.
            annot (bool): Show values in cells. Default True.
            fmt (str): Number format for annotations. Default ".2f".
            cmap (str): Colormap. Default "RdBu_r" (blue-white-red).
            vmin (float): Color scale minimum. Default -1.0.
            vmax (float): Color scale maximum. Default 1.0.
            title (str): Plot title.
            template (PlotTemplateBase): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.
        """
        from tabra.core.errors import NoResultError

        if data is None:
            data = self._tabra._result
            if data is None:
                raise NoResultError(
                    "No result to plot. Run tab.data.corr() first."
                )

        # Extract matrix and labels from various input types
        if hasattr(data, "matrix") and hasattr(data, "var_names"):
            matrix = np.asarray(data.matrix)
            labels = list(data.var_names)
            if hasattr(data, "accuracy"):
                # ConfusionMatrixResult — auto-switch defaults
                cmap = cmap if cmap != "RdBu_r" else "Blues"
                fmt = "d" if fmt == ".2f" else fmt
                vmin = vmin if vmin != -1.0 else 0
                vmax = vmax if vmax != 1.0 else int(matrix.max())
                if title is None:
                    title = "Confusion Matrix"
            else:
                if title is None:
                    title = f"Correlation ({getattr(data, 'method', 'pearson')})"
        elif isinstance(data, pd.DataFrame):
            matrix = data.values
            labels = list(data.columns)
        elif isinstance(data, np.ndarray):
            matrix = data
            labels = var_names or [f"V{i+1}" for i in range(matrix.shape[0])]
        else:
            raise TypeError(
                f"Expected CorrResult, DataFrame, or ndarray, "
                f"got {type(data).__name__}"
            )

        n = matrix.shape[0]
        template = template or self._tabra._config.plot_template
        template.apply()

        fig_size = max(template.fig_width, 2.0 + 0.6 * n)
        fig, ax = plt.subplots(
            figsize=(fig_size, fig_size),
            dpi=template.dpi,
        )

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="equal")

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=template.tick_size)
        ax.set_yticklabels(labels, fontsize=template.tick_size)

        # Annotations
        if annot:
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    text_color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, format(val, fmt),
                            ha="center", va="center",
                            color=text_color, fontsize=template.tick_size)

        if title is not None:
            ax.set_title(title, fontsize=template.title_size)

        # Colorbar
        fig.colorbar(im, ax=ax, shrink=0.8)

        if not template.spine_top:
            ax.spines["top"].set_visible(False)
        if not template.spine_right:
            ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return TabraFigure(fig, tabra=self._tabra)

    def rvfplot(self, result=None, title=None,
                xtitle=None, ytitle=None,
                template=None, fig_setting=None):
        """Draw a residual-vs-fitted plot (like Stata rvfplot).

        Plots residuals on the y-axis against fitted values on the x-axis,
        with a horizontal reference line at 0.

        Args:
            result: Result object with ``resid`` and ``fitted`` properties,
                or None (uses latest result).
            title (str): Plot title.
            xtitle (str): X-axis label. Defaults to "Fitted values".
            ytitle (str): Y-axis label. Defaults to "Residuals".
            template (PlotTemplateBase): Plot template to use.
            fig_setting: Reserved for compatibility. Ignored.

        Returns:
            TabraFigure: A wrapped figure object.

        Raises:
            NoResultError: If no result is stored.
            ResultTypeError: If the result lacks ``resid`` or ``fitted``.
        """
        from tabra.core.errors import NoResultError, ResultTypeError

        if result is None:
            result = self._tabra._result
            if result is None:
                raise NoResultError(
                    "No result to plot. Run a regression first."
                )

        if not hasattr(result, "resid") or not hasattr(result, "fitted"):
            raise ResultTypeError(
                "Result does not have resid/fitted. "
                "rvfplot requires a regression result (OLS, RegHDFE, etc.)."
            )

        template = template or self._tabra._config.plot_template
        template.apply()

        fig, ax = self._make_fig(template)
        resid = np.asarray(result.resid)
        fitted = np.asarray(result.fitted)

        ax.scatter(fitted, resid, s=template.marker_size,
                   c=template.primary_color, alpha=0.7)
        ax.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(xtitle if xtitle is not None else "Fitted values")
        ax.set_ylabel(ytitle if ytitle is not None else "Residuals")
        if title is not None:
            ax.set_title(title)
        self._apply_template(template, ax)
        return TabraFigure(fig, tabra=self._tabra)

# ---- Module-level global settings ----

_global_plot_template = None
_global_figure_save_base = None


def set_plot_template(template):
    """Set the global default plot template.

    Args:
        template (PlotTemplate): A PlotTemplate instance, e.g. ``AER``, ``QJE``.

    Example:
        >>> from tabra.plot.template import AER
        >>> from tabra.plot import set_plot_template
        >>> set_plot_template(AER)
    """
    global _global_plot_template
    from tabra.plot.templates import PlotTemplateBase
    if not isinstance(template, PlotTemplateBase):
        raise TypeError("template must be a PlotTemplateBase instance")
    _global_plot_template = template
    template.apply()


def set_save_base(base_dir):
    """Set the global base directory for relative figure save paths.

    Args:
        base_dir (str or Path): Base directory path.

    Example:
        >>> from tabra.plot import set_save_base
        >>> set_save_base("tmp/figs")
    """
    global _global_figure_save_base
    _global_figure_save_base = str(base_dir)
