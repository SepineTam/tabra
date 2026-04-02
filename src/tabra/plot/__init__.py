#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : __init__.py

import os

import matplotlib.pyplot as plt
import numpy as np

from tabra.plot.fig_setting import PlotKind


class TabraFigure:
    """Wrapper around matplotlib Figure with user-friendly API."""

    def __init__(self, fig, tabra=None):
        self._fig = fig
        self._tabra = tabra

    def save(self, filename: str, dpi: int = 300, **kwargs):
        """Save the figure to a file.

        Args:
            filename (str): Output file path.
            dpi (int): Resolution in dots per inch. Default 300.
            **kwargs: Additional keyword arguments passed to ``savefig``.

        Returns:
            TabraFigure: Returns self for method chaining.
        """
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        self._fig.savefig(filename, dpi=dpi, bbox_inches="tight", **kwargs)
        return self

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
