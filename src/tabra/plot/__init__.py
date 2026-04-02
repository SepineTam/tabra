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
