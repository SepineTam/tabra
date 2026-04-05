#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : _annotations.py

"""Shared utilities for figure notes and legend configuration."""


def apply_notes(fig, notes, template):
    """Add footnote-style text at the bottom of a matplotlib Figure.

    Args:
        fig: matplotlib Figure.
        notes: str, list[str], or None.
        template: PlotTemplateBase instance for font sizing.
    """
    if notes is None:
        return
    if isinstance(notes, str):
        notes = [notes]
    y = 0.01
    line_height = 0.015
    for line in notes:
        fig.text(
            0.05, y, line,
            fontsize=template.note_size,
            color=template.note_color,
            ha="left",
            va="bottom",
            transform=fig.transFigure,
        )
        y += line_height


def apply_legend(ax, legend, template):
    """Configure legend on a matplotlib Axes.

    Args:
        ax: matplotlib Axes.
        legend: dict or None. Keys: show, pos, labels, ncol, fontsize.
        template: PlotTemplateBase instance.
    """
    handles, labels = ax.get_legend_handles_labels()
    has_handles = len(handles) > 0

    if legend is None:
        if has_handles:
            ax.legend(fontsize=template.legend_size)
        return

    if not legend.get("show", True):
        if ax.legend_ is not None:
            ax.legend_.set_visible(False)
        return

    kwargs = {}
    kwargs["fontsize"] = legend.get("fontsize", template.legend_size)
    if "pos" in legend:
        kwargs["loc"] = legend["pos"]
    if "ncol" in legend:
        kwargs["ncol"] = legend["ncol"]
    if "labels" in legend:
        label_map = legend["labels"]
        labels = [label_map.get(l, l) for l in labels]
    if has_handles:
        ax.legend(handles=handles, labels=labels, **kwargs)
