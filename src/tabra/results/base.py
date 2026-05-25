#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 - Present Sepine Tam, Inc. All Rights Reserved
#
# @Author : Sepine Tam (谭淞)
# @Email  : sepinetam@gmail.com
# @File   : base.py

from abc import ABC, abstractmethod


class BaseResult(ABC):
    def __init__(self):
        self._style = "stata"
        self._display_command = ""

    def set_style(self, style: str):
        self._style = style

    def set_command(self, command: str):
        """Set the command line label used by interactive display."""
        self._display_command = command

    def set_display(self, is_display: bool = True):
        if is_display:
            print(self.render_display_block())

    @abstractmethod
    def summary(self): ...

    def __repr__(self):
        return self.formatted_summary()

    def display_command(self) -> str:
        """Return the display command label."""
        if self._display_command:
            return self._display_command
        class_name = self.__class__.__name__
        if class_name.endswith("Result"):
            class_name = class_name[:-6]
        return class_name.lower()

    def formatted_summary(self) -> str:
        """Return a display-safe summary string.

        This normalizes escaped control sequences that may appear in upstream
        string payloads and makes tab spacing predictable across terminals.
        """
        text = self.summary()
        if not isinstance(text, str):
            text = str(text)

        # Normalize escaped sequences when they are embedded as plain text.
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace tabs with spaces to reduce terminal-specific alignment drift.
        lines = [line.expandtabs(4).rstrip() for line in text.split("\n")]
        return "\n".join(lines).rstrip("\n")

    def render_display_block(self) -> str:
        """Render a concise Stata-like display block."""
        divider = "-" * 28
        command_line = f". {self.display_command()}"
        body = self.formatted_summary()
        return f"{divider}\n{command_line}\n\n{body}\n"

    @abstractmethod
    def save(self, path): ...

    def coefplot(self, **kwargs):
        """Create a coefficient plot from this result.

        Args:
            **kwargs: Passed to tabra.plot.coefplot.plot_coefplot().

        Returns:
            TabraFigure: Wrapped matplotlib figure.
        """
        from tabra.plot.coefplot import plot_coefplot
        return plot_coefplot(self, **kwargs)
