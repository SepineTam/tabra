import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tabra.plot._annotations import apply_legend, apply_notes


class DummyTemplate:
    note_size = 9
    note_color = "black"
    legend_size = 8


def test_apply_notes_handles_none_and_string_and_list():
    fig, _ = plt.subplots()
    template = DummyTemplate()

    apply_notes(fig, None, template)
    assert len(fig.texts) == 0

    apply_notes(fig, "single note", template)
    assert len(fig.texts) == 1
    assert fig.texts[0].get_text() == "single note"

    apply_notes(fig, ["note a", "note b"], template)
    assert len(fig.texts) == 3
    assert fig.texts[1].get_text() == "note a"
    assert fig.texts[2].get_text() == "note b"


def test_apply_legend_default_custom_and_hide():
    fig, ax = plt.subplots()
    template = DummyTemplate()
    ax.plot([0, 1], [0, 1], label="l1")

    apply_legend(ax, None, template)
    assert ax.legend_ is not None

    apply_legend(ax, {"labels": {"l1": "renamed"}, "ncol": 1}, template)
    legend_texts = [t.get_text() for t in ax.legend_.get_texts()]
    assert "renamed" in legend_texts

    apply_legend(ax, {"show": False}, template)
    assert ax.legend_ is not None
    assert ax.legend_.get_visible() is False
