from tabra.plot.templates import econ


def test_econ_templates_basic_shape_and_distinct_primary_colors():
    templates = [econ.AER, econ.QJE, econ.JPE, econ.ECONOMETRICA, econ.RES]
    assert len(templates) == 5

    for tpl in templates:
        assert tpl.default_format == "pdf"
        assert tpl.dpi == 300
        assert len(tpl.color_cycle) == 8
        assert tpl.fig_width > 0
        assert tpl.fig_height > 0

    primary_colors = {tpl.primary_color for tpl in templates}
    assert len(primary_colors) == 5
