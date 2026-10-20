"""Dark / light plot backgrounds."""
from pyqtgraph.Qt import QtCore, QtGui

from neusci_daq import config, theme
from neusci_daq.config import PALETTE

from test_channels import app, make_cfg, pump, win  # noqa: F401  (fixtures)
from test_colors import pen_hex, record_block


def contrast(a: str, b: str) -> float:
    la, lb = theme._luminance(QtGui.QColor(a)), theme._luminance(QtGui.QColor(b))
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def bg(w, i=0):
    return w.plot_widgets[i].backgroundBrush().color().name()


def test_starts_dark_and_toggles_to_light_and_back(win):
    assert win.plot_mode == theme.DARK and bg(win) == "#000000"
    assert "click for light" in win.btn_theme.toolTip()
    win.toggle_plot_theme()
    assert win.plot_mode == theme.LIGHT and all(bg(win, i) == "#ffffff" for i in range(len(win.plot_widgets)))
    assert win.plots[0].getAxis("left").textPen().color().name() == theme.THEMES[theme.LIGHT].foreground
    assert "click for dark" in win.btn_theme.toolTip()
    win.toggle_plot_theme()
    assert bg(win) == "#000000"


def test_choice_is_remembered(win, tmp_path, app):
    from neusci_daq.gui import MainWindow
    win.set_plot_theme(theme.LIGHT)
    w2 = MainWindow(make_cfg(tmp_path), settings=win.settings)
    assert w2.plot_mode == theme.LIGHT and bg(w2) == "#ffffff"
    w2.close()


def test_every_palette_colour_is_visible_on_both_backgrounds():
    for c in PALETTE:
        assert contrast(theme.display_color(c, theme.LIGHT), "#ffffff") >= 3, c
        assert contrast(theme.display_color(c, theme.DARK), "#000000") >= 3, c
    assert contrast(theme.display_color("#000000", theme.DARK), "#000000") >= 3


def test_dark_colours_are_left_alone_on_black():
    assert theme.display_color("#e69f00", theme.DARK) == "#e69f00"


def test_light_mode_changes_drawing_not_the_saved_colour(app, win):
    win.set_channel_color(1, "#ffffff")
    record_block(app, win)
    win.btn_run.setChecked(False)
    win.set_plot_theme(theme.LIGHT)
    assert pen_hex(win.curves[1]) != "#ffffff"                     # drawn darker
    assert win.rows[1].ch.color == "#ffffff"                       # saved colour unchanged
    assert session_colors(win.blocks[-1])[1] == "#ffffff"


def session_colors(path):
    from neusci_daq import session
    return session.read_block(path)["colors"]


def test_comment_markers_follow_the_theme(app, win):
    win.btn_run.setChecked(True)
    pump(app, 0.2)
    win.btn_rec.setChecked(True)
    pump(app, 0.1)
    win.comment_edit.setText("stim")
    win._add_comment()
    ln = win.markers[0][1][0]
    assert ln.pen.color().name() == theme.THEMES[theme.DARK].marker
    win.set_plot_theme(theme.LIGHT)
    assert ln.pen.color().name() == theme.THEMES[theme.LIGHT].marker
    win.btn_rec.setChecked(False)
    win.btn_run.setChecked(False)


def test_review_and_new_plots_use_the_current_theme(app, win):
    win.set_plot_theme(theme.LIGHT)
    record_block(app, win)
    win.btn_run.setChecked(False)
    win.open_session(win.session_dir)
    assert bg(win) == "#ffffff"
    win.back_to_live()
    win.add_channel()
    assert all(bg(win, i) == "#ffffff" for i in range(len(win.plot_widgets)))


def test_icons_are_one_colour(app):
    for mode in (theme.DARK, theme.LIGHT):
        img = theme.toggle_icon(mode, "#000000").pixmap(20, 20).toImage()
        seen = {img.pixelColor(x, y).name() for x in range(20) for y in range(20)
                if img.pixelColor(x, y).alpha() > 0}
        assert seen == {"#000000"}, (mode, seen)


def test_icon_is_drawn_at_full_resolution_when_scaled(app):
    """At 150% display scaling, a 20 px icon must be drawn with 30 real pixels, not enlarged."""
    icon = theme.toggle_icon(theme.LIGHT, "#000000")
    for scale in (1.0, 1.25, 1.5, 2.0):
        pix = icon.pixmap(QtCore.QSize(20, 20), scale)
        assert pix.width() == round(20 * scale) and pix.devicePixelRatio() == scale
