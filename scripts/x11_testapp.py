"""A tiny X11 app for testing window hands: a counter you raise by clicking, plus typed text.
Responds to real and synthetic (XSendEvent) input. usage: x11_testapp.py TITLE"""
import sys
from Xlib import X, display, XK

title = sys.argv[1] if len(sys.argv) > 1 else "testapp"
d = display.Display()
s = d.screen()
win = s.root.create_window(int(sys.argv[2]) if len(sys.argv) > 2 else 20, 20, 420, 260, 1, s.root_depth, X.InputOutput, X.CopyFromParent,
                           background_pixel=s.white_pixel, event_mask=X.ExposureMask | X.ButtonPressMask | X.KeyPressMask)
win.set_wm_name(title)
win.change_property(d.intern_atom("_NET_WM_NAME"), d.intern_atom("UTF8_STRING"), 8, title.encode())
win.change_property(d.intern_atom("_NET_WM_PID"), X.AnyPropertyType, 32, [__import__("os").getpid()])
gc = win.create_gc(foreground=s.black_pixel, background=s.white_pixel)
win.map()
clicks, typed = 0, ""

def draw():
    win.clear_area()
    win.draw_text(gc, 20, 50, f"{title}: clicks={clicks}")
    win.draw_text(gc, 20, 90, f"typed={typed}")
    d.flush()

while True:
    e = d.next_event()
    if e.type == X.Expose:
        draw()
    elif e.type == X.ButtonPress:
        clicks += 1; draw()
    elif e.type == X.KeyPress:
        sym = d.keycode_to_keysym(e.detail, 0)
        ch = XK.keysym_to_string(sym) if sym else None
        if ch: typed += ch
        draw()
