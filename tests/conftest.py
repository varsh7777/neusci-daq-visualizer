import os

# GUI tests run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
