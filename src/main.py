"""PitchVision application entry point.

Launch the desktop UI with::

    python -m src.main          # from the project root
    python src/main.py          # alternative
"""

from __future__ import annotations

import sys


def main() -> None:
    """Create the Qt application, show the main window, and enter the event loop."""
    from PySide6.QtWidgets import QApplication

    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("PitchVision")
    app.setOrganizationName("PitchVision")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
