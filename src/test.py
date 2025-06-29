from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMessageBox,
    QAction,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
    QDialogButtonBox,
)
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread

import sys


class NewGameDialog(QDialog):
    def __init__(self, agent_players, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start a New Game")

        main_layout = QVBoxLayout(self)
        columns_layout = QHBoxLayout()

        player1_group = QGroupBox("Player 1 (Black)")
        player1_layout = QVBoxLayout()
        player1_group.setLayout(player1_layout)

        self.player1_buttons = []
        human1_radio = QRadioButton("Human")
        human1_radio.setChecked(True)
        self.player1_buttons.append({"name": "Human", "radio": human1_radio})
        player1_layout.addWidget(human1_radio)

        for i, agent in enumerate(agent_players):
            radio_button = QRadioButton(agent["name"])
            self.player1_buttons.append({"name": f"AI_{i}", "radio": radio_button})
            player1_layout.addWidget(radio_button)

        player1_layout.addStretch()
        columns_layout.addWidget(player1_group)

        player2_group = QGroupBox("Player 2 (White)")
        player2_layout = QVBoxLayout()
        player2_group.setLayout(player2_layout)

        self.player2_buttons = []
        for i, agent in enumerate(agent_players):
            radio_button = QRadioButton(agent["name"])
            if i == 0:
                radio_button.setChecked(True)
            self.player2_buttons.append({"name": f"AI_{i}", "radio": radio_button})
            player2_layout.addWidget(radio_button)

        player2_layout.addStretch()
        columns_layout.addWidget(player2_group)

        main_layout.addLayout(columns_layout)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

    def get_selected_players(self):
        p1_selection = None
        for i, button_info in enumerate(self.player1_buttons):
            if button_info["radio"].isChecked():
                p1_selection = -1 if button_info["name"] == "Human" else i - 1
                break

        p2_selection = None
        for i, button_info in enumerate(self.player2_buttons):
            if button_info["radio"].isChecked():
                p2_selection = -1 if button_info["name"] == "Human" else i - 1
                break

        return p1_selection, p2_selection


if __name__ == "__main__":
    app = QApplication(sys.argv)

    d = NewGameDialog(
        agent_players=[
            {"name": "AI Player 1"},
            {"name": "AI Player 2"},
            {"name": "AI Player 3"},
        ],
        parent=app,
    )
    d.exec_()
    # main_window.start_ai_game()

    sys.exit(app.exec_())
