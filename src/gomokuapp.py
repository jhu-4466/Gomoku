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
    QLabel,
    QPushButton,
)
from PyQt5.QtGui import QImage, QPainter, QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread

import pygame
import sys
import os
import numpy as np

import requests
import random
import time
import json
from datetime import datetime


# Board attributes
PYGAME_CANVAS_WIDTH = 780
PYGAME_CANVAS_HEIGHT = 830
GRID_SIZE = 19
CELL_SIZE = 40
BOARD_MARGIN = 30
BOARD_WIDTH = CELL_SIZE * (GRID_SIZE - 1)
BOARD_HEIGHT = BOARD_WIDTH
INFO_BAR_HEIGHT = 50

# Colors
COLOR_BOARD = (230, 200, 150)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_LINE = (50, 50, 50)
COLOR_WINNER_BG = (0, 0, 0, 0)

# Star points
STAR_POINTS = [
    (3, 3),
    (3, 9),
    (3, 15),
    (9, 3),
    (9, 9),
    (9, 15),
    (15, 3),
    (15, 9),
    (15, 15),
]
PLAYER_MAPS = {
    -1: "Human",
    1: "Random Strategy",
    2: "Baseline Strategy",
}


class GomokuCanvas(QWidget):
    gameOverSignal = pyqtSignal(int)
    stateChangedSignal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT - INFO_BAR_HEIGHT)

        pygame.init()
        self.screen = pygame.Surface(
            (PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT - INFO_BAR_HEIGHT),
            pygame.SRCALPHA,
        )
        self.font_small = pygame.font.SysFont("sans", 20)
        self.font_large = pygame.font.SysFont("sans", 50)

        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.player_map = {1: "Human", 2: "AI"}
        self.is_in_game = False

    def paintEvent(self, event):
        self.draw_frame()
        view = pygame.surfarray.pixels3d(self.screen)
        view = view.transpose([1, 0, 2])
        img = QImage(
            view.tobytes(),
            view.shape[1],
            view.shape[0],
            view.shape[1] * 3,
            QImage.Format.Format_RGB888,
        )
        painter = QPainter(self)
        painter.drawImage(0, 0, img)

    def draw_frame(self):
        self.screen.fill(COLOR_BOARD)
        self.draw_board_grid()
        self.draw_pieces()

    def reset_game(self, player_map):
        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.player_map = player_map
        self.is_in_game = True
        self.emit_state_change()
        self.update()

    def agent_move(self, row, col):
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0:
            # record the move history
            move_data = {
                "turn": len(self.move_history) + 1,
                "player": self.current_player,
                "move": (row, col),
            }
            self.move_history.append(move_data)

            self.board[row][col] = self.current_player
            if self.check_win(row, col):
                self.game_over = True
                self.winner = self.current_player
                self.gameOverSignal.emit(self.winner)
            else:
                self.current_player = 3 - self.current_player
            self.emit_state_change()
            self.update()

    def mousePressEvent(self, event):
        if self.game_over or not self.is_in_game:
            return

        pos = (event.pos().x(), event.pos().y())
        row = round((pos[1] - BOARD_MARGIN) / CELL_SIZE)
        col = round((pos[0] - BOARD_MARGIN) / CELL_SIZE)

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0:
            # record the move history
            move_data = {
                "turn": len(self.move_history) + 1,
                "player": self.current_player,
                "move": (row, col),
            }
            self.move_history.append(move_data)

            self.board[row][col] = self.current_player
            if self.check_win(row, col):
                self.game_over = True
                self.is_in_game = False
                self.winner = self.current_player
                self.emit_state_change()
                self.gameOverSignal.emit(self.winner)
            else:
                self.current_player = 3 - self.current_player
                self.emit_state_change()
            self.update()

    def emit_state_change(self):
        state = {
            "p1_name": self.player_map.get(1, "Player 1"),
            "p2_name": self.player_map.get(2, "Player 2"),
            "current_player": self.current_player,
            "move_count": len(self.move_history),
            "game_over": self.game_over,
            "winner": self.winner,
        }
        self.stateChangedSignal.emit(state)

    def check_win(self, r, c):
        p = self.board[r][c]
        d = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for d_row, d_col in d:
            count = 1
            for i in range(1, 5):
                nr, nc = r + i * d_row, c + i * d_col
                if (
                    0 <= nr < GRID_SIZE
                    and 0 <= nc < GRID_SIZE
                    and self.board[nr][nc] == p
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - i * d_row, c - i * d_col
                if (
                    0 <= nr < GRID_SIZE
                    and 0 <= nc < GRID_SIZE
                    and self.board[nr][nc] == p
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def draw_board_grid(self):
        for i in range(GRID_SIZE):
            pygame.draw.line(
                self.screen,
                COLOR_LINE,
                (BOARD_MARGIN + i * CELL_SIZE, BOARD_MARGIN),
                (BOARD_MARGIN + i * CELL_SIZE, BOARD_HEIGHT + BOARD_MARGIN),
            )
            pygame.draw.line(
                self.screen,
                COLOR_LINE,
                (BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
                (BOARD_WIDTH + BOARD_MARGIN, BOARD_MARGIN + i * CELL_SIZE),
            )
        for r, c in STAR_POINTS:
            pygame.draw.circle(
                self.screen,
                COLOR_LINE,
                (BOARD_MARGIN + c * CELL_SIZE, BOARD_MARGIN + r * CELL_SIZE),
                6,
            )

    def draw_pieces(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.board[r][c] != 0:
                    piece_color = COLOR_BLACK if self.board[r][c] == 1 else COLOR_WHITE
                    center = (
                        BOARD_MARGIN + c * CELL_SIZE,
                        BOARD_MARGIN + r * CELL_SIZE,
                    )
                    pygame.draw.circle(
                        self.screen, piece_color, center, CELL_SIZE // 2 - 2
                    )


class GomokuBoard(QWidget):
    gameOverSignal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT)

        self.is_paused = False

        self.info_label = QLabel("Start a new game from the 'Game' menu.", self)
        self.info_label.setFont(QFont("Arial", 12))
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setFixedWidth(80)
        self.pause_button.hide()
        self.resume_button = QPushButton("Resume", self)
        self.resume_button.setFixedWidth(80)
        self.resume_button.hide()

        self.canvas = GomokuCanvas(self)
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(10, 0, 10, 0)
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.pause_button)
        info_layout.addWidget(self.resume_button)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 0, 0)
        main_layout.setSpacing(5)
        main_layout.addLayout(info_layout)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)

        self.pause_button.clicked.connect(self.toggle_pause)
        self.resume_button.clicked.connect(self.toggle_pause)
        self.canvas.stateChangedSignal.connect(self.update_info_bar)
        self.canvas.gameOverSignal.connect(self.gameOverSignal)

    def update_info_bar(self, state: dict):
        if state["game_over"]:
            winner_name = state["p1_name"] if state["winner"] == 1 else state["p2_name"]
            text = f"<b>Game Over! Winner is {winner_name}.</b>"
        elif self.is_paused:
            text = "<i>Game Paused</i>"
        else:
            p1_name = state["p1_name"]
            p2_name = state["p2_name"]
            current_player_name = p1_name if state["current_player"] == 1 else p2_name
            text = f"P1({p1_name}) vs P2({p2_name}) | <b>Turn {state['move_count'] + 1}</b>: {current_player_name} to move"

        self.info_label.setText(text)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.canvas.setEnabled(not self.is_paused)

        if self.is_paused:
            self.pause_button.hide()
            self.resume_button.show()
        else:
            self.pause_button.show()
            self.resume_button.hide()

        self.update_info_bar(
            self.canvas.stateChangedSignal.emit(
                self.canvas.emit_state_change() or self.canvas.__dict__
            )
        )

    def reset_game(self, players_config):
        player_map = {
            1: players_config[1]["name"] if players_config[1] else "Human",
            2: players_config[2]["name"] if players_config[2] else "AI",
        }
        if self.is_paused:
            self.toggle_pause()

        self.canvas.reset_game(player_map)
        self.pause_button.show()

    def agent_move(self, row, col):
        self.canvas.agent_move(row, col)


class GomokuAgentHandler(QThread):
    moveMade = pyqtSignal(int, int)
    gameOver = pyqtSignal(str)

    def __init__(self, game_canvas, players):
        super().__init__()
        self.game_canvas = game_canvas
        self.players = players
        self.running = True

    def run(self):
        while self.running and not self.game_canvas.game_over:
            current_player_id = self.game_canvas.current_player
            player_config = self.players.get(current_player_id)

            if player_config is None:
                self.msleep(100)
                continue

            player_url = player_config["url"]
            try:
                payload = {"board": self.game_canvas.board, "player": current_player_id}
                response = requests.post(player_url, json=payload, timeout=5)
                response.raise_for_status()

                data = response.json()
                move = data.get("move")
                if not move:
                    self.gameOver.emit("Draw")
                    break

                row, col = move[0], move[1]
                if self.game_canvas.board[row][col] != 0:
                    error_msg = (
                        f"Error: AI from {player_url} returned an occupied cell {move}."
                    )
                    print(error_msg)
                    self.gameOver.emit("Error: AI returned invalid move.")
                    break

                self.moveMade.emit(row, col)
                self.msleep(200)
                if self.game_canvas.game_over:
                    break

                self.msleep(800)
            except requests.RequestException as e:
                print(f"Error communicating with AI at {player_url}: {e}")
                self.gameOver.emit(f"Error: Cannot connect to AI.")
                break

    def stop(self):
        self.running = False


class NewGameDialog(QDialog):
    def __init__(self, agent_players, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start a New Game")

        main_layout = QVBoxLayout(self)
        columns_layout = QHBoxLayout()

        player1_group = QGroupBox("Player 1")
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

        player2_group = QGroupBox("Player 2")
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
                p2_selection = -1 if button_info["name"] == "Human" else i
                break

        return p1_selection, p2_selection


class GomokuApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gomoku(Five in a Row)")

        # Board Widget
        self.game_engine = GomokuBoard(self)
        self.setCentralWidget(self.game_engine)
        self.game_engine.gameOverSignal.connect(self.gameover_assert)

        # Menu Bar
        menu_bar = self.menuBar()
        game_menu = menu_bar.addMenu("&Game")
        help_menu = menu_bar.addMenu("&Help")
        new_game_action = QAction("New Game", self)
        # undo_action = QAction("Undo", self)
        exit_action = QAction("Exit", self)
        rules_action = QAction("Game Rules", self)

        new_game_action.triggered.connect(self.prompt_new_game)
        # undo_action.triggered.connect(self.game_engine.undo_move)
        exit_action.triggered.connect(self.close)
        rules_action.triggered.connect(self.rules_description)

        game_menu.addAction(new_game_action)
        # game_menu.addAction(undo_action)
        game_menu.addAction(exit_action)
        help_menu.addAction(rules_action)

        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.game_engine.update)
        # self.timer.start(16)  # ~60 FPS

        self.agent_players_config = [
            {"name": "Random Strategy", "url": "http://127.0.0.1:5001/get_move"},
            {"name": "Baseline Strategy", "url": "http://127.0.0.1:5002/get_move"},
        ]
        self.agent_handler = None
        self.players = {1: None, 2: None}  # 1: Black, 2: White

    def start_game_loop(self):
        self.stop_game_loop()

        self.agent_handler = GomokuAgentHandler(self.game_engine.canvas, self.players)

        self.agent_handler.moveMade.connect(self.game_engine.agent_move)
        self.agent_handler.gameOver.connect(self.agent_gameover_handler)

        self.agent_handler.start()

    def stop_game_loop(self):
        if self.agent_handler and self.agent_handler.isRunning():
            self.agent_handler.stop()
            self.agent_handler.quit()
            self.agent_handler.wait(3000)
        self.agent_handler = None

    def prompt_new_game(self):
        """
        Opens a dialog to select the AI difficulty and starts a new game.
        """
        dialog = NewGameDialog(self.agent_players_config, parent=self)
        if dialog.exec_():
            p1_index, p2_index = dialog.get_selected_players()
            self.players[1] = (
                None if p1_index == -1 else self.agent_players_config[p1_index]
            )
            self.players[2] = (
                None if p2_index == -1 else self.agent_players_config[p2_index]
            )

            self.game_engine.reset_game(self.players)
            self.start_game_loop()

    def gameover_assert(self, winner: int):
        self.save_game_history(winner)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Game Over")
        msg_box.setText(
            f"<h2>{self.players[winner]["name"]if self.players[winner] else "Human"} Wins!</h2>"
        )
        msg_box.setInformativeText("What would you like to do next?")
        msg_box.setIcon(QMessageBox.Icon.Question)

        play_again_btn = msg_box.addButton(
            "Play Again", QMessageBox.ButtonRole.AcceptRole
        )
        review_board_btn = msg_box.addButton(
            "Review Board", QMessageBox.ButtonRole.RejectRole
        )

        msg_box.setDefaultButton(play_again_btn)
        msg_box.exec_()

        if msg_box.clickedButton() == play_again_btn:
            self.game_engine.reset_game(self.players)
            self.start_game_loop()

    def rules_description(self):
        rules_text = """
            <b>1. Two players (Black and White) take turns.</b><br>
            <b>2. Black always moves first.</b><br>
            <b>3. Place a stone on an empty intersection.</b><br>
            <b>4. The first player to form an unbroken line of five stones wins.</b><br>
            <b>5. The line can be horizontal, vertical, or diagonal.</b><br>
            <b>6. Once placed, stones cannot be moved.</b><br>
        """
        QMessageBox.information(self, "Gomoku Rules", rules_text)

    def start_pure_ai_game(self):
        """
        待改
        """
        self.stop_game_loop()

        self.game_engine.reset_game()
        self.game_engine.is_agent_mode = True

        urls = ["http://127.0.0.1:5001/get_move", "http://127.0.0.1:5002/get_move"]
        partipants = [
            {"name": "Random Strategy", "url": urls[0]},
            {"name": "Baseline Strategy", "url": urls[1]},
        ]
        random.shuffle(partipants)

        self.players = {1: urls[0], 2: urls[1]}  # 1 - Black, 2 - White

        self.agent_handler = GomokuAgentHandler(self.game_engine, self.players)
        self.agent_handler.gameOver.connect(self.agent_gameover_handler)
        self.agent_handler.start()

    def agent_gameover_handler(self, message: str):
        pass

    def save_game_history(self, winner_player_id):
        """
        Saves the completed game's data to a JSON file.
        """
        if not os.path.exists("game_history"):
            os.makedirs("game_history")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./game_history/game_{timestamp}.json"

        p1_name = self.players[1]["name"] if self.players[1] else "Human"
        p2_name = self.players[2]["name"] if self.players[2] else "Human"
        winner_name = "None"
        if winner_player_id == 1:
            winner_name = p1_name
        elif winner_player_id == 2:
            winner_name = p2_name
        history_data = {
            "game_id": timestamp,
            "winner_player_id": winner_player_id,
            "winner_name": winner_name,
            "players": {
                "1": {"type": p1_name, "color": "Black"},
                "2": {"type": p2_name, "color": "White"},
            },
            "move_history": self.game_engine.canvas.move_history,
        }

        try:
            with open(filename, "w") as f:
                json.dump(history_data, f, indent=4)
            print(f"Game history saved to {filename}")
        except Exception as e:
            print(f"Error saving game history: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = GomokuApp()
    main_window.show()

    sys.exit(app.exec_())
