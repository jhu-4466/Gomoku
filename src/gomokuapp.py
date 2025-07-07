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
    QCheckBox,
)
from PyQt5.QtGui import QImage, QPainter, QFont, QColor
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


# Board attributes (15x15)
PYGAME_CANVAS_WIDTH = 620
PYGAME_CANVAS_HEIGHT = 670
GRID_SIZE = 15
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
COLOR_BANNED = (255, 0, 0, 150)

# Star points (15x15)
STAR_POINTS = [
    (3, 3),
    (3, 11),
    (7, 7),
    (11, 3),
    (11, 11),
]

# Reconnect
MAX_RETRIES = 12
RETRY_DELAY_S = 5


class GomokuCanvas(QWidget):
    gameOverSignal = pyqtSignal(int)
    stateChangedSignal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT - INFO_BAR_HEIGHT)

        self.setMouseTracking(True)

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
        self.banned_moves_enabled = False
        self.banned_point_on_click = None
        self.banned_point_on_hover = None

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

        if self.banned_point_on_hover:
            self._draw_banned_overlay(self.banned_point_on_hover)
        elif self.banned_point_on_click:
            self._draw_banned_overlay(self.banned_point_on_click)
            self.banned_point_on_click = None

    def _draw_banned_overlay(self, point):
        center = (
            BOARD_MARGIN + point[1] * CELL_SIZE,
            BOARD_MARGIN + point[0] * CELL_SIZE,
        )
        s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        s.fill(COLOR_BANNED)
        self.screen.blit(s, (center[0] - CELL_SIZE // 2, center[1] - CELL_SIZE // 2))

    def reset_game(self, player_map, banned_moves_enabled):
        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.player_map = player_map
        self.is_in_game = True
        self.banned_moves_enabled = banned_moves_enabled
        self.state_change()
        self.update()

    def agent_move(self, row, col):
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0:
            self.place_piece(row, col)

    def mouseMoveEvent(self, event):
        is_black_human_turn = (
            not self.game_over
            and self.is_in_game
            and self.banned_moves_enabled
            and self.player_map.get(self.current_player) == "Human"
            and self.current_player == 1
        )

        if not is_black_human_turn:
            if self.banned_point_on_hover:
                self.banned_point_on_hover = None
                self.update()
            return

        pos = (event.pos().x(), event.pos().y())
        row = round((pos[1] - BOARD_MARGIN) / CELL_SIZE)
        col = round((pos[0] - BOARD_MARGIN) / CELL_SIZE)

        new_hover_point = None
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0:
            is_banned, _ = self.check_banned_move(row, col)
            if is_banned:
                new_hover_point = (row, col)

        if self.banned_point_on_hover != new_hover_point:
            self.banned_point_on_hover = new_hover_point
            self.update()

    def mousePressEvent(self, event):
        if (
            self.game_over
            or not self.is_in_game
            or self.player_map.get(self.current_player) != "Human"
        ):
            return

        pos = (event.pos().x(), event.pos().y())
        row = round((pos[1] - BOARD_MARGIN) / CELL_SIZE)
        col = round((pos[0] - BOARD_MARGIN) / CELL_SIZE)

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0:
            if self.banned_moves_enabled and self.current_player == 1:
                is_banned, reason = self.check_banned_move(row, col)
                if is_banned:
                    print(f"Banned move at ({row}, {col}): {reason}")
                    self.banned_point_on_click = (row, col)
                    self.update()
                    QMessageBox.warning(
                        self,
                        "Banned Move",
                        f"This move ({reason}) is not allowed for Black.",
                    )
                    return

            self.place_piece(row, col)

    def place_piece(self, row, col):
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
            self.gameOverSignal.emit(self.winner)
        elif (
            self.banned_moves_enabled
            and self.current_player == 1
            and self.check_overline(row, col)
        ):
            self.game_over = True
            self.is_in_game = False
            self.winner = 2
            self.gameOverSignal.emit(self.winner)
        else:
            self.current_player = 3 - self.current_player

        self.state_change()
        self.update()

    def get_line(self, r, c, dr, dc):
        line = []
        for i in range(-4, 5):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                line.append(self.board[nr][nc])
            else:
                line.append(-1)
        return line

    def check_banned_move(self, r, c):
        if self.board[r][c] != 0:
            return False, ""

        self.board[r][c] = 1

        # [MODIFIED] Check for overline first
        if self.check_overline(r, c):
            self.board[r][c] = 0
            return True, "Overline"

        threes = 0
        fours = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            line = self.get_line(r, c, dr, dc)
            line_str = "".join(map(str, line))
            if "01110" in line_str.replace("2", "0"):
                threes += 1

            if "1111" in line_str:
                fours += 1

        self.board[r][c] = 0

        if threes >= 2:
            return True, "Three-Three"
        if fours >= 2:
            return True, "Four-Four"

        return False, ""

    def check_overline(self, r, c):
        p = self.board[r][c]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 6):
                nr, nc = r + i * dr, c + i * dc
                if (
                    0 <= nr < GRID_SIZE
                    and 0 <= nc < GRID_SIZE
                    and self.board[nr][nc] == p
                ):
                    count += 1
                else:
                    break
            for i in range(1, 6):
                nr, nc = r - i * dr, c - i * dc
                if (
                    0 <= nr < GRID_SIZE
                    and 0 <= nc < GRID_SIZE
                    and self.board[nr][nc] == p
                ):
                    count += 1
                else:
                    break
            if count > 5:
                return True
        return False

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
            if count == 5:
                return True
        return False

    def get_current_state(self):
        return {
            "p1_name": self.player_map.get(1, "Player 1"),
            "p2_name": self.player_map.get(2, "Player 2"),
            "current_player": self.current_player,
            "move_count": len(self.move_history),
            "game_over": self.game_over,
            "winner": self.winner,
        }

    def state_change(self):
        self.stateChangedSignal.emit(self.get_current_state())

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
        info_bar_container = QWidget()
        info_bar_container.setFixedHeight(INFO_BAR_HEIGHT)
        info_layout = QHBoxLayout(info_bar_container)
        info_layout.setContentsMargins(10, 0, 10, 0)
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        info_layout.addWidget(self.pause_button)
        info_layout.addWidget(self.resume_button)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(info_bar_container)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.resume_button.clicked.connect(self.toggle_pause)
        self.canvas.stateChangedSignal.connect(self.update_info_bar)
        self.canvas.gameOverSignal.connect(self.gameOverSignal)

    def update_info_bar(self, state: dict):
        if self.is_paused:
            text = "<i>Game Paused</i>"
        elif state["game_over"]:
            winner_name = "No one"
            if state["winner"] is not None:
                winner_name = (
                    state["p1_name"] if state["winner"] == 1 else state["p2_name"]
                )
            text = f"<b>Game Over! Winner is {winner_name}.</b>"
            self.pause_button.hide()
        else:
            p1_name = state["p1_name"]
            p2_name = state["p2_name"]
            current_player_name = p1_name if state["current_player"] == 1 else p2_name
            text = f"P1({p1_name}) vs P2({p2_name}) | <b>Turn {state['move_count'] + 1}</b>: {current_player_name} to move"
            self.pause_button.show()
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
        self.update_info_bar(self.canvas.get_current_state())

    def reset_game(self, players_config, banned_moves_enabled):
        player_map = {
            1: players_config[1]["name"] if players_config[1] else "Human",
            2: players_config[2]["name"] if players_config[2] else "AI",
        }
        if self.is_paused:
            self.toggle_pause()
        self.canvas.reset_game(player_map, banned_moves_enabled)
        self.pause_button.show()

    def agent_move(self, row, col):
        self.canvas.agent_move(row, col)

    def reconnectstatus_show(self, message: str):
        self.info_label.setText(f"<i>{message}</i>")
        self.pause_button.hide()


class GomokuAgentHandler(QThread):
    movedSignal = pyqtSignal(int, int)
    reconnectSignal = pyqtSignal(str)
    gameOverSignal = pyqtSignal(str)

    def __init__(self, game_canvas, players):
        super().__init__()
        self.game_canvas = game_canvas
        self.players = players
        self.running = True

    def run(self):
        while self.running and not self.game_canvas.game_over:
            if not self.running:
                break

            current_player_id = self.game_canvas.current_player
            player_config = self.players.get(current_player_id)

            if player_config is None:
                self.msleep(100)
                continue

            player_url = player_config["url"]
            player_timeout = player_config.get("timeout", 5)
            move = None
            for attempt in range(MAX_RETRIES):
                if not self.running:
                    break
                try:
                    payload = {
                        "board": self.game_canvas.board,
                        "player": current_player_id,
                        "banned_moves_enabled": self.game_canvas.banned_moves_enabled,
                    }
                    response = requests.post(
                        player_url, json=payload, timeout=player_timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    move = data.get("move")
                    break
                except requests.RequestException:
                    if attempt < MAX_RETRIES - 1:
                        reconnect_notice = f"Connection lost. Retrying... ({attempt + 1}/{MAX_RETRIES})"
                        self.reconnectSignal.emit(reconnect_notice)

                        for _ in range(RETRY_DELAY_S * 10):
                            if not self.running:
                                break
                            self.msleep(100)
                        if not self.running:
                            break
                    else:
                        error_msg = "Error: Cannot connect to AI player."
                        self.gameOverSignal.emit(error_msg)
                        self.running = False

            if not self.running:
                break

            if move is None:
                if self.running:
                    self.gameOverSignal.emit("Draw")
                break

            row, col = move[0], move[1]
            if self.game_canvas.board[row][col] != 0:
                print(f"Error: AI from {player_url} returned an occupied cell {move}.")
                break

            self.movedSignal.emit(row, col)
            self.msleep(200)
            if self.game_canvas.game_over:
                break
            self.msleep(800)

    def stop(self):
        self.running = False


class NewGameDialog(QDialog):
    def __init__(self, agent_players, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start a New Game")

        main_layout = QVBoxLayout(self)

        rules_group = QGroupBox("Game Rules")
        rules_layout = QHBoxLayout(rules_group)
        self.banned_checkbox = QCheckBox("Enable Banned Moves (Renju Rules)")
        self.banned_checkbox.setToolTip(
            "Forbids Black from making Three-Three, Four-Four, and Overline moves."
        )
        rules_layout.addWidget(self.banned_checkbox)
        main_layout.addWidget(rules_group)

        columns_layout = QHBoxLayout()
        player1_group = QGroupBox("Player 1 (Black)")
        player1_layout = QVBoxLayout(player1_group)
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
        player2_layout = QVBoxLayout(player2_group)
        self.player2_buttons = []
        for i, agent in enumerate(agent_players):
            radio_button = QRadioButton(agent["name"])
            if i == 2:
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

    def get_settings(self):
        p1_selection, p2_selection = None, None
        for i, button_info in enumerate(self.player1_buttons):
            if button_info["radio"].isChecked():
                p1_selection = -1 if button_info["name"] == "Human" else i - 1
                break
        for i, button_info in enumerate(self.player2_buttons):
            if button_info["radio"].isChecked():
                p2_selection = -1 if button_info["name"] == "Human" else i
                break

        return {
            "p1_index": p1_selection,
            "p2_index": p2_selection,
            "banned_moves_enabled": self.banned_checkbox.isChecked(),
        }


class GomokuApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gomoku / Renju")
        self.game_engine = GomokuBoard(self)
        self.setCentralWidget(self.game_engine)
        self.game_engine.gameOverSignal.connect(self.gameover_assert)
        self._create_menu()
        self.agent_players_config = [
            {"name": "Random Strategy", "url": "http://127.0.0.1:5001/get_move"},
            {"name": "Baseline Strategy", "url": "http://127.0.0.1:5002/get_move"},
            {
                "name": "Negamax AI",
                "url": "http://127.0.0.1:5003/get_move",
                "timeout": 180,
            },
        ]
        self.agent_handler = None
        self.players = {1: None, 2: None}
        self.game_settings = {}

    def _create_menu(self):
        menu_bar = self.menuBar()
        game_menu = menu_bar.addMenu("&Game")
        help_menu = menu_bar.addMenu("&Help")
        new_game_action = QAction("New Game", self)
        exit_action = QAction("Exit", self)
        rules_action = QAction("Game Rules", self)
        new_game_action.triggered.connect(self.prompt_new_game)
        exit_action.triggered.connect(self.close)
        rules_action.triggered.connect(self.rules_description)
        game_menu.addAction(new_game_action)
        game_menu.addAction(exit_action)
        help_menu.addAction(rules_action)

    def start_game_loop(self):
        self.stop_game_loop()
        self.agent_handler = GomokuAgentHandler(self.game_engine.canvas, self.players)
        self.agent_handler.movedSignal.connect(self.game_engine.agent_move)
        self.agent_handler.reconnectSignal.connect(
            self.game_engine.reconnectstatus_show
        )
        self.agent_handler.gameOverSignal.connect(self.agent_gameover_handler)
        self.agent_handler.start()

    def stop_game_loop(self):
        if self.agent_handler and self.agent_handler.isRunning():
            self.agent_handler.stop()
            self.agent_handler.quit()
            self.agent_handler.wait(3000)
        self.agent_handler = None

    def prompt_new_game(self):
        self.stop_game_loop()

        dialog = NewGameDialog(self.agent_players_config, parent=self)
        if dialog.exec_():
            self.game_settings = dialog.get_settings()
            p1_index = self.game_settings["p1_index"]
            p2_index = self.game_settings["p2_index"]

            self.players[1] = (
                None if p1_index == -1 else self.agent_players_config[p1_index]
            )
            self.players[2] = (
                None if p2_index == -1 else self.agent_players_config[p2_index]
            )

            self.game_engine.reset_game(
                self.players, self.game_settings["banned_moves_enabled"]
            )
            self.start_game_loop()

    def gameover_assert(self, winner: int):
        self.save_game_history(winner)
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Game Over")
        winner_name = "No one"
        if winner in self.players:
            winner_name = (
                self.players[winner]["name"] if self.players[winner] else "Human"
            )
        msg_box.setText(f"<h2>{winner_name} Wins!</h2>")
        msg_box.setInformativeText("What would you like to do next?")
        msg_box.setIcon(QMessageBox.Icon.Question)
        play_again_btn = msg_box.addButton(
            "Play Again", QMessageBox.ButtonRole.AcceptRole
        )
        msg_box.addButton("Review Board", QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(play_again_btn)
        msg_box.exec_()
        if msg_box.clickedButton() == play_again_btn:
            self.prompt_new_game()

    def rules_description(self):
        rules_text = """
            <h3>Freestyle Rules:</h3>
            The first player to get an unbroken row of five stones wins.<br>
            <h3>Renju Rules (Banned Moves Enabled):</h3>
            To balance the game, the Black player (first player) cannot make moves that create:<br>
            - <b>Three-Three:</b> Two open lines of three stones simultaneously.<br>
            - <b>Four-Four:</b> Two lines of four stones simultaneously.<br>
            - <b>Overline:</b> A line of six or more stones. This results in a loss for Black.<br>
            <br><i>These restrictions do not apply to the White player.</i>
        """
        QMessageBox.information(self, "Gomoku Rules", rules_text)

    def agent_gameover_handler(self, message: str):
        self.stop_game_loop()
        self.game_engine.canvas.game_over = True
        self.game_engine.canvas.is_in_game = False
        if self.game_engine.is_paused:
            self.game_engine.is_paused = False
            self.game_engine.canvas.setEnabled(True)
        self.game_engine.pause_button.hide()
        self.game_engine.resume_button.hide()
        self.game_engine.info_label.setText(f"<i>{message}</i>")

    def save_game_history(self, winner_player_id):
        if not os.path.exists("game_history"):
            os.makedirs("game_history")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./game_history/game_{timestamp}.json"
        p1_name = self.players[1]["name"] if self.players[1] else "Human"
        p2_name = self.players[2]["name"] if self.players[2] else "Human"
        winner_name = "None"
        if winner_player_id in self.players:
            winner_name = (
                self.players[winner_player_id]["name"]
                if self.players[winner_player_id]
                else "Human"
            )

        history_data = {
            "game_id": timestamp,
            "winner_player_id": winner_player_id,
            "winner_name": winner_name,
            "players": {"1": {"type": p1_name}, "2": {"type": p2_name}},
            "rules": self.game_settings,
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
