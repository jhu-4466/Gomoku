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


# Game Phases for Swap2
class GamePhase:
    NORMAL = 0
    SWAP2_P1_PLACE_3 = 1
    SWAP2_P2_CHOOSE_ACTION = 2
    SWAP2_P2_PLACE_2 = 3
    SWAP2_P1_CHOOSE_COLOR = 4


class GomokuCanvas(QWidget):
    gameOverSignal = pyqtSignal(int)
    stateChangedSignal = pyqtSignal(dict)
    requestP2ChoiceSignal = pyqtSignal()
    requestP1ColorChoiceSignal = pyqtSignal()

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

        # --- State variables ---
        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.current_player_id = 1
        self.game_over = False
        self.winner_id = None
        self.move_history = []
        self.is_in_game = False

        # --- Player and Color Mapping ---
        self.player_names = {1: "Player 1", 2: "Player 2"}  # Static map of ID to name
        self.player_colors = {
            1: 1,
            2: 2,
        }  # Dynamic map of ID to color (1=Black, 2=White)

        # --- Rule settings ---
        self.banned_moves_enabled = False
        self.swap2_enabled = False
        self.banned_point_on_click = None
        self.banned_point_on_hover = None
        self.game_phase = GamePhase.NORMAL
        self.swap2_stones_to_place = []

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

    def reset_game(self, player_names, rules):
        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.current_player_id = 1
        self.game_over = False
        self.winner_id = None
        self.move_history = []
        self.player_names = player_names.copy()
        self.player_colors = {1: 1, 2: 2}  # Reset to default
        self.is_in_game = True

        self.banned_moves_enabled = rules.get("banned_moves_enabled", False)
        self.swap2_enabled = rules.get("swap2_enabled", False)

        if self.swap2_enabled:
            self.game_phase = GamePhase.SWAP2_P1_PLACE_3
            self.swap2_stones_to_place = [1, 1, 2]  # 2 Black, 1 White
        else:
            self.game_phase = GamePhase.NORMAL

        self.state_change()
        self.update()

    def mousePressEvent(self, event):
        if self.game_over or not self.is_in_game:
            return

        if self.player_names.get(self.current_player_id) != "Human":
            return

        pos = (event.pos().x(), event.pos().y())
        row = round((pos[1] - BOARD_MARGIN) / CELL_SIZE)
        col = round((pos[0] - BOARD_MARGIN) / CELL_SIZE)
        if not (
            0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0
        ):
            return

        if self.game_phase == GamePhase.NORMAL:
            self.handle_normal_move(row, col)
        elif self.game_phase in [
            GamePhase.SWAP2_P1_PLACE_3,
            GamePhase.SWAP2_P2_PLACE_2,
        ]:
            self.handle_swap2_placement(row, col)

    # BUG FIX: New method to correctly route agent moves
    def agent_place_stone(self, row, col):
        """Public method for the agent handler to place a stone."""
        if self.game_phase in [GamePhase.SWAP2_P1_PLACE_3, GamePhase.SWAP2_P2_PLACE_2]:
            self.handle_swap2_placement(row, col)
        else:  # Assumes GamePhase.NORMAL
            self.handle_normal_move(row, col)

    def handle_swap2_placement(self, row, col):
        if not self.swap2_stones_to_place:
            print("Warning: handle_swap2_placement called with no stones to place.")
            return

        stone_color = self.swap2_stones_to_place.pop(0)
        self.board[row][col] = stone_color

        # Determine which player is making the move for history
        turn_player_id = 1 if self.game_phase == GamePhase.SWAP2_P1_PLACE_3 else 2
        self.move_history.append(
            {
                "turn": len(self.move_history) + 1,
                "player_id": turn_player_id,
                "move": (row, col),
                "color": stone_color,
            }
        )
        self.update()

        if not self.swap2_stones_to_place:
            if self.game_phase == GamePhase.SWAP2_P1_PLACE_3:
                self.game_phase = GamePhase.SWAP2_P2_CHOOSE_ACTION
                self.current_player_id = 2
                self.requestP2ChoiceSignal.emit()
            elif self.game_phase == GamePhase.SWAP2_P2_PLACE_2:
                self.game_phase = GamePhase.SWAP2_P1_CHOOSE_COLOR
                self.current_player_id = 1
                self.requestP1ColorChoiceSignal.emit()
        self.state_change()

    def handle_p2_choice(self, choice):
        if choice == "TAKE_BLACK":
            self.player_colors = {1: 2, 2: 1}
            self.current_player_id = 1
        elif choice == "TAKE_WHITE":
            self.player_colors = {1: 1, 2: 2}
            self.current_player_id = 2
        elif choice == "PLACE_2":
            self.game_phase = GamePhase.SWAP2_P2_PLACE_2
            self.swap2_stones_to_place = [1, 2]

        if choice != "PLACE_2":
            self.game_phase = GamePhase.NORMAL
        self.state_change()

    def handle_p1_final_choice(self, color_choice):
        if color_choice == "CHOOSE_BLACK":
            self.player_colors = {1: 1, 2: 2}
            self.current_player_id = 2
        else:
            self.player_colors = {1: 2, 2: 1}
            self.current_player_id = 1
        self.game_phase = GamePhase.NORMAL
        self.state_change()

    def handle_normal_move(self, row, col):
        if self.board[row][col] != 0:
            print(
                f"Warning: Attempted to place piece on occupied cell ({row}, {col}). Ignoring."
            )
            return

        color_to_play = self.player_colors[self.current_player_id]
        if self.banned_moves_enabled and color_to_play == 1:
            is_banned, reason = self.check_banned_move(row, col)
            if is_banned:
                self.banned_point_on_click = (row, col)
                self.update()
                QMessageBox.warning(
                    self,
                    "Banned Move",
                    f"This move ({reason}) is not allowed for Black.",
                )
                return

        # Place piece and check for win/end of game
        self.board[row][col] = color_to_play
        self.move_history.append(
            {
                "turn": len(self.move_history) + 1,
                "player_id": self.current_player_id,
                "move": (row, col),
                "color": color_to_play,
            }
        )

        if self.check_win(row, col, color_to_play):
            self.game_over = True
            self.is_in_game = False
            self.winner_id = self.current_player_id
            self.gameOverSignal.emit(self.winner_id)
        elif (
            self.banned_moves_enabled
            and color_to_play == 1
            and self.check_overline(row, col, color_to_play)
        ):
            self.game_over = True
            self.is_in_game = False
            self.winner_id = 3 - self.current_player_id
            self.gameOverSignal.emit(self.winner_id)
        else:
            self.current_player_id = 3 - self.current_player_id

        self.state_change()
        self.update()

    def get_current_state(self):
        return {
            "player_names": self.player_names,
            "player_colors": self.player_colors,
            "current_player_id": self.current_player_id,
            "move_count": len(self.move_history),
            "game_over": self.game_over,
            "winner_id": self.winner_id,
            "game_phase": self.game_phase,
            "swap2_stones_to_place": self.swap2_stones_to_place,
        }

    def mouseMoveEvent(self, event):
        color_to_play = self.player_colors.get(self.current_player_id, 1)
        is_black_human_turn = (
            not self.game_over
            and self.is_in_game
            and self.banned_moves_enabled
            and self.player_names.get(self.current_player_id) == "Human"
            and color_to_play == 1
            and self.game_phase == GamePhase.NORMAL
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

    def check_banned_move(self, r, c):
        if self.board[r][c] != 0:
            return False, ""
        self.board[r][c] = 1
        if self.check_overline(r, c, 1):
            self.board[r][c] = 0
            return True, "Overline"
        threes, fours = 0, 0
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            line = [
                (
                    self.board[r + i * dr][c + i * dc]
                    if 0 <= r + i * dr < 15 and 0 <= c + i * dc < 15
                    else -1
                )
                for i in range(-4, 5)
            ]
            line_str = "".join(map(str, line)).replace("2", "0")
            if "01110" in line_str:
                threes += 1
            if "1111" in line_str:
                fours += 1
        self.board[r][c] = 0
        if threes >= 2:
            return True, "Three-Three"
        if fours >= 2:
            return True, "Four-Four"
        return False, ""

    def check_overline(self, r, c, p):
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 6):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < 15 and 0 <= nc < 15 and self.board[nr][nc] == p:
                    count += 1
                else:
                    break
            for i in range(1, 6):
                nr, nc = r - i * dr, c - i * dc
                if 0 <= nr < 15 and 0 <= nc < 15 and self.board[nr][nc] == p:
                    count += 1
                else:
                    break
            if count > 5:
                return True
        return False

    def check_win(self, r, c, p):
        for d_row, d_col in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 5):
                nr, nc = r + i * d_row, c + i * d_col
                if 0 <= nr < 15 and 0 <= nc < 15 and self.board[nr][nc] == p:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - i * d_row, c - i * d_col
                if 0 <= nr < 15 and 0 <= nc < 15 and self.board[nr][nc] == p:
                    count += 1
                else:
                    break
            if count == 5:
                return True
        return False

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
        self.canvas.requestP2ChoiceSignal.connect(self.prompt_p2_choice)
        self.canvas.requestP1ColorChoiceSignal.connect(self.prompt_p1_color_choice)

    def update_info_bar(self, state: dict):
        if self.is_paused:
            text = "<i>Game Paused</i>"
        elif state["game_over"]:
            winner_name = "No one"
            if state["winner_id"] is not None:
                winner_name = state["player_names"].get(state["winner_id"], "Unknown")
            text = f"<b>Game Over! Winner is {winner_name}.</b>"
            self.pause_button.hide()
        elif state["game_phase"] != GamePhase.NORMAL:
            text = self.get_swap2_info_text(state)
            self.pause_button.hide()
        else:
            p1_name = state["player_names"][1]
            p2_name = state["player_names"][2]
            p1_color_str = "Black" if state["player_colors"][1] == 1 else "White"
            p2_color_str = "Black" if state["player_colors"][2] == 1 else "White"
            current_player_id = state["current_player_id"]
            current_player_color_str = (
                "Black" if state["player_colors"][current_player_id] == 1 else "White"
            )

            text = f"<b>{p1_name} ({p1_color_str})</b> vs <b>{p2_name} ({p2_color_str})</b> | Turn {state['move_count'] + 1}: <b>{current_player_color_str}</b> to move"
            self.pause_button.show()
        self.info_label.setText(text)
        self.info_label.setText(text)

    def get_swap2_info_text(self, state):
        phase = state["game_phase"]

        if phase == GamePhase.SWAP2_P1_PLACE_3:
            player_name = state["player_names"][1]
            return f"<b>Swap2:</b> {player_name} places 3 stones."
        elif phase == GamePhase.SWAP2_P2_CHOOSE_ACTION:
            player_name = state["player_names"][2]
            return f"<b>Swap2:</b> {player_name}, make your choice."
        elif phase == GamePhase.SWAP2_P2_PLACE_2:
            player_name = state["player_names"][2]
            return f"<b>Swap2:</b> {player_name} places 2 more stones."
        elif phase == GamePhase.SWAP2_P1_CHOOSE_COLOR:
            player_name = state["player_names"][1]
            return f"<b>Swap2:</b> {player_name}, choose your final color."
        return ""

    def prompt_p2_choice(self):
        if self.canvas.player_names.get(2) == "Human":
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Player 2's Choice")
            dialog.setText("Player 2, choose your action:")
            dialog.setIcon(QMessageBox.Question)
            black_btn = dialog.addButton("Take Black", QMessageBox.AcceptRole)
            white_btn = dialog.addButton("Take White", QMessageBox.AcceptRole)
            swap_btn = dialog.addButton("Place 2 More Stones", QMessageBox.RejectRole)
            dialog.exec_()
            if dialog.clickedButton() == black_btn:
                self.canvas.handle_p2_choice("TAKE_BLACK")
            elif dialog.clickedButton() == white_btn:
                self.canvas.handle_p2_choice("TAKE_WHITE")
            else:
                self.canvas.handle_p2_choice("PLACE_2")

    def prompt_p1_color_choice(self):
        if self.canvas.player_names.get(1) == "Human":
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Player 1's Final Choice")
            dialog.setText("Player 1, choose your color for the rest of the game:")
            dialog.setIcon(QMessageBox.Question)
            black_btn = dialog.addButton("Choose Black", QMessageBox.AcceptRole)
            white_btn = dialog.addButton("Choose White", QMessageBox.AcceptRole)
            dialog.exec_()
            if dialog.clickedButton() == black_btn:
                self.canvas.handle_p1_final_choice("CHOOSE_BLACK")
            else:
                self.canvas.handle_p1_final_choice("CHOOSE_WHITE")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.canvas.setEnabled(not self.is_paused)
        self.update_info_bar(self.canvas.get_current_state())

    def reset_game(self, player_names, rules):
        if self.is_paused:
            self.toggle_pause()
        self.canvas.reset_game(player_names, rules)

    def reconnectstatus_show(self, message: str):
        self.info_label.setText(f"<i>{message}</i>")
        self.pause_button.hide()


class GomokuAgentHandler(QThread):
    movedSignal = pyqtSignal(int, int)
    p2ChoiceSignal = pyqtSignal(str)
    p1ChoiceSignal = pyqtSignal(str)
    reconnectSignal = pyqtSignal(str)
    gameOverSignal = pyqtSignal(str)

    def __init__(self, game_canvas, player_configs):
        super().__init__()
        self.game_canvas = game_canvas
        self.player_configs = player_configs
        self.running = True

    def run(self):
        while self.running and not self.game_canvas.game_over:
            if not self.running:
                break

            state = self.game_canvas.get_current_state()
            player_id = state["current_player_id"]
            player_name = state["player_names"][player_id]

            if player_name == "Human":
                self.msleep(100)
                continue

            player_config = self.player_configs[player_id]
            game_phase = state["game_phase"]

            # AI Action Logic
            if game_phase == GamePhase.SWAP2_P2_CHOOSE_ACTION and player_id == 2:
                self.get_ai_swap2_choice(player_config, state, "P2_CHOOSE")
            elif game_phase == GamePhase.SWAP2_P1_CHOOSE_COLOR and player_id == 1:
                self.get_ai_swap2_choice(player_config, state, "P1_CHOOSE")
            else:
                # All other phases are move-making phases for the current player
                self.get_ai_move(player_config, state)

            if self.game_canvas.game_over:
                break
            # Add a small delay to make AI moves more observable
            self.msleep(300)

    def get_ai_swap2_choice(self, player_config, state, choice_type):
        payload = self.build_payload(state)
        payload["game_phase"] = choice_type

        choice = self.make_request(player_config, payload).get("choice")

        if choice and self.running:
            if choice_type == "P2_CHOOSE":
                self.p2ChoiceSignal.emit(choice)
            elif choice_type == "P1_CHOOSE":
                self.p1ChoiceSignal.emit(choice)

    def get_ai_move(self, player_config, state):
        payload = self.build_payload(state)
        move = self.make_request(player_config, payload).get("move")

        if move and self.running:
            row, col = move[0], move[1]
            if self.game_canvas.board[row][col] != 0:
                print(
                    f"Error: AI from {player_config['url']} returned an occupied cell {move}."
                )
                self.gameOverSignal.emit(f"Error: AI returned invalid move.")
                return
            self.movedSignal.emit(row, col)

    def build_payload(self, state):
        player_id = state["current_player_id"]
        return {
            "board": self.game_canvas.board,
            "player_id": player_id,
            "color_to_play": state["player_colors"][player_id],
            "banned_moves_enabled": self.game_canvas.banned_moves_enabled,
            "game_phase": state["game_phase"],
            "move_history": self.game_canvas.move_history,
        }

    def make_request(self, player_config, payload):
        player_url = player_config["url"]
        player_timeout = player_config.get("timeout", 5)
        for attempt in range(MAX_RETRIES):
            if not self.running:
                return {}
            try:
                response = requests.post(
                    player_url, json=payload, timeout=player_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                print(f"Request failed for {player_config['name']}: {e}")
                if attempt < MAX_RETRIES - 1:
                    reconnect_notice = (
                        f"Connection lost. Retrying... ({attempt + 1}/{MAX_RETRIES})"
                    )
                    self.reconnectSignal.emit(reconnect_notice)
                    self.msleep(RETRY_DELAY_S * 1000)
                else:
                    self.gameOverSignal.emit(
                        f"Error: Cannot connect to {player_config['name']}."
                    )
                    self.running = False
        return {}

    def stop(self):
        self.running = False


class NewGameDialog(QDialog):
    def __init__(self, agent_players, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start a New Game")
        main_layout = QVBoxLayout(self)

        # --- Player Selection ---
        player_selection_layout = QHBoxLayout()
        self.left_player_group = QGroupBox("Left Player")
        left_player_layout = QVBoxLayout(self.left_player_group)
        self.right_player_group = QGroupBox("Right Player")
        right_player_layout = QVBoxLayout(self.right_player_group)

        self.left_player_buttons, self.right_player_buttons = [], []

        # Left Player Buttons
        human_left_radio = QRadioButton("Human")
        human_left_radio.setChecked(True)
        self.left_player_buttons.append({"name": "Human", "radio": human_left_radio})
        left_player_layout.addWidget(human_left_radio)
        for agent in agent_players:
            radio = QRadioButton(agent["name"])
            self.left_player_buttons.append({"name": agent["name"], "radio": radio})
            left_player_layout.addWidget(radio)

        # Right Player Buttons
        human_right_radio = QRadioButton("Human")
        self.right_player_buttons.append({"name": "Human", "radio": human_right_radio})
        right_player_layout.addWidget(human_right_radio)
        for i, agent in enumerate(agent_players):
            radio = QRadioButton(agent["name"])
            if i == 0:
                radio.setChecked(True)  # Default to first AI
            self.right_player_buttons.append({"name": agent["name"], "radio": radio})
            right_player_layout.addWidget(radio)

        left_player_layout.addStretch()
        right_player_layout.addStretch()
        player_selection_layout.addWidget(self.left_player_group)
        player_selection_layout.addWidget(self.right_player_group)
        main_layout.addLayout(player_selection_layout)

        # --- First Player Selection ---
        first_player_group = QGroupBox("First Player (Black)")
        first_player_layout = QHBoxLayout(first_player_group)
        self.first_player_left = QRadioButton("Left Player")
        self.first_player_right = QRadioButton("Right Player")
        self.first_player_random = QRadioButton("Random")
        self.first_player_left.setChecked(True)
        first_player_layout.addWidget(self.first_player_left)
        first_player_layout.addWidget(self.first_player_right)
        first_player_layout.addWidget(self.first_player_random)
        main_layout.addWidget(first_player_group)

        # --- Rules ---
        rules_group = QGroupBox("Game Rules")
        rules_layout = QVBoxLayout(rules_group)
        self.banned_checkbox = QCheckBox("Enable Banned Moves (Renju Rules)")
        self.swap2_checkbox = QCheckBox("Enable Swap2 Opening Rule")
        self.swap2_checkbox.setToolTip(
            "A fair opening rule. Player 1 places 3 stones, Player 2 chooses a side or places 2 more."
        )
        rules_layout.addWidget(self.banned_checkbox)
        rules_layout.addWidget(self.swap2_checkbox)
        main_layout.addWidget(rules_group)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
        self.swap2_checkbox.toggled.connect(self.on_swap2_toggled)

    def on_swap2_toggled(self, checked):
        if checked:
            self.banned_checkbox.setChecked(True)
            self.banned_checkbox.setEnabled(False)
        else:
            self.banned_checkbox.setEnabled(True)

    def get_settings(self):
        left_player_name = next(
            b["name"] for b in self.left_player_buttons if b["radio"].isChecked()
        )
        right_player_name = next(
            b["name"] for b in self.right_player_buttons if b["radio"].isChecked()
        )

        first_player = "Left"
        if self.first_player_right.isChecked():
            first_player = "Right"
        elif self.first_player_random.isChecked():
            first_player = "Random"

        return {
            "left_player": left_player_name,
            "right_player": right_player_name,
            "first_player": first_player,
            "rules": {
                "banned_moves_enabled": self.banned_checkbox.isChecked(),
                "swap2_enabled": self.swap2_checkbox.isChecked(),
            },
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
        self.player_configs = {1: None, 2: None}
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
        if not self.player_configs[1] and not self.player_configs[2]:
            return
        self.agent_handler = GomokuAgentHandler(
            self.game_engine.canvas, self.player_configs
        )
        # BUG FIX: Connect to the new agent_place_stone method
        self.agent_handler.movedSignal.connect(
            self.game_engine.canvas.agent_place_stone
        )
        self.agent_handler.p2ChoiceSignal.connect(
            self.game_engine.canvas.handle_p2_choice
        )
        self.agent_handler.p1ChoiceSignal.connect(
            self.game_engine.canvas.handle_p1_final_choice
        )
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

            name_to_config = {cfg["name"]: cfg for cfg in self.agent_players_config}
            left_config = name_to_config.get(self.game_settings["left_player"])
            right_config = name_to_config.get(self.game_settings["right_player"])

            first_player_choice = self.game_settings["first_player"]
            if first_player_choice == "Random":
                first_player_choice = random.choice(["Left", "Right"])

            if first_player_choice == "Left":
                p1_name, p2_name = (
                    self.game_settings["left_player"],
                    self.game_settings["right_player"],
                )
                p1_config, p2_config = left_config, right_config
            else:  # Right
                p1_name, p2_name = (
                    self.game_settings["right_player"],
                    self.game_settings["left_player"],
                )
                p1_config, p2_config = right_config, left_config

            final_player_names = {1: p1_name, 2: p2_name}
            self.player_configs = {1: p1_config, 2: p2_config}

            self.game_engine.reset_game(final_player_names, self.game_settings["rules"])
            self.start_game_loop()

    def gameover_assert(self, winner_id: int):
        self.stop_game_loop()
        self.save_game_history(winner_id)
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Game Over")

        winner_name = "No one"
        if winner_id is not None:
            winner_name = self.game_engine.canvas.player_names.get(winner_id, "Unknown")

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
            <h3>Freestyle Rules:</h3> The first player to get an unbroken row of five stones wins.<br>
            <h3>Renju Rules (Banned Moves Enabled):</h3> To balance the game, the Black player (first player) cannot make moves that create:<br>
            - <b>Three-Three, Four-Four, or Overline.</b><br>
            <h3>Swap2 Opening Rule:</h3> A fair opening protocol:<br>
            1. Player 1 places 2 black stones and 1 white stone.<br>
            2. Player 2 can: (a) Take Black, (b) Take White, or (c) Place 2 more stones and pass the choice back.<br>
            3. If (c), Player 1 chooses their final color.<br>
            <br><i>Swap2 is always played with Renju rules.</i>
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

    def save_game_history(self, winner_id):
        if not os.path.exists("game_history"):
            os.makedirs("game_history")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./game_history/game_{timestamp}.json"

        player_names = self.game_engine.canvas.player_names
        winner_name = "None"
        if winner_id is not None:
            winner_name = player_names.get(winner_id, "Unknown")

        history_data = {
            "game_id": timestamp,
            "winner_id": winner_id,
            "winner_name": winner_name,
            "player_setup": {
                "p1_name": player_names[1],
                "p2_name": player_names[2],
            },
            "final_colors": self.game_engine.canvas.player_colors,
            "rules": self.game_settings.get("rules", {}),
            "move_history": self.game_engine.canvas.move_history,
        }
        try:
            with open(filename, "w") as f:
                json.dump(history_data, f, indent=4)
            print(f"Error saving game history: {filename}")
        except Exception as e:
            print(f"Error saving game history: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GomokuApp()
    main_window.show()
    sys.exit(app.exec_())
