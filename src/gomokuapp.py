# -*- coding: utf-8 -*-
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
    QSpinBox,
    QFormLayout,
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
from collections import defaultdict


# Board attributes (15x15)
PYGAME_CANVAS_WIDTH = 620
PYGAME_CANVAS_HEIGHT = 690
GRID_SIZE = 15
CELL_SIZE = 40
BOARD_MARGIN = 30
BOARD_WIDTH = CELL_SIZE * (GRID_SIZE - 1)
BOARD_HEIGHT = BOARD_WIDTH
INFO_BAR_HEIGHT = 70

# Colors
# 1 - Black, 2 - White
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
        self.batch_info = None

        # --- Player and Color Mapping ---
        self.player_names = {1: "Player 1", 2: "Player 2"}
        self.player_colors = {1: 1, 2: 2}

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
        self.player_colors = {1: 1, 2: 2}
        self.is_in_game = True

        self.banned_moves_enabled = rules.get("banned_moves_enabled", False)
        self.swap2_enabled = rules.get("swap2_enabled", False)
        self.batch_info = rules.get("batch_info", None)

        if self.swap2_enabled:
            self.game_phase = GamePhase.SWAP2_P1_PLACE_3
            self.swap2_stones_to_place = [1, 1, 2]
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

    def agent_place_stone(self, row, col, thinking_time):
        if self.game_phase in [GamePhase.SWAP2_P1_PLACE_3, GamePhase.SWAP2_P2_PLACE_2]:
            self.handle_swap2_placement(row, col, thinking_time)
        else:
            self.handle_normal_move(row, col, thinking_time=thinking_time)

    def handle_swap2_placement(self, row, col, thinking_time=None):
        if not self.swap2_stones_to_place:
            print("Warning: handle_swap2_placement called with no stones to place.")
            return

        stone_color = self.swap2_stones_to_place.pop(0)
        self.board[row][col] = stone_color

        turn_player_id = 1 if self.game_phase == GamePhase.SWAP2_P1_PLACE_3 else 2

        move_data = {
            "turn": len(self.move_history) + 1,
            "player_id": turn_player_id,
            "move": (row, col),
            "color": stone_color,
        }
        if thinking_time is not None:
            move_data["thinking_time"] = round(thinking_time, 4)
        self.move_history.append(move_data)

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

    def handle_normal_move(self, row, col, thinking_time=None):
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

        self.board[row][col] = color_to_play

        move_data = {
            "turn": len(self.move_history) + 1,
            "player_id": self.current_player_id,
            "move": (row, col),
            "color": color_to_play,
        }
        if thinking_time is not None:
            move_data["thinking_time"] = round(thinking_time, 4)
        self.move_history.append(move_data)

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
        state = {
            "player_names": self.player_names,
            "player_colors": self.player_colors,
            "current_player_id": self.current_player_id,
            "move_count": len(self.move_history),
            "game_over": self.game_over,
            "winner_id": self.winner_id,
            "game_phase": self.game_phase,
            "swap2_stones_to_place": self.swap2_stones_to_place,
        }
        if self.batch_info:
            state["batch_info"] = self.batch_info
        return state

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

        self.main_info_label = QLabel("Start a new game from the 'Game' menu.")
        self.main_info_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.sub_info_label = QLabel()
        self.sub_info_label.setFont(QFont("Arial", 10))

        self.pause_button = QPushButton("Pause", self)
        self.pause_button.setFixedWidth(80)
        self.pause_button.hide()
        self.resume_button = QPushButton("Resume", self)
        self.resume_button.setFixedWidth(80)
        self.resume_button.hide()
        self.canvas = GomokuCanvas(self)

        info_bar_container = QWidget()
        info_bar_container.setFixedHeight(INFO_BAR_HEIGHT)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)
        text_layout.addWidget(self.main_info_label)
        text_layout.addWidget(self.sub_info_label)

        info_layout = QHBoxLayout(info_bar_container)
        info_layout.setContentsMargins(10, 0, 10, 0)
        info_layout.addLayout(text_layout)
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
        batch_info = state.get("batch_info")
        batch_text = ""
        if batch_info:
            batch_text = f"[Batch {batch_info['current']}/{batch_info['total']}] "

        main_text = ""
        sub_text = ""

        if self.is_paused:
            main_text = f"{batch_text}Game Paused"
            self.pause_button.hide()
            self.resume_button.show()
        else:
            self.resume_button.hide()
            if state["game_over"]:
                winner_name = "No one"
                if state["winner_id"] is not None:
                    winner_name = state["player_names"].get(
                        state["winner_id"], "Unknown"
                    )
                main_text = f"{batch_text}<b>Game Over! Winner is {winner_name}.</b>"
                self.pause_button.hide()
            elif state["game_phase"] != GamePhase.NORMAL:
                main_text = f"{batch_text}{self.get_swap2_info_text(state)}"
                self.pause_button.hide()
            else:
                p1_name = state["player_names"][1]
                p2_name = state["player_names"][2]
                p1_color_str = "Black" if state["player_colors"][1] == 1 else "White"
                p2_color_str = "Black" if state["player_colors"][2] == 1 else "White"
                current_player_id = state["current_player_id"]
                current_player_name = state["player_names"][current_player_id]
                current_player_color_str = (
                    "Black"
                    if state["player_colors"][current_player_id] == 1
                    else "White"
                )

                main_text = f"{batch_text}Turn {state['move_count'] + 1}: <b>{current_player_name}</b> ({current_player_color_str}) to move"
                sub_text = (
                    f"<b>{p1_name} ({p1_color_str}) vs {p2_name} ({p2_color_str})</b>"
                )
                self.pause_button.show()

        self.main_info_label.setText(main_text)
        self.sub_info_label.setText(sub_text)

    def get_swap2_info_text(self, state):
        phase = state["game_phase"]
        player_name = state["player_names"][state["current_player_id"]]
        if phase == GamePhase.SWAP2_P1_PLACE_3:
            return f"<b>Swap2:</b> {player_name} places 3 stones."
        elif phase == GamePhase.SWAP2_P2_CHOOSE_ACTION:
            return f"<b>Swap2:</b> {player_name}, make your choice."
        elif phase == GamePhase.SWAP2_P2_PLACE_2:
            return f"<b>Swap2:</b> {player_name} places 2 more stones."
        elif phase == GamePhase.SWAP2_P1_CHOOSE_COLOR:
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
        self.main_info_label.setText(f"{message}")
        self.sub_info_label.setText("")
        self.pause_button.hide()


class GomokuAgentHandler(QThread):
    movedSignal = pyqtSignal(int, int, float)
    p2ChoiceSignal = pyqtSignal(str)
    p1ChoiceSignal = pyqtSignal(str)
    reconnectSignal = pyqtSignal(str)
    gameOverSignal = pyqtSignal(str)

    def __init__(self, game_engine, player_configs):
        super().__init__()
        self.game_engine = game_engine
        self.player_configs = player_configs
        self.running = True

    def run(self):
        while self.running:
            if self.game_engine.is_paused or self.game_engine.canvas.game_over:
                self.msleep(200)
                continue

            state = self.game_engine.canvas.get_current_state()
            player_id = state["current_player_id"]
            player_name = state["player_names"].get(player_id)

            if player_name == "Human" or player_name is None:
                self.msleep(100)
                continue

            player_config = self.player_configs.get(player_id)
            if not player_config:
                self.msleep(100)
                continue

            game_phase = state["game_phase"]

            if game_phase == GamePhase.SWAP2_P2_CHOOSE_ACTION and player_id == 2:
                self.get_ai_swap2_choice(player_config, state, "P2_CHOOSE")
            elif game_phase == GamePhase.SWAP2_P1_CHOOSE_COLOR and player_id == 1:
                self.get_ai_swap2_choice(player_config, state, "P1_CHOOSE")
            else:
                self.get_ai_move(player_config, state)
            self.msleep(100)

    def get_ai_swap2_choice(self, player_config, state, choice_type):
        payload = self.build_payload(state)
        payload["game_phase"] = choice_type
        response_data, _ = self.make_request(player_config, payload)
        choice = response_data.get("choice")
        if choice and self.running:
            if choice_type == "P2_CHOOSE":
                self.p2ChoiceSignal.emit(choice)
            elif choice_type == "P1_CHOOSE":
                self.p1ChoiceSignal.emit(choice)

    def get_ai_move(self, player_config, state):
        payload = self.build_payload(state)
        response_data, thinking_time = self.make_request(player_config, payload)
        move = response_data.get("move")
        if move and self.running:
            row, col = move[0], move[1]
            if self.game_engine.canvas.board[row][col] != 0:
                self.gameOverSignal.emit(f"Error: AI returned invalid move.")
                return
            self.movedSignal.emit(row, col, thinking_time)

    def build_payload(self, state):
        return {
            "board": self.game_engine.canvas.board,
            "player_id": state["current_player_id"],
            "color_to_play": state["player_colors"][state["current_player_id"]],
            "banned_moves_enabled": self.game_engine.canvas.banned_moves_enabled,
            "game_phase": state["game_phase"],
            "move_history": self.game_engine.canvas.move_history,
        }

    def make_request(self, player_config, payload):
        player_url = player_config["url"]
        player_timeout = player_config.get("timeout", 10)
        for attempt in range(MAX_RETRIES):
            if not self.running:
                return {}, 0.0
            try:
                start_time = time.time()
                response = requests.post(
                    player_url, json=payload, timeout=player_timeout
                )
                thinking_time = time.time() - start_time
                response.raise_for_status()
                return response.json(), thinking_time
            except requests.RequestException as e:
                print(f"Request failed for {player_config['name']}: {e}")
                if attempt < MAX_RETRIES - 1:
                    self.reconnectSignal.emit(
                        f"Connection lost. Retrying... ({attempt + 1}/{MAX_RETRIES})"
                    )
                    self.msleep(RETRY_DELAY_S * 1000)
                else:
                    self.gameOverSignal.emit(
                        f"Error: Cannot connect to {player_config['name']}."
                    )
                    self.running = False
        return {}, 0.0

    def stop(self):
        self.running = False


class NewGameDialog(QDialog):
    def __init__(self, agent_players, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start a New Game")
        main_layout = QVBoxLayout(self)
        player_selection_layout = QHBoxLayout()

        self.left_player_group = QGroupBox("Player 1")
        left_player_layout = QVBoxLayout(self.left_player_group)
        self.left_player_buttons, self.right_player_buttons = [], []
        human_left_radio = QRadioButton("Human")
        human_left_radio.setChecked(True)
        self.left_player_buttons.append({"name": "Human", "radio": human_left_radio})
        left_player_layout.addWidget(human_left_radio)
        for agent in agent_players:
            radio = QRadioButton(agent["name"])
            self.left_player_buttons.append({"name": agent["name"], "radio": radio})
            left_player_layout.addWidget(radio)

        self.right_player_group = QGroupBox("Player 2")
        right_player_layout = QVBoxLayout(self.right_player_group)
        for i, agent in enumerate(agent_players):
            radio = QRadioButton(agent["name"])
            if i == 0:
                radio.setChecked(True)
            self.right_player_buttons.append({"name": agent["name"], "radio": radio})
            right_player_layout.addWidget(radio)
        left_player_layout.addStretch()
        right_player_layout.addStretch()
        player_selection_layout.addWidget(self.left_player_group)
        player_selection_layout.addWidget(self.right_player_group)
        main_layout.addLayout(player_selection_layout)
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
        rules_group = QGroupBox("Game Rules")
        rules_layout = QVBoxLayout(rules_group)
        self.banned_checkbox = QCheckBox("Enable Banned Moves (Renju Rules)")
        self.swap2_checkbox = QCheckBox("Enable Swap2 Opening Rule")
        rules_layout.addWidget(self.banned_checkbox)
        rules_layout.addWidget(self.swap2_checkbox)
        main_layout.addWidget(rules_group)
        self.batch_group = QGroupBox("Batch Play")
        batch_layout = QFormLayout(self.batch_group)
        self.num_games_spinbox = QSpinBox()
        self.num_games_spinbox.setMinimum(1)
        self.num_games_spinbox.setMaximum(1000)
        self.num_games_spinbox.setValue(1)
        batch_layout.addRow("Number of Games:", self.num_games_spinbox)
        main_layout.addWidget(self.batch_group)
        self.batch_group.setEnabled(False)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)
        self.swap2_checkbox.toggled.connect(self.on_swap2_toggled)
        for btn_data in self.left_player_buttons:
            btn_data["radio"].toggled.connect(self.update_batch_mode_availability)
        for btn_data in self.right_player_buttons:
            btn_data["radio"].toggled.connect(self.update_batch_mode_availability)
        self.update_batch_mode_availability()

    def update_batch_mode_availability(self):
        left_is_human = self.left_player_buttons[0]["radio"].isChecked()
        right_is_human = self.right_player_buttons[0]["radio"].isChecked()
        is_ai_vs_ai = not left_is_human and not right_is_human
        self.batch_group.setEnabled(is_ai_vs_ai)
        if not is_ai_vs_ai:
            self.num_games_spinbox.setValue(1)

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
            "num_games": self.num_games_spinbox.value(),
        }


class GomokuApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gomoku")
        self.game_engine = GomokuBoard(self)
        self.setCentralWidget(self.game_engine)
        self.game_engine.gameOverSignal.connect(self.gameover_handler)
        self._create_menu()
        self.agent_players_config = [
            {"name": "Random Strategy", "url": "http://127.0.0.1:5001/get_move"},
            {"name": "Baseline Strategy", "url": "http://127.0.0.1:5002/get_move"},
            {"name": "Negamax AI", "url": "http://127.0.0.1:5003/get_move"},
        ]
        self.agent_handler = None
        self.player_configs = {1: None, 2: None}
        self.game_settings = {}
        self.is_batch_mode = False
        self.batch_total_games = 0
        self.batch_current_game = 0
        self.batch_results = defaultdict(int)

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
        self.agent_handler = GomokuAgentHandler(self.game_engine, self.player_configs)
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
            num_games = self.game_settings.get("num_games", 1)
            left_is_human = self.game_settings["left_player"] == "Human"
            right_is_human = self.game_settings["right_player"] == "Human"
            if num_games > 1 and not left_is_human and not right_is_human:
                self.is_batch_mode = True
                self.batch_total_games = num_games
                self.batch_current_game = 1
                self.batch_results.clear()
                self.start_next_game_in_batch()
            else:
                self.is_batch_mode = False
                self.start_single_game()

    def start_single_game(self):
        self.setup_players_and_start()

    def start_next_game_in_batch(self):
        print(
            f"\n--- Starting Batch Game {self.batch_current_game} of {self.batch_total_games} ---"
        )
        self.setup_players_and_start(is_batch=True)

    def setup_players_and_start(self, is_batch=False):
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
        else:
            p1_name, p2_name = (
                self.game_settings["right_player"],
                self.game_settings["left_player"],
            )
            p1_config, p2_config = right_config, left_config
        self.player_configs = {1: p1_config, 2: p2_config}
        rules = self.game_settings["rules"].copy()
        if is_batch:
            rules["batch_info"] = {
                "current": self.batch_current_game,
                "total": self.batch_total_games,
            }
        self.game_engine.reset_game({1: p1_name, 2: p2_name}, rules)
        self.start_game_loop()

    def gameover_handler(self, winner_id: int):
        self.stop_game_loop()
        self.save_game_history(winner_id)
        if self.is_batch_mode:
            winner_name = self.game_engine.canvas.player_names.get(winner_id, "Draw")
            self.batch_results[winner_name] += 1
            if self.batch_current_game < self.batch_total_games:
                self.batch_current_game += 1
                QTimer.singleShot(500, self.start_next_game_in_batch)
            else:
                self.is_batch_mode = False
                summary_text = f"<h2>Batch of {self.batch_total_games} games complete!</h2><b>Results:</b><br>"
                for name, wins in self.batch_results.items():
                    summary_text += f"- {name}: {wins} wins<br>"
                QMessageBox.information(self, "Batch Complete", summary_text)
        else:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Game Over")
            winner_name = self.game_engine.canvas.player_names.get(winner_id, "No one")
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
            <h3>Freestyle Gomoku:</h3>
            <p>The first player to form an unbroken chain of five or more stones of their color wins.</p>
            <h3>Banned Moves Rules:</h3>
            <p>To balance the first-player advantage, the player with the black stones (the first player) is subject to several move restrictions, known as <b>banned moves</b>. Making a banned move results in a loss for Black. The White player has no restrictions. Banned moves include:</p>
            <ul>
                <li><b>Three-Three:</b> A move that simultaneously forms two or more open threes.</li>
                <li><b>Four-Four:</b> A move that simultaneously forms two or more open or semi-open fours.</li>
                <li><b>Overline:</b> A move that forms a continuous line of six or more stones of the same color.</li>
            </ul>
            <p><i>Note: If the White player forms an overline, it is a winning move, not a ban.</i></p>
            <h3>Swap2 Opening Rule:</h3>
            <p>This is a complex opening protocol designed to ensure fairness, which proceeds as follows:</p>
            <ol>
                <li><b>Step 1:</b> The first player (Player 1) places <b>two black stones and one white stone</b> anywhere on the board.</li>
                <li><b>Step 2:</b> It is now the second player's (Player 2) turn to choose:
                    <ul>
                        <li>(a) To play with the black stones.</li>
                        <li>(b) To play with the white stones.</li>
                        <li>(c) To place <b>two more stones</b> (one black, one white) on the board and pass the choice of color back to Player 1.</li>
                    </ul>
                </li>
                <li><b>Step 3:</b> If Player 2 chose option (c), it is now Player 1's turn to make the final decision on which color to play with (black or white) based on the five stones on the board.</li>
            </ol>
            <p><i><b>Important:</b> When the Swap2 rule is enabled, the banned moves rules (banned moves) are automatically enforced.</i></p>
        """
        QMessageBox.information(self, "Game Rules", rules_text)

    def agent_gameover_handler(self, message: str):
        self.stop_game_loop()
        self.game_engine.canvas.game_over = True
        self.game_engine.canvas.is_in_game = False
        if self.game_engine.is_paused:
            self.game_engine.is_paused = False
            self.game_engine.canvas.setEnabled(True)
        self.game_engine.pause_button.hide()
        self.game_engine.resume_button.hide()
        # FIX: Use the correct label attributes
        self.game_engine.main_info_label.setText(f"<i>{message}</i>")
        self.game_engine.sub_info_label.setText("")

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
            "player_setup": {"p1_name": player_names[1], "p2_name": player_names[2]},
            "final_colors": self.game_engine.canvas.player_colors,
            "rules": self.game_settings.get("rules", {}),
            "move_history": self.game_engine.canvas.move_history,
        }
        try:
            with open(filename, "w") as f:
                json.dump(history_data, f, indent=4)
            print(f"Game history saved to: {filename}")
        except Exception as e:
            print(f"Error saving game history: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GomokuApp()
    main_window.show()
    sys.exit(app.exec_())
