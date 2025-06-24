import pygame
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QAction,
)
from PyQt5.QtGui import QImage, QPainter, QIcon
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

# Board attributes
PYGAME_CANVAS_WIDTH = 780
PYGAME_CANVAS_HEIGHT = 780
GRID_SIZE = 19
CELL_SIZE = 40
BOARD_MARGIN = 30
BOARD_WIDTH = CELL_SIZE * (GRID_SIZE - 1)
BOARD_HEIGHT = BOARD_WIDTH

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


class GomokuBoard(QWidget):
    gameOverSignal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT)

        pygame.init()
        self.screen = pygame.Surface((PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT))

        self.font_small = pygame.font.SysFont("sans", 20)
        self.font_large = pygame.font.SysFont("sans", 50)

        self.reset_game()

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
        if self.game_over:
            overlay = pygame.Surface(
                (PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT), pygame.SRCALPHA
            )
            overlay.fill(COLOR_WINNER_BG)
            self.screen.blit(overlay, (0, 0))

    def reset_game(self):
        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.update()

    def undo_move(self):
        if self.move_history and not self.game_over:
            last_row, last_col = self.move_history.pop()
            self.board[last_row][last_col] = 0
            self.current_player = 3 - self.current_player
            print("Undo last move.")
            self.update()

    def handle_board_click(self, pos):
        if self.game_over:
            return

        board_rect_in_canvas = pygame.Rect(
            BOARD_MARGIN, BOARD_MARGIN, BOARD_WIDTH, BOARD_HEIGHT
        )
        if board_rect_in_canvas.collidepoint(pos):
            x, y = pos
            row = round((y - board_rect_in_canvas.top) / CELL_SIZE)
            col = round((x - board_rect_in_canvas.left) / CELL_SIZE)
            if (
                0 <= row < GRID_SIZE
                and 0 <= col < GRID_SIZE
                and self.board[row][col] == 0
            ):
                self.board[row][col] = self.current_player
                self.move_history.append((row, col))
                if self.check_win(row, col):
                    self.game_over = True
                    self.winner = self.current_player
                    self.gameOverSignal.emit("Black" if self.winner == 1 else "White")
                else:
                    self.current_player = 3 - self.current_player
                self.update()

    def mousePressEvent(self, event):
        pos = (event.pos().x(), event.pos().y())
        self.handle_board_click(pos)

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
        board_widget = pygame.Rect(
            BOARD_MARGIN, BOARD_MARGIN, BOARD_WIDTH, BOARD_HEIGHT
        )
        for i in range(GRID_SIZE):
            pygame.draw.line(
                self.screen,
                COLOR_LINE,
                (board_widget.left + i * CELL_SIZE, board_widget.top),
                (board_widget.left + i * CELL_SIZE, board_widget.bottom),
            )
            pygame.draw.line(
                self.screen,
                COLOR_LINE,
                (board_widget.left, board_widget.top + i * CELL_SIZE),
                (board_widget.right, board_widget.top + i * CELL_SIZE),
            )
        for r, c in STAR_POINTS:
            pygame.draw.circle(
                self.screen,
                COLOR_LINE,
                (board_widget.left + c * CELL_SIZE, board_widget.top + r * CELL_SIZE),
                6,
            )

    def draw_pieces(self):
        board_widget = pygame.Rect(
            BOARD_MARGIN, BOARD_MARGIN, BOARD_WIDTH, BOARD_HEIGHT
        )
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.board[r][c] != 0:
                    piece = (
                        board_widget.left + c * CELL_SIZE,
                        board_widget.top + r * CELL_SIZE,
                    )
                    piece_color = COLOR_BLACK if self.board[r][c] == 1 else COLOR_WHITE
                    pygame.draw.circle(
                        self.screen, piece_color, piece, CELL_SIZE // 2 - 2
                    )

    # def render_winner_message(self):
    # ov = pygame.Surface(
    #     (PYGAME_CANVAS_WIDTH, PYGAME_CANVAS_HEIGHT), pygame.SRCALPHA
    # )
    # ov.fill(COLOR_WINNER_BG)
    # self.screen.blit(ov, (0, 0))
    # wt = "Black Wins!" if self.winner == 1 else "White Wins!"
    # ts = self.font_large.render(wt, True, COLOR_WHITE)
    # tr = ts.get_rect(center=(PYGAME_CANVAS_WIDTH / 2, PYGAME_CANVAS_HEIGHT / 2))
    # self.screen.blit(ts, tr)


class GomokuApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gomoku(Five in a Row)")

        # Board Widget
        self.game_widget = GomokuBoard(self)
        self.setCentralWidget(self.game_widget)
        self.game_widget.gameOverSignal.connect(self.gameover_assert)

        # Menu Bar
        menu_bar = self.menuBar()
        game_menu = menu_bar.addMenu("&Game")
        help_menu = menu_bar.addMenu("&Help")
        new_game_action = QAction("New Game", self)
        undo_action = QAction("Undo", self)
        exit_action = QAction("Exit", self)
        rules_action = QAction("Game Rules", self)

        new_game_action.triggered.connect(self.game_widget.reset_game)
        undo_action.triggered.connect(self.game_widget.undo_move)
        exit_action.triggered.connect(self.close)
        rules_action.triggered.connect(self.rules_description)

        game_menu.addAction(new_game_action)
        game_menu.addAction(undo_action)
        game_menu.addAction(exit_action)
        help_menu.addAction(rules_action)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_widget.update)
        self.timer.start(16)  # ~60 FPS

    def gameover_assert(self, winner):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Game Over")
        msg_box.setText(f"<h2>{winner} Wins!</h2>")
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
            self.game_widget.reset_game()

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = GomokuApp()
    main_window.show()
    sys.exit(app.exec_())
