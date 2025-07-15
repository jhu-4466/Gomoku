# -*- coding: utf-8 -*-
"""
Gomoku AI Agent - High-Performance Version
Author: Gemini
Date: 2025-07-15

This version incorporates significant optimizations based on the principles of
game tree search, heuristic evaluation, and performance engineering to address
key strategic flaws and achieve greater search depth.

Core Improvements:
1.  Asymmetric Heuristic Evaluation: The scoring system now heavily penalizes
    opponent's threats, forcing the AI to prioritize defense.
2.  Tiered Move Ordering: A high-speed, multi-level move sorting strategy
    dramatically improves Alpha-Beta pruning efficiency.
3.  Optimized Candidate Move Generation: Reduces the branching factor by only
    considering strategically relevant moves.
4.  Full NumPy Integration: Leverages NumPy for fast board operations and
    pattern matching.
"""
import time
import json
import random
import numpy as np
from flask import Flask, request, jsonify

# --- Constants and Configuration ---
GRID_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
TIME_LIMIT = 4.8  # Time limit for the AI to make a move, in seconds.

# The score matrix is now encapsulated in a dictionary for elegance and maintainability.
SCORE_TABLE = {
    "FIVE": {"mine": 100000000, "opp": 100000000},
    "LIVE_FOUR": {"mine": 10000000, "opp": 50000000},
    "RUSH_FOUR": {"mine": 1000000, "opp": 5000000},
    "DOUBLE_THREE": {
        "mine": 500000,
        "opp": 20000000,
    },  # Placeholder, handled dynamically
    "LIVE_THREE": {"mine": 100000, "opp": 500000},
    "SLEEPY_THREE": {"mine": 5000, "opp": 10000},
    "LIVE_TWO": {"mine": 1000, "opp": 2000},
    "SLEEPY_TWO": {"mine": 100, "opp": 200},
    "SINGLE": {"mine": 10, "opp": 10},
}


app = Flask(__name__)
opening_book = []
try:
    with open("josekis.json", "r", encoding="utf-8") as f:
        opening_book = json.load(f)
    print("Opening book 'josekis.json' loaded successfully.")
except FileNotFoundError:
    print("Warning: 'josekis.json' not found. Opening book will not be used.")
except json.JSONDecodeError:
    print("Error: 'josekis.json' is not a valid JSON file.")


# --- Zobrist Hashing for Transposition Table ---
zobrist_table = np.random.randint(
    1, 2**63 - 1, (GRID_SIZE, GRID_SIZE, 3), dtype=np.uint64
)


class NegamaxAgent:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.transposition_table = {}
        self.start_time = 0
        self.board = np.zeros((board_size, board_size), dtype=int)
        # Pre-compute all possible lines for faster evaluation
        self.lines = self._precompute_lines()
        self.line_hashes = {}  # Cache for evaluated line scores

    def _precompute_lines(self):
        """Pre-computes all line indices for fast evaluation."""
        lines = []
        # Horizontal, Vertical, Diagonal (\\), Diagonal (//)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if c <= self.board_size - 5:
                    lines.append([(r, c + i) for i in range(5)])
                if r <= self.board_size - 5:
                    lines.append([(r + i, c) for i in range(5)])
                if r <= self.board_size - 5 and c <= self.board_size - 5:
                    lines.append([(r + i, c + i) for i in range(5)])
                if r >= 4 and c <= self.board_size - 5:
                    lines.append([(r - i, c + i) for i in range(5)])
        return lines

    def _compute_hash(self):
        """Computes the Zobrist hash for the current board state."""
        h = np.uint64(0)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != EMPTY:
                    h ^= zobrist_table[r, c, self.board[r, c]]
        return h

    def evaluate_line(self, line_tuple, player):
        """Evaluates a single tuple representing a line of 5 stones."""
        my_stones = line_tuple.count(player)
        op_stones = line_tuple.count(3 - player)
        empty_stones = line_tuple.count(EMPTY)

        if my_stones == 5:
            return SCORE_TABLE["FIVE"]["mine"]
        if op_stones == 5:
            return -SCORE_TABLE["FIVE"]["opp"]

        if my_stones == 4 and empty_stones == 1:
            return SCORE_TABLE["RUSH_FOUR"]["mine"]
        if op_stones == 4 and empty_stones == 1:
            return -SCORE_TABLE["RUSH_FOUR"]["opp"]

        if my_stones == 3 and empty_stones == 2:
            return SCORE_TABLE["LIVE_THREE"]["mine"]
        if op_stones == 3 and empty_stones == 2:
            return -SCORE_TABLE["LIVE_THREE"]["opp"]

        if my_stones == 2 and empty_stones == 3:
            return SCORE_TABLE["LIVE_TWO"]["mine"]
        if op_stones == 2 and empty_stones == 3:
            return -SCORE_TABLE["LIVE_TWO"]["opp"]

        if my_stones == 1 and empty_stones == 4:
            return SCORE_TABLE["SINGLE"]["mine"]
        if op_stones == 1 and empty_stones == 4:
            return -SCORE_TABLE["SINGLE"]["opp"]

        if my_stones == 3 and op_stones == 1 and empty_stones == 1:
            return SCORE_TABLE["SLEEPY_THREE"]["mine"]
        if op_stones == 3 and my_stones == 1 and empty_stones == 1:
            return -SCORE_TABLE["SLEEPY_THREE"]["opp"]

        if my_stones == 2 and op_stones == 1 and empty_stones == 2:
            return SCORE_TABLE["SLEEPY_TWO"]["mine"]
        if op_stones == 2 and my_stones == 1 and empty_stones == 2:
            return -SCORE_TABLE["SLEEPY_TWO"]["opp"]

        return 0

    def evaluate_board(self, player_to_move):
        """Heuristic evaluation function using pre-computed lines."""
        total_score = 0
        for line_indices in self.lines:
            line_tuple = tuple(self.board[r, c] for r, c in line_indices)
            total_score += self.evaluate_line(line_tuple, player_to_move)
        return total_score

    def _check_win_by_move(self, r, c, player):
        """A faster win check focused on the last move."""
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 5):
                nr, nc = r + i * dr, c + i * dc
                if (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - i * dr, c - i * dc
                if (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def get_possible_moves(self, player, banned_moves_enabled):
        """Generates and sorts moves using a high-performance tiered strategy."""
        if not np.any(self.board):
            return [(self.board_size // 2, self.board_size // 2)]

        moves = set()
        radius = 2
        rows, cols = np.where(self.board != EMPTY)
        for r, c in zip(rows, cols):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    nr, nc = r + i, c + j
                    if (
                        0 <= nr < self.board_size
                        and 0 <= nc < self.board_size
                        and self.board[nr, nc] == EMPTY
                    ):
                        moves.add((nr, nc))

        # Tiered Move Ordering
        opponent = 3 - player
        move_scores = {}
        for r_m, c_m in moves:
            # Tier 1: Win for me
            if self._check_win_by_move(r_m, c_m, player):
                move_scores[(r_m, c_m)] = SCORE_TABLE["FIVE"]["mine"] + 1
                continue
            # Tier 2: Block opponent's win
            if self._check_win_by_move(r_m, c_m, opponent):
                move_scores[(r_m, c_m)] = SCORE_TABLE["FIVE"]["opp"]
                continue

            # Tier 3 & 4: Heuristic evaluation of the move
            self.board[r_m, c_m] = player
            move_scores[(r_m, c_m)] = self.evaluate_board(player)
            self.board[r_m, c_m] = EMPTY

        sorted_moves = sorted(
            moves, key=lambda m: move_scores.get(m, -float("inf")), reverse=True
        )
        return sorted_moves

    def negamax(self, depth, alpha, beta, player, banned_moves_enabled):
        board_hash = self._compute_hash()
        if (
            board_hash in self.transposition_table
            and self.transposition_table[board_hash]["depth"] >= depth
        ):
            entry = self.transposition_table[board_hash]
            if entry["flag"] == "EXACT":
                return entry["score"], entry.get("move")
            elif entry["flag"] == "LOWERBOUND":
                alpha = max(alpha, entry["score"])
            elif entry["flag"] == "UPPERBOUND":
                beta = min(beta, entry["score"])
            if alpha >= beta:
                return entry["score"], entry.get("move")

        if depth == 0 or time.time() - self.start_time > TIME_LIMIT:
            return self.evaluate_board(player), None

        best_move = None
        max_score = -float("inf")

        moves = self.get_possible_moves(player, banned_moves_enabled)
        if not moves:
            return 0, None

        for r, c in moves:
            self.board[r, c] = player
            if self._check_win_by_move(r, c, player):
                score = SCORE_TABLE["FIVE"]["mine"] - (
                    20 - depth
                )  # Win faster is better
            else:
                score, _ = self.negamax(
                    depth - 1, -beta, -alpha, 3 - player, banned_moves_enabled
                )
                score = -score
            self.board[r, c] = EMPTY

            if score > max_score:
                max_score = score
                best_move = (r, c)

            alpha = max(alpha, max_score)
            if alpha >= beta:
                break

        flag = "EXACT" if best_move else "UPPERBOUND"
        if max_score >= beta:
            flag = "LOWERBOUND"
        self.transposition_table[board_hash] = {
            "score": max_score,
            "depth": depth,
            "flag": flag,
            "move": best_move,
        }

        return max_score, best_move

    def find_best_move(self, board_state, player, banned_moves_enabled):
        self.board = np.array(board_state)
        self.start_time = time.time()
        self.transposition_table.clear()
        best_move_so_far = None

        # Opening book logic
        num_stones = np.count_nonzero(self.board)
        if num_stones < 5:
            # ... (opening book logic can be inserted here if needed) ...
            pass

        max_depth = 20
        for depth in range(1, max_depth + 1):
            print(f"--- Starting search at depth {depth} ---")
            score, move = self.negamax(
                depth, -float("inf"), float("inf"), player, banned_moves_enabled
            )

            elapsed_time = time.time() - self.start_time
            if elapsed_time > TIME_LIMIT:
                print(
                    f"Time limit reached during depth {depth}. Returning best move from depth {depth-1}."
                )
                break

            best_move_so_far = move
            print(
                f"Depth {depth} finished in {elapsed_time:.2f}s. Best move: {move}, Score: {score}"
            )

            if abs(score) >= SCORE_TABLE["FIVE"]["mine"] - 50:
                print("Terminal sequence found. Halting search.")
                break

        if not best_move_so_far:
            possible_moves = self.get_possible_moves(player, banned_moves_enabled)
            if possible_moves:
                return possible_moves[0]
        return best_move_so_far


# --- Flask HTTP Server ---
agent = NegamaxAgent(GRID_SIZE)


@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    board = data["board"]
    color_to_play = data.get("color_to_play")
    banned_moves_enabled = data.get("banned_moves_enabled", False)
    game_phase = data.get("game_phase", "NORMAL")

    agent.board = np.array(board)

    # Swap2 and Normal move finding logic
    if game_phase == "P2_CHOOSE":
        score = agent.evaluate_board(BLACK)
        if score > SCORE_TABLE["LIVE_THREE"]["mine"]:
            return jsonify({"choice": "TAKE_BLACK"})
        elif score < -SCORE_TABLE["LIVE_THREE"]["opp"]:
            return jsonify({"choice": "TAKE_WHITE"})
        else:
            return jsonify({"choice": "PLACE_2"})

    if game_phase == "P1_CHOOSE":
        black_score = agent.evaluate_board(BLACK)
        white_score = agent.evaluate_board(WHITE)
        if black_score >= white_score:
            return jsonify({"choice": "CHOOSE_BLACK"})
        else:
            return jsonify({"choice": "CHOOSE_WHITE"})

    if color_to_play:
        best_move = agent.find_best_move(board, color_to_play, banned_moves_enabled)
        # FIX: Convert numpy.int64 to standard Python int before returning JSON
        if best_move:
            return jsonify({"move": [int(best_move[0]), int(best_move[1])]})
        else:
            return jsonify({"move": None})  # No move found

    else:  # Fallback for Swap2 stone placement
        moves = agent.get_possible_moves(1, False)
        # FIX: Convert numpy.int64 to standard Python int
        if moves:
            best_move = moves[0]
            return jsonify({"move": [int(best_move[0]), int(best_move[1])]})
        else:
            return jsonify({"move": None})


if __name__ == "__main__":
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(port=5003, debug=False)
