# -*- coding: utf-8 -*-
"""
Core Improvements:
1.  Asymmetric Heuristic Evaluation: The scoring system now heavily penalizes
    opponent's threats, forcing the AI to prioritize defense.
2.  Tiered Move Ordering: A high-speed, multi-level move sorting strategy
    dramatically improves Alpha-Beta pruning efficiency.
3.  Optimized Candidate Move Generation: Reduces the branching factor by only
    considering strategically relevant moves.
4.  Full NumPy Integration: Leverages NumPy for fast board operations and
    pattern matching.
5.  Banned Move Detection: The agent can now identify and avoid illegal moves
    for Black (Three-Three, Four-Four, Overline) when Renju rules are enabled.
"""
import time
import json
import random
import numpy as np
from flask import Flask, request, jsonify

# --- Configuration ---
GRID_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
TIME_LIMIT = 9  # Time limit for the AI to make a move, in seconds.


"""
The values for opponent's threats ('opp') are now significantly higher than 'mine'
to force the AI to block critical threats instead of making risky offensive moves.
"""
SCORE_TABLE = {
    "FIVE": {"mine": 100_000_000, "opp": 200_000_000},
    "LIVE_FOUR": {
        "mine": 10_000_000,
        "opp": 20_000_000,
    },  # _OOOO_
    "RUSH_FOUR": {
        "mine": 1_000_000,
        "opp": 2_000_000,
    },  # XOOOO_
    "DOUBLE_THREE": {
        "mine": 10_000_000,
        "opp": 20_000_000,
    },
    "LIVE_THREE": {
        "mine": 5_000_000,
        "opp": 10_000_000,
    },  # _OOO_
    "SLEEPY_THREE": {"mine": 5_000, "opp": 10_000},  # XOOO_
    "LIVE_TWO": {"mine": 1_000, "opp": 2_000},
    "SLEEPY_TWO": {"mine": 100, "opp": 200},
    "SINGLE": {"mine": 10, "opp": 10},
}


app = Flask(__name__)
# opening_book = []
# try:
#     with open("josekis.json", "r", encoding="utf-8") as f:
#         opening_book = json.load(f)
#     print("Opening book 'josekis.json' loaded successfully.")
# except FileNotFoundError:
#     print("Warning: 'josekis.json' not found. Opening book will not be used.")
# except json.JSONDecodeError:
#     print("Error: 'josekis.json' is not a valid JSON file.")


# --- Zobrist Hashing for Transposition Table ---
zobrist_table = np.random.randint(
    1, 2**63 - 1, (GRID_SIZE, GRID_SIZE, 3), dtype=np.uint64
)  # Three-dimensional array: 0 - EMPTY, 1 - BLACK, 2 - WHITE


class NegamaxAgent:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.transposition_table = {}
        self.start_time = 0
        self.board = np.zeros((board_size, board_size), dtype=int)
        # Pre-compute all possible lines for faster evaluation
        self.possible_win_lines = self._precompute_lines()
        self.line_hashes = {}  # Cache for evaluated line scores

    def _precompute_lines(self):
        """
        Pre-computes all line indices for fast evaluation.
        Omit the need to repeated loops and boundary judgments.
        """
        lines = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                # -
                if c <= self.board_size - 5:
                    lines.append([(r, c + i) for i in range(5)])
                # Vertical
                if r <= self.board_size - 5:
                    lines.append([(r + i, c) for i in range(5)])
                # \\
                if r <= self.board_size - 5 and c <= self.board_size - 5:
                    lines.append([(r + i, c + i) for i in range(5)])
                # //
                if r >= 4 and c <= self.board_size - 5:
                    lines.append([(r - i, c + i) for i in range(5)])
        return lines

    def _compute_hash(self):
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

        # check score patterns
        if my_stones > 0 and op_stones > 0:
            if my_stones == 3 and op_stones == 1 and empty_stones == 1:
                return SCORE_TABLE["SLEEPY_THREE"]["mine"]
            if op_stones == 3 and my_stones == 1 and empty_stones == 1:
                return -SCORE_TABLE["SLEEPY_THREE"]["opp"]
            if my_stones == 2 and op_stones == 1 and empty_stones == 2:
                return SCORE_TABLE["SLEEPY_TWO"]["mine"]
            if op_stones == 2 and my_stones == 1 and empty_stones == 2:
                return -SCORE_TABLE["SLEEPY_TWO"]["opp"]
            return 0  # Mixed stones with no clear pattern are neutral

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

        return 0

    def evaluate_board(self, player_to_move):
        """Heuristic evaluation function using pre-computed lines."""
        total_score = 0
        for line_indices in self.possible_win_lines:
            line_tuple = tuple(self.board[r, c] for r, c in line_indices)
            total_score += self.evaluate_line(line_tuple, player_to_move)
        return total_score

    def _check_win_by_move(self, r, c, player):
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

    def _is_banned_move(self, r, c, player):
        """
        Checks if a move is a banned move for Black (three-three, four-four, or overline).
        """
        if self.board[r, c] != EMPTY:
            return False, ""  # Not a valid move

        # Temporarily place the stone to analyze the board state
        self.board[r, c] = player
        opponent = 3 - player

        # 1. Check for Overline (more than 5 stones)
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            # Count forward
            for i in range(1, 6):
                nr, nc = r + i * dr, c + i * dc
                if (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    count += 1
                else:
                    break
            # Count backward
            for i in range(1, 6):
                nr, nc = r - i * dr, c - i * dc
                if (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    count += 1
                else:
                    break
            if count > 5:
                self.board[r, c] = EMPTY  # Revert the move
                return True, "Overline"

        # 2. Check for Double-Three and Double-Four
        three_count = 0
        four_count = 0
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            # Check for live threes
            # Pattern: _OOO_ -> 01110
            # Check O_OO, OO_O patterns created by the new stone
            for i in range(-4, 1):
                line = []
                for j in range(5):
                    nr, nc = r + (i + j) * dr, c + (i + j) * dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        line.append(self.board[nr, nc])
                    else:
                        line.append(opponent)  # Treat board edge as opponent stone

                line_tuple = tuple(line)
                # Live three check: e.g., (0,1,1,1,0)
                if line_tuple == (EMPTY, player, player, player, EMPTY):
                    three_count += 1
                    break  # A live three is found in this direction, no need to check further

            # Check for fours (live or rush)
            # Pattern: OOOO
            for i in range(-4, 1):
                line = []
                for j in range(5):
                    nr, nc = r + (i + j) * dr, c + (i + j) * dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        line.append(self.board[nr, nc])
                    else:
                        line.append(opponent)

                if line.count(player) == 4 and line.count(EMPTY) == 1:
                    four_count += 1
                    break  # A four is found

        self.board[r, c] = EMPTY  # Revert the move before returning

        if three_count >= 2:
            return True, "Three-Three"
        if four_count >= 2:
            return True, "Four-Four"

        return False, ""

    def get_possible_moves(self, player, banned_moves_enabled):
        if not np.any(self.board):
            return [(self.board_size // 2, self.board_size // 2)]

        moves = set()
        radius = 2  # Search radius around existing stones, sweet spot for efficiency
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
                        if banned_moves_enabled and player == BLACK:
                            is_banned, reason = self._is_banned_move(nr, nc, player)
                            if is_banned:
                                continue
                        moves.add((nr, nc))

        if not moves:
            empty_cells = np.argwhere(self.board == EMPTY)
            return [tuple(cell) for cell in empty_cells]

        # Tiered Move Ordering: Prioritize moves that win or block immediate threats
        opponent = 3 - player
        my_win_moves = []
        opponent_win_moves = []
        for r_m, c_m in moves:
            # my win
            self.board[r_m, c_m] = player
            if self._check_win_by_move(r_m, c_m, player):
                my_win_moves.append((r_m, c_m))
            self.board[r_m, c_m] = EMPTY

            # opponent win
            self.board[r_m, c_m] = opponent
            if self._check_win_by_move(r_m, c_m, opponent):
                opponent_win_moves.append((r_m, c_m))
            self.board[r_m, c_m] = EMPTY

        if my_win_moves:
            return my_win_moves

        # heuristic evaluation for non-my-winning moves
        move_scores = {}
        for r_m, c_m in moves:
            if (r_m, c_m) in opponent_win_moves:
                continue  # has to defense
            self.board[r_m, c_m] = player
            move_scores[(r_m, c_m)] = self.evaluate_board(player)
            self.board[r_m, c_m] = EMPTY

        # Combine the tiers: block moves first, then other moves sorted by score.
        sorted_other_moves = sorted(
            move_scores.keys(),
            key=lambda m: move_scores.get(m, -float("inf")),
            reverse=True,
        )

        return opponent_win_moves + sorted_other_moves

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

        # If the first move in the sorted list leads to a win, the search will prioritize it.
        # If it blocks a threat, it will also be prioritized.
        moves = self.get_possible_moves(player, banned_moves_enabled)
        if not moves:
            return 0, None
        for r, c in moves:
            self.board[r, c] = player
            # After making a move, check for win condition to assign max score
            if self._check_win_by_move(r, c, player):
                score = SCORE_TABLE["FIVE"]["mine"] - (20 - depth)
            else:
                score, _ = self.negamax(
                    depth - 1, -beta, -alpha, 3 - player, banned_moves_enabled
                )
                score = -score  # Negate: min(a, b) = -max(-a, -b)
            self.board[r, c] = EMPTY

            if score > max_score:
                max_score = score
                best_move = (r, c)

            """Alpha-beta pruning"""
            alpha = max(alpha, max_score)
            if alpha >= beta:
                break

        flag = "EXACT"
        if max_score <= alpha:
            flag = "UPPERBOUND"
        elif max_score >= beta:
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

        num_stones = np.count_nonzero(self.board)
        if num_stones < 5:
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
            print(
                "No best move found from search, falling back to first possible move."
            )
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
        if best_move:
            return jsonify({"move": [int(best_move[0]), int(best_move[1])]})
        else:
            return jsonify({"move": None})

    else:
        moves = agent.get_possible_moves(1, False)
        if moves:
            best_move = moves[0]
            return jsonify({"move": [int(best_move[0]), int(best_move[1])]})
        else:
            return jsonify({"move": None})


if __name__ == "__main__":
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    app.run(port=5003, debug=False)
