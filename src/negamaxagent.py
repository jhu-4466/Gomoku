# -*- coding: utf-8 -*-

import time
import json
import re
import logging
from collections import defaultdict
import numpy as np
from flask import Flask, request, jsonify
import os
from datetime import datetime
import random


# --- Logging Setup ---
if not os.path.exists("./logs"):
    os.makedirs("./logs")

log_filename = f"./logs/negamax_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],  # , logging.StreamHandler()
)
logger = logging.getLogger(__name__)


# --- Configuration ---
GRID_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
TIME_LIMIT = 29.5  # Time limit for the AI to make a move, in seconds.
MAX_DEPTH = 50  # Max search depth for IDDFS
MIN_DEPTH = 3  # The minimum depth the AI must complete, regardless of time.
TOP_K_BY_DEPTH = [32, 28, 24, 20]


"""
The values for opponent's threats ('opp') are now significantly higher than 'mine'
to force the AI to block critical threats instead of making risky offensive moves.
"""
SCORE_TABLE = {
    "FIVE": {"mine": 100_000_000, "opp": 200_000_000},
    "LIVE_FOUR": {"mine": 100_000, "opp": 200_000},
    "DOUBLE_THREE": {"mine": 50_000, "opp": 150_000},
    "LIVE_THREE": {"mine": 20_000, "opp": 35_000},
    "RUSH_FOUR": {"mine": 1_500, "opp": 5_000},
    "SLEEPY_THREE": {"mine": 800, "opp": 1500},
    "LIVE_TWO": {"mine": 500, "opp": 1000},
    "SLEEPY_TWO": {"mine": 100, "opp": 200},
    "SYNERGY_BONUS": {"mine": 3000, "opp": 3000},
    "POSITIONAL_BONUS_FACTOR": 5,
}
PATTERNS_PLAYER = {
    "FIVE": re.compile(r"11111"),
    "LIVE_FOUR": re.compile(r"011110"),
    "RUSH_FOUR": re.compile(r"211110|011112|10111|11011|11101"),
    "LIVE_THREE": re.compile(r"01110|010110"),
    "SLEEPY_THREE": re.compile(r"21110|01112|210110|011012|21101|10112"),
    "LIVE_TWO": re.compile(r"001100|01010|010010"),
    "SLEEPY_TWO": re.compile(r"21100|00112|21010|01012|21001|10012"),
}

# Zobrist hashing table for fast board state hashing
zobrist_table = np.random.randint(
    1, 2**63 - 1, (GRID_SIZE, GRID_SIZE, 3), dtype=np.uint64
)
zobrist_player_turn = np.random.randint(1, 2**63 - 1, dtype=np.uint64)


# Game phases for swap2
class GamePhase:
    NORMAL = 0
    SWAP2_P1_PLACE_3 = 1
    SWAP2_P2_CHOOSE_ACTION = 2
    SWAP2_P2_PLACE_2 = 3
    SWAP2_P1_CHOOSE_COLOR = 4


# --- Flask App Setup ---
app = Flask(__name__)


# Custom exception for reliable timeout handling.
class TimeoutException(Exception):
    pass


class IncrementalEvaluator:
    """
    Manages board evaluation incrementally to avoid expensive full-board scans.
    """

    def __init__(self, board_size=15):
        self.board_size = board_size
        self.lines = (
            {}
        )  # Stores all line coordinates, e.g., { "h_0": [(0,0), ...], ... }
        self.line_to_squares = {}  # Maps line_id to the set of squares in it
        self.square_to_lines = defaultdict(list)  # Maps (r,c) to list of line_ids

        self.line_values = defaultdict(int)  # Stores the score of each line
        self.total_score = 0

        self._init_lines()

    def _init_lines(self):
        # Pre-calculates all horizontal, vertical, and diagonal lines
        # and maps squares to the lines they belong to.
        # Horizontal
        for r in range(self.board_size):
            line_id = f"h_{r}"
            self.lines[line_id] = [(r, c) for c in range(self.board_size)]
        # Vertical
        for c in range(self.board_size):
            line_id = f"v_{c}"
            self.lines[line_id] = [(r, c) for r in range(self.board_size)]
        # Diagonal (top-left to bottom-right)
        for i in range(self.board_size * 2 - 1):
            line_id = f"d1_{i}"
            self.lines[line_id] = []
            for r in range(self.board_size):
                c = r - (self.board_size - 1 - i)
                if 0 <= c < self.board_size:
                    self.lines[line_id].append((r, c))
        # Diagonal (top-right to bottom-left)
        for i in range(self.board_size * 2 - 1):
            line_id = f"d2_{i}"
            self.lines[line_id] = []
            for r in range(self.board_size):
                c = (self.board_size - 1 - r) - (self.board_size - 1 - i)
                if 0 <= c < self.board_size:
                    self.lines[line_id].append((r, c))

        # Create the reverse mappings
        for line_id, squares in self.lines.items():
            self.line_to_squares[line_id] = set(squares)
            for r, c in squares:
                self.square_to_lines[(r, c)].append(line_id)

    def _score_line(self, line_str):
        # Scores a single line string based on patterns
        score = 0
        # Player 1 (Black)
        p1_board = line_str.replace("2", "0")
        for pattern_name, regex in PATTERNS_PLAYER.items():
            count = len(regex.findall(p1_board))
            if count > 0:
                score += SCORE_TABLE[pattern_name]["mine"] * count
        # Player 2 (White)
        p2_board = line_str.replace("1", "X").replace("2", "1").replace("X", "2")
        for pattern_name, regex in PATTERNS_PLAYER.items():
            count = len(regex.findall(p2_board))
            if count > 0:
                score -= SCORE_TABLE[pattern_name]["opp"] * count
        return score

    def full_recalc(self, board):
        # Calculates the score of the entire board from scratch.
        # Called once at the beginning of each move search.
        self.total_score = 0
        self.line_values.clear()

        for line_id, squares in self.lines.items():
            line_str = "".join(str(board[r, c]) for r, c in squares)
            line_score = self._score_line(line_str)
            self.line_values[line_id] = line_score
            self.total_score += line_score

    def update_score(self, board, r, c, player):
        # The core incremental update function.
        # It's called BEFORE the move is made on the main board.
        affected_lines = self.square_to_lines[(r, c)]

        for line_id in affected_lines:
            # 1. Subtract the old score of the line
            self.total_score -= self.line_values[line_id]

            # 2. Recalculate the new score for the line
            squares = self.lines[line_id]
            # Temporarily make the move to build the string
            board[r, c] = player
            new_line_str = "".join(str(board[sq_r, sq_c]) for sq_r, sq_c in squares)
            board[r, c] = EMPTY  # Revert immediately

            new_line_score = self._score_line(new_line_str)

            # 3. Add the new score and update the line value
            self.line_values[line_id] = new_line_score
            self.total_score += new_line_score

    def get_current_score(self, color_to_play):
        # Returns the score from the perspective of the current player.
        # In our calculation, positive is good for Black, negative for White.
        return self.total_score if color_to_play == BLACK else -self.total_score


class NegamaxAgent:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.transposition_table = {}
        self.start_time = 0
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.evaluator = IncrementalEvaluator(board_size)
        self.current_turn = 0

        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
        self.history_heuristic = defaultdict(int)
        self.search_generation = 0
        self.last_depth_start_time = 0

        self.current_search_depth = 0
        self.nodes_evaluated_at_root = 0
        self.total_nodes_at_root = 0

        self.swap2_opening_sequence = []
        self.joseki_book = self._load_joseki()

    def _load_joseki(self, joseki_file="josekis.json"):
        """Loads the joseki book from a JSON file."""
        try:
            if os.path.exists(joseki_file):
                with open(joseki_file, "r", encoding="utf-8") as f:
                    return json.load(f)
                logger.info(
                    f"Successfully loaded {len(self.joseki_book)} joseki patterns."
                )
            else:
                logger.warning(
                    f"Joseki file '{joseki_file}' not found. Continuing without opening book."
                )
                return None
        except Exception as e:
            logger.error(f"Error loading joseki file: {e}")
            return None

    def find_opening_move(self, color_to_play):
        if not self.joseki_book:
            return None

        # 1st stone
        if self.current_turn == 1 and color_to_play == BLACK:
            favorable_josekis = [j for j in self.joseki_book if j.get("trend") == 1]
            if not favorable_josekis:
                favorable_josekis = self.joseki_book

            if favorable_josekis:
                logger.info(
                    "Turn 1: Choosing an opening from trend=1 (favorable for Black) josekis."
                )
                chosen_joseki = random.choice(favorable_josekis)
                return chosen_joseki["joseki"][0][:2]

        # 2nd stone
        if self.current_turn == 2 and color_to_play == WHITE:
            black_stone_pos = np.argwhere(self.board == BLACK)
            if len(black_stone_pos) != 1:
                return None
            p1_move = black_stone_pos[0].tolist()

            matching_josekis = [
                j for j in self.joseki_book if j["joseki"][0][:2] == p1_move
            ]
            if not matching_josekis:
                return None

            preferred_josekis = [j for j in matching_josekis if j.get("trend") == 2]

            if preferred_josekis:
                logger.info(
                    f"Turn 2: Found trend=2 (favorable for White) josekis. Responding."
                )
                chosen_joseki = random.choice(preferred_josekis)
                return chosen_joseki["joseki"][1][:2]
            else:
                logger.info(
                    f"Turn 2: No trend=2 josekis found. Falling back to any matched joseki."
                )
                chosen_joseki = random.choice(matching_josekis)
                return chosen_joseki["joseki"][1][:2]

        # 3rd stone
        if self.current_turn == 3 and color_to_play == BLACK:
            black_stones = np.argwhere(self.board == BLACK)
            white_stones = np.argwhere(self.board == WHITE)
            if len(black_stones) != 1 or len(white_stones) != 1:
                return None

            p1_move = black_stones[0].tolist()
            p2_move = white_stones[0].tolist()

            matching_josekis = [
                j
                for j in self.joseki_book
                if j["joseki"][0][:2] == p1_move and j["joseki"][1][:2] == p2_move
            ]
            if not matching_josekis:
                return None

            # 优先选择对黑棋有利的(trend: 1)
            preferred_josekis = [j for j in matching_josekis if j.get("trend") == 1]

            if preferred_josekis:
                logger.info(
                    f"Turn 3: Found trend=1 (favorable for Black) josekis. Playing third move."
                )
                chosen_joseki = random.choice(preferred_josekis)
                return chosen_joseki["joseki"][2][:2]
            else:
                logger.info(
                    f"Turn 3: No trend=1 josekis found. Falling back to any matched joseki."
                )
                chosen_joseki = random.choice(matching_josekis)
                return chosen_joseki["joseki"][2][:2]

        return None

    def _compute_hash(self, player):
        h = np.uint64(0)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != EMPTY:
                    h ^= zobrist_table[r, c, self.board[r, c]]
        if player == WHITE:
            h ^= zobrist_player_turn
        return h

    def _check_win_by_move(self, r, c, player):
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 5):
                nr, nc = r + i * dr, c + i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    break
                count += 1
            for i in range(1, 5):
                nr, nc = r - i * dr, c - i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    break
                count += 1
            if count >= 5:
                return True
        return False

    def _is_banned_move(self, r, c, player):
        if player != BLACK:
            return False, ""
        if self.board[r, c] != EMPTY:
            return False, ""

        self.board[r, c] = player
        if self._check_win_by_move(r, c, player):
            self.board[r, c] = EMPTY  # Revert board
            return False, ""

        # --- Check for overline ---
        is_overline = False
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 6):
                nr, nc = r + i * dr, c + i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    break
                count += 1
            for i in range(1, 6):
                nr, nc = r - i * dr, c - i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    break
                count += 1
            if count > 5:
                is_overline = True
                break
        if is_overline:
            self.board[r, c] = EMPTY
            return True, "Overline"

        # --- Check for live three and four ---
        threes, fours = 0, 0
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            # Create a line of 9 points centered on the move
            line = [
                (
                    self.board[r + i * dr][c + i * dc]
                    if 0 <= r + i * dr < self.board_size
                    and 0 <= c + i * dc < self.board_size
                    else -1  # Use -1 for off-board
                )
                for i in range(-4, 5)
            ]

            line_str = "".join(map(str, line)).replace(str(WHITE), str(EMPTY))
            if "01110" in line_str:
                threes += 1
            if "1111" in line_str:
                fours += 1

        self.board[r, c] = EMPTY
        if threes >= 2:
            return True, "Three-Three"
        if fours >= 2:
            return True, "Four-Four"
        return False, ""

    def _count_patterns_at(self, r, c, dr, dc, player):
        threes, fours = 0, 0
        for i in range(-2, 1):
            p = [(r + (i + j) * dr, c + (i + j) * dc) for j in range(5)]
            if all(
                0 <= pr < self.board_size and 0 <= pc < self.board_size
                for pr, pc in [p[0], p[4]]
            ):
                line = tuple(self.board[pr, pc] for pr, pc in p)
                if line == (EMPTY, player, player, player, EMPTY):
                    threes += 1
                    break
        for i in range(-3, 1):
            p = [(r + (i + j) * dr, c + (i + j) * dc) for j in range(4)]
            if all(
                0 <= pr < self.board_size and 0 <= pc < self.board_size for pr, pc in p
            ):
                line = [self.board[pr, pc] for pr, pc in p]
                if line.count(player) == 4:
                    fours += 1
                    break
        return threes, fours

    def evaluate_board(self, color_to_play):
        self.evaluator.full_recalc(self.board)
        return self.evaluator.get_current_score(color_to_play)

    def _find_patterns_fast(self, player):
        patterns = defaultdict(int)

        # transform the board to a player-centric view:
        # 1- player, 2- opponent
        player_board = np.copy(self.board)
        if player == WHITE:
            player_board[self.board == WHITE] = 1
            player_board[self.board == BLACK] = 2
        else:
            player_board[self.board == BLACK] = 1
            player_board[self.board == WHITE] = 2

        board_str_lines = []
        for i in range(self.board_size):
            board_str_lines.append("".join(map(str, player_board[i, :])))
            board_str_lines.append("".join(map(str, player_board[:, i])))
        for i in range(-self.board_size + 1, self.board_size):
            board_str_lines.append("".join(map(str, player_board.diagonal(i))))
            board_str_lines.append(
                "".join(map(str, np.fliplr(player_board).diagonal(i)))
            )

        for line in board_str_lines:
            for pattern_name, regex in PATTERNS_PLAYER.items():
                patterns[pattern_name] += len(regex.findall(line))

        if patterns["FIVE"] > 0:
            patterns["RUSH_FOUR"] -= patterns["FIVE"] * 4

        return patterns

    def _rate_move_statically(self, r, c, player):
        """
        Statically evaluate the threat of a single move for sorting purposes.
        """
        opponent = 3 - player
        total_score_change = 0

        # --- 1. Positional Bonus ---
        center = self.board_size // 2
        dist = max(abs(r - center), abs(c - center))
        total_score_change += (center - dist) * SCORE_TABLE["POSITIONAL_BONUS_FACTOR"]

        # --- 2. Calculate Offensive and Defensive score changes ---
        affected_lines = self.evaluator.square_to_lines.get((r, c), [])

        for line_id in affected_lines:
            squares = self.evaluator.lines[line_id]

            # A) Original line score before the move
            line_str_before = "".join(
                str(self.board[sq_r, sq_c]) for sq_r, sq_c in squares
            )
            score_before = self.evaluator._score_line(line_str_before)

            # B) If I move here
            line_str_mine = "".join(
                str(player) if (sq_r, sq_c) == (r, c) else str(self.board[sq_r, sq_c])
                for sq_r, sq_c in squares
            )
            score_after_mine = self.evaluator._score_line(line_str_mine)
            # C) If opponent moves here
            line_str_opp = "".join(
                str(opponent) if (sq_r, sq_c) == (r, c) else str(self.board[sq_r, sq_c])
                for sq_r, sq_c in squares
            )
            score_after_opp = self.evaluator._score_line(line_str_opp)

            my_gain = score_after_mine - score_before
            opp_gain = score_after_opp - score_before

            perspective = 1 if player == BLACK else -1
            total_score_change += my_gain * perspective
            total_score_change += opp_gain * -perspective

        total_score_change += self.history_heuristic.get((r, c), 0)

        return total_score_change

    def get_possible_moves(self, player, banned_moves_enabled, depth, hash_move):
        """
        Uses a sophisticated, multi-layered sorting approach,
        including static threat analysis, to achieve
        Threat Space Search.
        """
        board_hash = self._compute_hash(player)
        tt_entry = self.transposition_table.get(board_hash)
        if tt_entry and "sorted_moves" in tt_entry:
            return tt_entry["sorted_moves"]

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
                        if banned_moves_enabled and player == BLACK:
                            if self._is_banned_move(nr, nc, player)[0]:
                                continue
                        moves.add((nr, nc))

        if not moves:
            empty_cells = np.argwhere(self.board == EMPTY)
            return [tuple(cell) for cell in empty_cells]

        opponent = 3 - player

        # --- Check for immediate winning moves ---
        my_win_moves = [m for m in moves if self._check_win_by_move(m[0], m[1], player)]
        opponent_win_moves = [
            m for m in moves if self._check_win_by_move(m[0], m[1], opponent)
        ]

        # --- Urgent moves: Immediate threats ---
        my_urgent_attacks = []
        opponent_urgent_defenses = []
        for r, c in moves:
            self.board[r, c] = player
            my_patterns = self._find_patterns_fast(player)
            if (
                my_patterns.get("LIVE_FOUR", 0) > 0
                or my_patterns.get("LIVE_THREE", 0) >= 2
            ):
                my_urgent_attacks.append((r, c))
            self.board[r, c] = EMPTY

            self.board[r, c] = opponent
            opp_patterns = self._find_patterns_fast(opponent)
            if (
                opp_patterns.get("LIVE_FOUR", 0) > 0
                or opp_patterns.get("LIVE_THREE", 0) >= 2
            ):
                opponent_urgent_defenses.append((r, c))
            self.board[r, c] = EMPTY

        # --- Threat-based move ordering ---
        move_scores = {
            m: self._rate_move_statically(m[0], m[1], player)
            + self.history_heuristic.get(m, 0)
            for m in moves
        }

        sorted_moves = sorted(moves, key=lambda m: move_scores.get(m, 0), reverse=True)

        # --- Final prioritized list construction ---
        final_ordered_list = []
        # 1. Urgent: winning and threatening moves
        # (1) Winning moves
        for move in my_win_moves:
            if move not in final_ordered_list:
                final_ordered_list.append(move)
        for move in opponent_win_moves:
            if move not in final_ordered_list:
                final_ordered_list.append(move)
        # (2) Live Four and Double Three moves
        my_must_win_moves = []
        opponent_must_defend_moves = []
        for r, c in moves:
            self.board[r, c] = player
            patterns = self._find_patterns_fast(player)
            if patterns.get("LIVE_FOUR", 0) > 0 or patterns.get("LIVE_THREE", 0) >= 2:
                my_must_win_moves.append((r, c))
            self.board[r, c] = EMPTY
        for r, c in moves:
            self.board[r, c] = opponent
            patterns = self._find_patterns_fast(opponent)
            if patterns.get("LIVE_FOUR", 0) > 0 or patterns.get("LIVE_THREE", 0) >= 2:
                opponent_must_defend_moves.append((r, c))
            self.board[r, c] = EMPTY
        for move in opponent_must_defend_moves:
            if move not in final_ordered_list:
                final_ordered_list.append(move)
        for move in my_must_win_moves:
            if move not in final_ordered_list:
                final_ordered_list.append(move)
        # 2. Hash Move from Transposition Table
        if hash_move and hash_move in moves:
            final_ordered_list.append(hash_move)
        # 3. Killer Moves
        killers = self.killer_moves[depth]
        if killers[0] and killers[0] in moves and killers[0] not in final_ordered_list:
            final_ordered_list.append(killers[0])
        if killers[1] and killers[1] in moves and killers[1] not in final_ordered_list:
            final_ordered_list.append(killers[1])
        # 4. All other moves, sorted by threat+history
        for move in sorted_moves:
            if move not in final_ordered_list:
                final_ordered_list.append(move)
        # 5. Top-K Move Pruning to reduce the branching factor.
        if depth > 0:
            absolute_depth = self.current_search_depth - depth
            if absolute_depth < 0:
                absolute_depth = 0

            if move_scores:
                eval_spread = max(move_scores.values()) - min(move_scores.values())
                if eval_spread > 20000:  # Just a magic number
                    top_k = max(
                        8,
                        TOP_K_BY_DEPTH[min(absolute_depth, len(TOP_K_BY_DEPTH) - 1)]
                        // 2,
                    )
                else:
                    top_k = TOP_K_BY_DEPTH[min(absolute_depth, len(TOP_K_BY_DEPTH) - 1)]
            else:
                top_k = TOP_K_BY_DEPTH[min(absolute_depth, len(TOP_K_BY_DEPTH) - 1)]

            if len(final_ordered_list) > top_k:
                return final_ordered_list[:top_k]

        return final_ordered_list

    def _is_vcf_threat(self, move, player, banned_moves_enabled):
        r, c = move
        if self.board[r, c] != EMPTY:
            return False
        self.board[r, c] = player
        patterns = self._find_patterns_fast(player)
        self.board[r, c] = EMPTY
        if patterns.get("LIVE_FOUR", 0) > 0:
            return True
        if patterns.get("LIVE_THREE", 0) >= 2:
            if player == WHITE or (player == BLACK and not banned_moves_enabled):
                return True
        return False

    def negamax(self, depth, alpha, beta, player, banned_moves_enabled):
        if time.time() - self.start_time > TIME_LIMIT:
            raise TimeoutException()

        original_alpha = alpha
        board_hash = self._compute_hash(player)
        tt_entry = self.transposition_table.get(board_hash)

        if (
            tt_entry
            and tt_entry["depth"] >= depth
            and tt_entry["age"] == self.search_generation
        ):
            if tt_entry["flag"] == "EXACT":
                return tt_entry["score"], tt_entry.get("move")
            elif tt_entry["flag"] == "LOWERBOUND":
                alpha = max(alpha, tt_entry["score"])
            elif tt_entry["flag"] == "UPPERBOUND":
                beta = min(beta, tt_entry["score"])
            if alpha >= beta:
                return tt_entry["score"], tt_entry.get("move")

        if depth == 0:
            # At leaf nodes, call quiescence search for tactical stability
            q_score = self.quiescence_search(alpha, beta, player, q_depth=2)
            return q_score, None

        # Null-move pruning
        is_not_root_node = depth < self.current_search_depth
        if (
            depth >= 3
            and np.any(self.board)
            and is_not_root_node
            and beta < float("inf")
        ):
            # When we do a null move, we don't change the board or the score
            score, _ = self.negamax(
                depth - 3, -beta, -beta + 1, 3 - player, banned_moves_enabled
            )
            score = -score
            if score >= beta:
                return beta, None

        best_move, max_score = None, -float("inf")
        hash_move = tt_entry.get("move") if tt_entry else None
        moves = self.get_possible_moves(player, banned_moves_enabled, depth, hash_move)

        if not moves:
            return 0, None

        for i, move in enumerate(moves):
            r, c = move

            # Store original scores before making a move to allow for perfect restoration
            original_total_score = self.evaluator.total_score
            original_line_values = {
                line_id: self.evaluator.line_values[line_id]
                for line_id in self.evaluator.square_to_lines[(r, c)]
            }

            # Make the move and update the score incrementally
            self.evaluator.update_score(self.board, r, c, player)
            self.board[r, c] = player

            if self._check_win_by_move(r, c, player):
                score = SCORE_TABLE["FIVE"]["mine"]
            else:
                # Late Move Reduction / Extensions logic (no change needed here)
                reduction = 0
                extension = 0
                if self._is_vcf_threat(move, player, banned_moves_enabled):
                    extension = 1
                if depth >= 3 and i >= 3 and extension == 0:
                    reduction = 2

                search_depth = depth - 1 + extension - reduction
                if search_depth < 0:
                    search_depth = 0

                score, _ = self.negamax(
                    search_depth, -beta, -alpha, 3 - player, banned_moves_enabled
                )
                score = -score

                if reduction > 0 and score > alpha:
                    re_search_depth = depth - 1 + extension
                    score, _ = self.negamax(
                        re_search_depth, -beta, -alpha, 3 - player, banned_moves_enabled
                    )
                    score = -score

            # 3. Undo the move and restore the score state EXACTLY
            self.board[r, c] = EMPTY
            self.evaluator.total_score = original_total_score
            self.evaluator.line_values.update(original_line_values)

            # Add depth-aware scoring to distinguish between wins/losses at different speeds
            if abs(score) >= SCORE_TABLE["LIVE_FOUR"]["mine"]:
                if score > 0:
                    score -= 1  # Reward faster wins
                else:
                    score += 1  # Penalize faster losses

            if score > max_score:
                max_score = score
                best_move = (r, c)

            alpha = max(alpha, max_score)
            if alpha >= beta:
                if move != self.killer_moves[depth][0]:
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
                self.history_heuristic[move] += depth * depth
                break

        # Transposition table saving
        flag = "EXACT"
        if max_score <= original_alpha:
            flag = "UPPERBOUND"
        elif max_score >= beta:
            flag = "LOWERBOUND"
        existing_entry = self.transposition_table.get(board_hash)
        if not existing_entry or depth >= existing_entry["depth"]:
            self.transposition_table[board_hash] = {
                "score": max_score,
                "depth": depth,
                "flag": flag,
                "move": best_move,
                "age": self.search_generation,
                "sorted_moves": moves,
            }

        return max_score, best_move

    def quiescence_search(self, alpha, beta, player, q_depth):
        # The stand_pat score is the score of the current board state
        stand_pat_score = self.evaluator.get_current_score(player)

        if q_depth == 0:
            return stand_pat_score

        if stand_pat_score >= beta:
            return beta
        alpha = max(alpha, stand_pat_score)

        moves = self._get_qsearch_moves(player)  # Only considers threatening moves

        for move in moves:
            r, c = move

            # --- Incremental Update & Revert Logic ---
            original_total_score = self.evaluator.total_score
            original_line_values = {
                line_id: self.evaluator.line_values[line_id]
                for line_id in self.evaluator.square_to_lines[(r, c)]
            }

            self.evaluator.update_score(self.board, r, c, player)
            self.board[r, c] = player

            score = -self.quiescence_search(-beta, -alpha, 3 - player, q_depth - 1)

            self.board[r, c] = EMPTY
            self.evaluator.total_score = original_total_score
            self.evaluator.line_values.update(original_line_values)
            # --- End of Logic ---

            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def _get_qsearch_moves(self, player):
        opponent = 3 - player
        threatening_moves = []
        empty_cells = set()
        rows, cols = np.where(self.board != EMPTY)
        for r, c in zip(rows, cols):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    nr, nc = r + i, c + j
                    if (
                        0 <= nr < self.board_size
                        and 0 <= nc < self.board_size
                        and self.board[nr, nc] == EMPTY
                    ):
                        empty_cells.add((nr, nc))

        for r, c in empty_cells:
            self.board[r, c] = player
            if self._check_for_four(r, c, player):
                threatening_moves.append((r, c))
            self.board[r, c] = EMPTY

            self.board[r, c] = opponent
            if self._check_for_four(r, c, opponent):
                if (r, c) not in threatening_moves:
                    threatening_moves.append((r, c))
            self.board[r, c] = EMPTY
        return threatening_moves

    def _check_for_four(self, r, c, player):
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 4):
                nr, nc = r + i * dr, c + i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    break
                count += 1
            for i in range(1, 4):
                nr, nc = r - i * dr, c - i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and self.board[nr, nc] == player
                ):
                    break
                count += 1
            if count >= 4:
                return True
        return False

    def find_best_move(self, board_state, player, banned_moves_enabled):
        self.board = np.array(board_state)
        self.start_time = time.time()
        self.search_generation += 1
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
        self.history_heuristic.clear()
        best_move_so_far = None
        final_search_depth = 0

        self.evaluator.full_recalc(self.board)

        for depth in range(1, MAX_DEPTH + 1):
            try:
                self.current_search_depth = depth
                logger.info(f"--- Starting search at depth {depth} ---")
                score, move = self.negamax(
                    depth, -float("inf"), float("inf"), player, banned_moves_enabled
                )

                if move is not None:
                    best_move_so_far = move
                final_search_depth = depth

                elapsed_time = time.time() - self.start_time
                logger.info(
                    f"[Turn {self.current_turn}] Depth {depth} finished in {elapsed_time:.2f}s. Best move: {move}, Score: {score}"
                )

                if (
                    score != float("inf")
                    and score != -float("inf")
                    and abs(score) >= SCORE_TABLE["FIVE"]["mine"] - MAX_DEPTH
                ):
                    logger.info("Terminal sequence found. Halting search.")
                    break
            except TimeoutException:
                logger.warning(
                    f"[Turn {self.current_turn}] Timeout! Search at depth {depth} was forcefully interrupted."
                )
                final_search_depth = depth - 1
                logger.info(
                    f"[Turn {self.current_turn}] Returning best move from last completed depth ({final_search_depth})."
                )
                break
            except Exception:
                logger.exception(
                    f"[Turn {self.current_turn}] An unexpected error occurred during search at depth {depth}."
                )
                break

        if not best_move_so_far:
            possible_moves = self.get_possible_moves(
                player, banned_moves_enabled, 0, None
            )
            if possible_moves:
                best_move_so_far = possible_moves[0]
            logger.info(
                f"Search failed or timed out. Falling back to first possible move: {best_move_so_far}"
            )

        return best_move_so_far, final_search_depth

    def reset_for_new_game(self):
        """
        Resets the agent's state for a new game.
        Clears the transposition table, history heuristic, and killer moves.
        """
        self.transposition_table.clear()
        logger.info("New game signal received. Transposition table has been cleared.")


# --- Flask HTTP Server ---
agent = NegamaxAgent(GRID_SIZE)


@app.route("/get_move", methods=["POST"])
def get_move():
    try:
        data = request.get_json()
        board = data["board"]
        color_to_play = data.get("color_to_play")
        banned_moves_enabled = data.get("banned_moves_enabled", False)
        game_phase = data.get("game_phase", GamePhase.NORMAL)
        is_new_game = data.get("new_game", False)
        if is_new_game:
            agent.reset_for_new_game()

        """
        Logs
        """
        agent.board = np.array(board)
        stone_count = np.count_nonzero(agent.board)
        agent.current_turn = stone_count + 1
        logger.info(
            f"Received request for game_phase: {game_phase}, color_to_play: {color_to_play}"
        )

        agent.board = np.array(board)

        # --- Joseki (Opening Book) and Special Phase Logic ---
        # Swap2: P1 places 3 stones.
        if game_phase == GamePhase.SWAP2_P1_PLACE_3:
            if not agent.swap2_opening_sequence:
                if agent.joseki_book:
                    chosen_joseki = random.choice(agent.joseki_book)
                    agent.swap2_opening_sequence = list(chosen_joseki["joseki"][:3])
                    logger.info(
                        f"Swap2 Opening: Starting sequence for '{chosen_joseki['name_cn']}'."
                    )
                else:
                    logger.warning(
                        "Joseki book not loaded. Using default fallback opening."
                    )
                    agent.swap2_opening_sequence = [[7, 7, 1], [6, 7, 2], [6, 8, 1]]
            if agent.swap2_opening_sequence:
                next_move_data = agent.swap2_opening_sequence.pop(0)
                move = [next_move_data[0], next_move_data[1]]
                logger.info(
                    f"Returning move {3 - len(agent.swap2_opening_sequence)}/3 of opening sequence: {move}"
                )
                return jsonify({"move": move, "search_depth": 0})
        # Swap2: P2 chooses an action after P1 places 3 stones.
        elif game_phase == GamePhase.SWAP2_P2_CHOOSE_ACTION:
            logger.info("Evaluating board to choose action for Swap2.")

            score = agent.evaluate_board(BLACK)
            if score > SCORE_TABLE["LIVE_THREE"]["mine"]:
                choice = "TAKE_BLACK"
            elif score < -SCORE_TABLE["LIVE_THREE"]["opp"]:
                choice = "TAKE_WHITE"
            else:
                choice = "PLACE_2"
            logger.info(f"P2_CHOOSE_ACTION evaluation score: {score}, Choice: {choice}")
            return jsonify({"choice": choice})
        # Swap2: P1 chooses color after P2 places 2 more stones.
        elif game_phase == GamePhase.SWAP2_P1_CHOOSE_COLOR:
            black_score = agent.evaluate_board(BLACK)
            white_score = agent.evaluate_board(WHITE)
            choice = "CHOOSE_WHITE"
            if black_score >= white_score:
                choice = "CHOOSE_BLACK"
            logger.info(
                f"P1_CHOOSE_COLOR scores (B/W): {black_score}/{white_score}, Choice: {choice}"
            )
            return jsonify({"choice": choice})

        # Joseki matching
        opening_move = agent.find_opening_move(color_to_play)
        if opening_move:
            logger.info(
                f"Normal Phase: Opening book move found: {opening_move}. Returning immediately."
            )
            return jsonify(
                {
                    "move": [int(opening_move[0]), int(opening_move[1])],
                    "search_depth": 0,
                }
            )

        # --- Normal Search Logic and SWAP2_P2_PLACE_2 ---
        if color_to_play:
            best_move, search_depth = agent.find_best_move(
                board, color_to_play, banned_moves_enabled
            )
            if best_move:
                logger.info(f"Move found: {best_move} at depth {search_depth}")
                return jsonify(
                    {
                        "move": [int(best_move[0]), int(best_move[1])],
                        "search_depth": search_depth,
                    }
                )
            else:
                logger.warning("No best move found, trying to find any possible move.")
                moves = agent.get_possible_moves(
                    color_to_play, banned_moves_enabled, 0, None
                )
                if moves:
                    logger.info(f"Fallback move: {moves[0]}")
                    return jsonify(
                        {
                            "move": [int(moves[0][0]), int(moves[0][1])],
                            "search_depth": 0,
                        }
                    )
                else:
                    logger.error("No possible moves available.")
                    return jsonify({"move": None, "search_depth": 0})
        else:
            logger.error(
                "Request received without a 'color_to_play' in a move-required phase."
            )
            moves = agent.get_possible_moves(BLACK, False, 0, None)
            if moves:
                return jsonify({"move": [int(moves[0][0]), int(moves[0][1])]})
            else:
                return jsonify({"move": None})

    except Exception:
        logger.exception("An unhandled error occurred in the get_move endpoint.")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    try:
        logger.info("Starting Flask server on port 5003.")
        app.run(port=5003, debug=False)
    except Exception:
        logger.critical("Failed to start the Flask server.", exc_info=True)
