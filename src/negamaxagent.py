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
class GameLogger:
    def __init__(self, logs_dir="./logs"):
        self.logs_dir = logs_dir
        self.current_game_id = None
        self.logger = None
        self.handler = None

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def start_new_game(self):
        if self.handler and self.logger:
            self.logger.removeHandler(self.handler)
            self.handler.close()

        self.current_game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.logs_dir}/gamelog_{self.current_game_id}.json"
        self.logger = logging.getLogger(f"gamelog_{self.current_game_id}")
        self.logger.setLevel(logging.INFO)

        self.logger.handlers.clear()

        self.handler = logging.FileHandler(log_filename, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)

        self.logger.info(f"=== Game {self.current_game_id} Started ===")

    def get_logger(self):
        if not self.logger:
            self.start_new_game()
        return self.logger


game_logger = GameLogger()


# --- Configuration ---
GRID_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
TIME_LIMIT = 29.5  # Time limit for the AI to make a move, in seconds.
MAX_DEPTH = 50  # Max search depth for IDDFS
TOP_K_BY_DEPTH = [20, 16, 14, 12]


"""
The values for opponent's threats ('opp') are now significantly higher than 'mine'
to force the AI to block critical threats instead of making risky offensive moves.
"""
SYNERGY_FACTOR = 0.5
SCORE_TABLE = {
    "FIVE": {"mine": 100_000_000, "opp": 200_000_000},
    "LIVE_FOUR": {"mine": 80_000, "opp": 1_000_000},
    "LIVE_THREE": {"mine": 30_000, "opp": 100_000},
    "RUSH_FOUR": {"mine": 6_000, "opp": 20_000},
    "SLEEPY_THREE": {"mine": 1_300, "opp": 3_000},
    "LIVE_TWO": {"mine": 1_600, "opp": 2_000},
    "SLEEPY_TWO": {"mine": 100, "opp": 150},
    "POSITIONAL_BONUS_FACTOR": 5,
}
EXCEPTION_PATTERN = {
    "LIVE_THREE_FOUR": re.compile(r"0011100|00101100|00110100"),
}
PATTERNS_PLAYER = {
    "FIVE": re.compile(r"11111"),
    "LIVE_FOUR": re.compile(r"011110"),
    "RUSH_FOUR": re.compile(r"211110|011112|10111|11011|11101"),
    "LIVE_THREE": re.compile(r"01110|010110|011010"),
    "SLEEPY_THREE": re.compile(r"21110|01112|210110|011012|21101|10112"),
    "LIVE_TWO": re.compile(r"001100|01010|010010"),
    "SLEEPY_TWO": re.compile(r"21100|00112|21010|01012"),
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

        black_board = line_str
        for pattern_name, regex in PATTERNS_PLAYER.items():
            count = len(regex.findall(black_board))
            if count > 0:
                score += SCORE_TABLE[pattern_name]["mine"] * count

        white_board = line_str.replace("1", "X").replace("2", "1").replace("X", "2")
        for pattern_name, regex in PATTERNS_PLAYER.items():
            count = len(regex.findall(white_board))
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
        self.zobrist_stack = []

        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
        self.search_generation = 0
        self.last_depth_start_time = 0

        self.current_search_depth = 0
        self.nodes_evaluated_at_root = 0
        self.total_nodes_at_root = 0

        self.swap2_opening_sequence = []
        self.joseki_book = self._load_joseki()

        # Timeout
        self.safety_buffer = 0.2
        self.max_search_time = TIME_LIMIT - self.safety_buffer

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

    def _init_hash(self, player):
        h = np.uint64(0)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] != EMPTY:
                    h ^= zobrist_table[r, c, self.board[r, c]]
        if player == WHITE:
            h ^= zobrist_player_turn
        return h

    def _push_hash_state(self, r, c, player, current_hash):
        new_hash = current_hash

        old_piece = self.board[r, c]
        if old_piece != EMPTY:
            new_hash ^= zobrist_table[r, c, old_piece]

        new_hash ^= zobrist_table[r, c, player]
        new_hash ^= zobrist_player_turn
        self.zobrist_stack.append(
            {
                "old_hash": current_hash,
                "new_hash": new_hash,
                "position": (r, c),
                "old_piece": old_piece,
                "player": player,
            }
        )

        return new_hash

    def _pop_hash_state(self):
        if not self.zobrist_stack:
            raise Exception("Zobrist stack is empty!")

        state = self.zobrist_stack.pop()
        return state["old_hash"]

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
            if "011110" in line_str:
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

    def _find_patterns(self, player):
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

        # if patterns["FIVE"] > 0:
        #     patterns["RUSH_FOUR"] -= patterns["FIVE"] * 4

        return patterns

    def _calculate_synergy_at(self, r, c, player):
        if self._is_vcf_threat(r, c, player):
            return SCORE_TABLE["LIVE_FOUR"]["mine"]

        opponent = 3 - player
        offensive_patterns_found = []
        defensive_patterns_found = []
        synergy_score = 0

        affected_lines = self.evaluator.square_to_lines.get((r, c), [])
        original_piece = self.board[r, c]
        if original_piece != EMPTY:
            return 0

        for line_id in affected_lines:
            squares = self.evaluator.lines[line_id]
            self.board[r, c] = player
            line_str_mine = "".join(
                str(self.board[sq_r, sq_c]) for sq_r, sq_c in squares
            )
            player_centric_str = line_str_mine.replace(str(opponent), "2").replace(
                str(player), "1"
            )
            for name, regex in PATTERNS_PLAYER.items():
                if regex.search(player_centric_str):
                    offensive_patterns_found.append(name)
            self.board[r, c] = opponent
            line_str_opp = "".join(
                str(self.board[sq_r, sq_c]) for sq_r, sq_c in squares
            )
            opponent_centric_str = line_str_opp.replace(str(player), "2").replace(
                str(opponent), "1"
            )
            for name, regex in PATTERNS_PLAYER.items():
                if regex.search(opponent_centric_str):
                    defensive_patterns_found.append(name)
        self.board[r, c] = original_piece

        offensive_threat_count = len(offensive_patterns_found)
        defensive_block_count = len(defensive_patterns_found)

        if offensive_threat_count >= 2:
            base_bonus_score = sum(
                SCORE_TABLE[p]["mine"] for p in offensive_patterns_found
            )
            synergy_score += base_bonus_score
        if offensive_threat_count >= 1 and defensive_block_count >= 1:
            base_bonus_score = sum(
                SCORE_TABLE[p]["mine"] for p in offensive_patterns_found
            ) + sum(SCORE_TABLE[p]["opp"] for p in defensive_patterns_found)
            synergy_score += base_bonus_score
        if defensive_block_count >= 2:
            base_bonus_score = sum(
                SCORE_TABLE[p]["opp"] for p in defensive_patterns_found
            )
            synergy_score += base_bonus_score

        return synergy_score

    def _rate_move_statically(self, r, c, player):
        """
        Statically evaluate the threat of a single move for sorting purposes.
        """
        total_score_change = 0
        perspective = 1 if player == BLACK else -1

        # # --- 1. Positional Bonus ---
        # center = self.board_size // 2
        # dist = max(abs(r - center), abs(c - center))
        # total_score_change += (center - dist) * SCORE_TABLE["POSITIONAL_BONUS_FACTOR"]

        # --- 2. Calculate direct score gain using the EFFICIENT incremental method ---
        original_total_score = self.evaluator.total_score
        affected_lines = self.evaluator.square_to_lines.get((r, c), [])
        original_line_values = {
            line_id: self.evaluator.line_values[line_id] for line_id in affected_lines
        }

        # Perform incremental update
        self.evaluator.update_score(self.board, r, c, player)
        score_after_move = self.evaluator.total_score
        # Calculate gain and then REVERT the state
        my_gain = (score_after_move - original_total_score) * perspective
        total_score_change += my_gain

        self.evaluator.total_score = original_total_score
        self.evaluator.line_values.update(original_line_values)

        # --- 3. Calculate Synergy Bonus ---
        total_score_change += self._calculate_synergy_at(r, c, player)

        return total_score_change

    def get_possible_moves(self, player, banned_moves_enabled, depth, hash_move):
        """
        Uses a sophisticated, multi-layered sorting approach,
        including static threat analysis, to achieve
        Threat Space Search.
        """
        # board_hash = self._compute_hash(player)
        # tt_entry = self.transposition_table.get(board_hash)
        # if tt_entry and "sorted_moves" in tt_entry:
        #     return tt_entry["sorted_moves"]

        moves = self._get_candidate_moves(player, banned_moves_enabled)
        opponent = 3 - player
        urgent_defenses = []
        urgent_attacks = []
        hash_moves = []
        killer_moves_list = []
        regular_moves = []
        move_scores = {}

        for move in moves:
            r, c = move
            if self._is_vcf_threat(r, c, opponent):
                urgent_defenses.append(move)
            elif self._is_vcf_threat(r, c, player):
                urgent_attacks.append(move)
            else:
                regular_moves.append(move)

        if len(urgent_defenses) == 1 and depth > 0:
            return urgent_defenses
        elif len(urgent_defenses) > 1 and depth > 3:
            defense_scores = {
                m: self._rate_move_statically(m[0], m[1], player)
                for m in urgent_defenses
            }
            sorted_defenses = sorted(
                urgent_defenses, key=lambda m: defense_scores[m], reverse=True
            )
            return sorted_defenses[: min(3, len(sorted_defenses))]

        # Hash Moves
        if hash_move and hash_move in moves:
            if hash_move not in urgent_defenses and hash_move not in urgent_attacks:
                hash_moves.append(hash_move)
                if hash_move in regular_moves:
                    regular_moves.remove(hash_move)

        # Killer Moves
        killers = self.killer_moves[depth]
        for killer in [killers[0], killers[1]]:
            if (
                killer
                and killer in moves
                and killer not in urgent_defenses
                and killer not in urgent_attacks
                and killer not in hash_moves
            ):
                killer_moves_list.append(killer)
                if killer in regular_moves:
                    regular_moves.remove(killer)

        # Regular Moves + static scoring
        if regular_moves:
            move_scores = {
                m: self._rate_move_statically(m[0], m[1], player) for m in regular_moves
            }
            regular_moves.sort(key=lambda m: move_scores[m], reverse=True)

        final_ordered_list = []
        final_ordered_list.extend(urgent_defenses)
        final_ordered_list.extend(urgent_attacks)
        final_ordered_list.extend(hash_moves)
        final_ordered_list.extend(killer_moves_list)
        final_ordered_list.extend(regular_moves)

        if depth > 0 and len(final_ordered_list) > 1:
            absolute_depth = self.current_search_depth - depth
            if absolute_depth < 0:
                absolute_depth = 0

            critical_moves_count = (
                len(urgent_defenses)
                + len(urgent_attacks)
                + len(hash_moves)
                + len(killer_moves_list)
            )
            top_k = min(
                TOP_K_BY_DEPTH[
                    (absolute_depth if absolute_depth < len(TOP_K_BY_DEPTH) else -1)
                ],
                critical_moves_count + 2,
            )

            if len(final_ordered_list) > top_k:
                final_ordered_list = final_ordered_list[:top_k]

        return final_ordered_list

    def _find_live_three_ends(self, player):
        completion_moves = set()

        live_three_regex = EXCEPTION_PATTERN["LIVE_THREE_FOUR"]

        for line_id, squares in self.evaluator.lines.items():
            line_str_raw = "".join(str(self.board[r, c]) for r, c in squares)
            if player == BLACK:
                line_str = line_str_raw
            else:
                line_str = (
                    line_str_raw.replace("1", "T").replace("2", "1").replace("T", "2")
                )

            for match in live_three_regex.finditer(line_str):
                matched_pattern = match.group(0)
                start_index = match.start()

                if matched_pattern == "0011100":
                    completion_moves.add(squares[start_index + 1])
                    completion_moves.add(squares[start_index + 5])
                elif matched_pattern == "00101100":
                    # For _X_XX_, the key empty cell is at relative position 2.
                    completion_moves.add(squares[start_index + 3])
                elif matched_pattern == "00110100":
                    # For _XX_X_, the key empty cell is at relative position 3.
                    completion_moves.add(squares[start_index + 4])

        return list(completion_moves)

    def _is_killer_move_candidate(self, r, c, player):
        if self._check_win_by_move(r, c, player):
            return False

        return self._is_vcf_threat(r, c, player)

    def _update_killer_moves(self, depth, move, player):
        r, c = move
        if not self._is_killer_move_candidate(r, c, player):
            return

        if move == self.killer_moves[depth][0]:
            return
        if move == self.killer_moves[depth][1]:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move
        else:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def negamax(
        self, depth, alpha, beta, player, banned_moves_enabled, current_hash=None
    ):
        self._check_timeout()

        if depth <= 0:
            q_score = self.quiescence_search(alpha, beta, player, q_depth=1)
            return q_score, None

        original_alpha = alpha
        if current_hash is None:
            current_hash = self._init_hash(player)
        board_hash = current_hash
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

        best_move, max_score = None, -float("inf")
        hash_move = tt_entry.get("move") if tt_entry else None
        moves = self.get_possible_moves(player, banned_moves_enabled, depth, hash_move)

        if not moves:
            return 0, None

        for i, move in enumerate(moves):
            self._check_timeout()

            r, c = move

            # update hash
            original_total_score = self.evaluator.total_score
            original_line_values = {
                line_id: self.evaluator.line_values[line_id]
                for line_id in self.evaluator.square_to_lines.get((r, c), [])
            }
            new_hash = self._push_hash_state(r, c, player, current_hash)

            original_total_score = self.evaluator.total_score
            original_line_values = {
                line_id: self.evaluator.line_values[line_id]
                for line_id in self.evaluator.square_to_lines.get((r, c), [])
            }

            self.evaluator.update_score(self.board, r, c, player)
            self.board[r, c] = player

            if self._check_win_by_move(r, c, player):
                self.board[r, c] = EMPTY
                self.evaluator.total_score = original_total_score
                self.evaluator.line_values.update(original_line_values)
                return SCORE_TABLE["FIVE"]["mine"], (r, c)
            else:
                # The core PVS logic
                if i == 0:
                    # 1. Principal Variation) - full search
                    score, _ = self.negamax(
                        depth - 1,
                        -beta,
                        -alpha,
                        3 - player,
                        banned_moves_enabled,
                        new_hash,
                    )
                    score = -score
                else:
                    # 2. Other moves with quick "scout search" (zero-window search)(-alpha-1, -alpha)
                    score, _ = self.negamax(
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        3 - player,
                        banned_moves_enabled,
                        new_hash,
                    )
                    score = -score

                    # 3. if score > alpha, do a full search again
                    if alpha < score < beta:
                        score, _ = self.negamax(
                            depth - 1,
                            -beta,
                            -alpha,
                            3 - player,
                            banned_moves_enabled,
                            new_hash,
                        )
                        score = -score

            # 3. Undo the move and restore the score state EXACTLY
            self.board[r, c] = EMPTY
            self.evaluator.total_score = original_total_score
            self.evaluator.line_values.update(original_line_values)
            old_hash = self._pop_hash_state()

            # Add depth-aware scoring to distinguish between wins/losses at different speeds
            if abs(score) >= SCORE_TABLE["LIVE_FOUR"]["mine"]:
                if score > 0:
                    score -= 1  # Reward faster wins
                else:
                    score += 1  # Penalize faster losses

            if score > max_score:
                max_score = score
                best_move = (r, c)

            if score >= beta:
                self._update_killer_moves(depth, move, player)
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
                # "sorted_moves": moves,
            }

        return max_score, best_move

    def quiescence_search(self, alpha, beta, player, q_depth):
        self._check_timeout()

        # The stand_pat score is the score of the current board state
        stand_pat_score = self.evaluator.get_current_score(player)

        if q_depth == 0:
            return stand_pat_score

        if stand_pat_score >= beta:
            return beta
        alpha = max(alpha, stand_pat_score)

        moves = self._get_qsearch_moves(player)  # Only considers threatening moves

        for move in moves:
            self._check_timeout()

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

            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    def _get_qsearch_moves(self, player):
        opponent = 3 - player
        threatening_moves = []

        candidate_moves = set()
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
                        candidate_moves.add((nr, nc))

        for r, c in candidate_moves:
            if self._is_vcf_threat(r, c, player) or self._is_vcf_threat(r, c, opponent):
                threatening_moves.append((r, c))

        return list(set(threatening_moves))

    def _is_vcf_threat(self, r, c, player):
        if self.board[r, c] != EMPTY:
            return False
        self.board[r, c] = player
        patterns = self._find_patterns(player)
        self.board[r, c] = EMPTY

        if patterns.get("LIVE_FOUR", 0) >= 1:
            return True
        if patterns.get("RUSH_FOUR", 0) >= 1:
            return True
        if patterns.get("LIVE_THREE", 0) >= 1:
            return True
        if patterns.get("LIVE_THREE", 0) >= 1 and patterns.get("RUSH_FOUR", 0) >= 1:
            return True
        return False

    def _get_candidate_moves(self, player, banned_moves_enabled):
        """
        Generates a set of plausible moves around existing stones.
        """
        if not np.any(self.board):
            return [(self.board_size // 2, self.board_size // 2)]

        moves = set()
        radius = 2  # Search radius around existing stones
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

        if not moves:  # If board is full except for a few spots
            empty_cells = np.argwhere(self.board == EMPTY)
            return [tuple(cell) for cell in empty_cells]

        return list(moves)

    def _find_urgent_defense_moves(self, opponent):
        urgent_defense_moves = []

        live_four_blocks = self._find_pattern_completion_moves(opponent, "LIVE_FOUR")
        urgent_defense_moves.extend(live_four_blocks)

        rush_four_blocks = self._find_pattern_completion_moves(opponent, "RUSH_FOUR")
        urgent_defense_moves.extend(rush_four_blocks)

        live_three_blocks = self._find_pattern_completion_moves(opponent, "LIVE_THREE")
        urgent_defense_moves.extend(live_three_blocks)

        return urgent_defense_moves

    def _find_pattern_completion_moves(self, player, pattern_name):
        completion_moves = set()
        pattern_regex = PATTERNS_PLAYER[pattern_name]

        for line_id, squares in self.evaluator.lines.items():
            line_str_raw = "".join(str(self.board[r, c]) for r, c in squares)

            if player == BLACK:
                line_str = line_str_raw
            else:
                line_str = (
                    line_str_raw.replace("1", "T").replace("2", "1").replace("T", "2")
                )

            for match in pattern_regex.finditer(line_str):
                matched_pattern = match.group(0)
                start_index = match.start()

                if pattern_name == "LIVE_FOUR":
                    if matched_pattern == "011110":
                        completion_moves.add(squares[start_index])
                        completion_moves.add(squares[start_index + 5])
                elif pattern_name == "RUSH_FOUR":
                    if matched_pattern == "211110":
                        completion_moves.add(squares[start_index + 5])
                    elif matched_pattern == "011112":
                        completion_moves.add(squares[start_index])
                    elif matched_pattern == "10111":
                        completion_moves.add(squares[start_index + 1])
                    elif matched_pattern == "11011":
                        completion_moves.add(squares[start_index + 2])
                    elif matched_pattern == "11101":
                        completion_moves.add(squares[start_index + 3])
                elif pattern_name == "LIVE_THREE":
                    if matched_pattern == "01110":
                        completion_moves.add(squares[start_index])
                        completion_moves.add(squares[start_index + 4])
                    elif matched_pattern == "010110":
                        completion_moves.add(squares[start_index])
                        completion_moves.add(squares[start_index + 2])
                        completion_moves.add(squares[start_index + 5])
                    elif matched_pattern == "011010":
                        completion_moves.add(squares[start_index])
                        completion_moves.add(squares[start_index + 3])
                        completion_moves.add(squares[start_index + 5])

        valid_moves = []
        for r, c in completion_moves:
            if (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == EMPTY
            ):
                valid_moves.append((r, c))

        return valid_moves

    def find_best_move(self, board_state, player, banned_moves_enabled):
        self.board = np.array(board_state)
        self.start_time = time.time()
        self.search_generation += 1
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
        best_move_so_far = None
        final_search_depth = 0

        # 1st Priority: Check for immediate win
        candidate_moves = self._get_candidate_moves(player, banned_moves_enabled)
        my_win_moves = [
            m for m in candidate_moves if self._check_win_by_move(m[0], m[1], player)
        ]
        if my_win_moves:
            logger.info(
                f"[Turn {self.current_turn}] High-level rule: Found immediate win at {my_win_moves[0]}. Skipping search."
            )
            return my_win_moves[0], 0  # Return move, depth=0
        opponent = 3 - player
        opponent_win_moves = [
            m for m in candidate_moves if self._check_win_by_move(m[0], m[1], opponent)
        ]
        if opponent_win_moves:
            logger.info(
                f"[Turn {self.current_turn}] High-level rule: Blocking opponent's win at {opponent_win_moves[0]}. Skipping search."
            )
            return opponent_win_moves[0], 0

        # 2nd Priority: Check for opponent's urgent threats
        urgent_defenses = []
        patterns = self._find_patterns(opponent)
        if (
            patterns.get("LIVE_FOUR", 0) > 0
            or patterns.get("RUSH_FOUR", 0) > 0
            or patterns.get("LIVE_THREE", 0) > 0
            or (
                patterns.get("LIVE_THREE", 0) >= 1
                and patterns.get("SLEEPY_THREE", 0) >= 1
            )
        ):
            urgent_defenses = self._find_urgent_defense_moves(opponent)
            if urgent_defenses:
                valid_defenses = [m for m in urgent_defenses if m in candidate_moves]
                if valid_defenses:
                    if len(valid_defenses) == 1:
                        logger.info(
                            f"[Turn {self.current_turn}] Urgent Defense Rule: Found urgent defense at {valid_defenses[0]}. Skipping search."
                        )
                        return valid_defenses[0], 0
                    else:
                        move_scores = {
                            m: self._rate_move_statically(m[0], m[1], player)
                            for m in valid_defenses
                        }
                        chosen_move = max(move_scores, key=move_scores.get)
                        logger.info(
                            f"[Turn {self.current_turn}] Urgent Defense Rule: Found urgent defense at {chosen_move}. Skipping search."
                        )
                        return chosen_move, 0
        # EXCEPTION: forcedly LIVE_THREE-> LIVE_FOUR(offensive) or SLEEPY_THREE(defense) to avoid OVERLAPPING
        live_three_completion_moves = self._find_live_three_ends(player)
        valid_completion_moves = [
            m for m in live_three_completion_moves if m in candidate_moves
        ]
        if valid_completion_moves:
            chosen_move = None
            if len(valid_completion_moves) == 1:
                # If there's only one way to complete the three, take it.
                chosen_move = valid_completion_moves[0]
            else:
                # If there are two endpoints, evaluate which one is better.
                move_scores = {
                    m: self._rate_move_statically(m[0], m[1], player)
                    for m in valid_completion_moves
                }
                chosen_move = max(move_scores, key=move_scores.get)

            logger.info(
                f"[Turn {self.current_turn}] Live_3 -> Live_4 Rule: Found LIVE_THREE. Choosing best endpoint {chosen_move} from {valid_completion_moves}. Skipping search."
            )
            return chosen_move, 0

        # Normal search
        for depth in range(1, MAX_DEPTH + 1):
            try:
                self.current_search_depth = depth
                logger.info(f"--- Starting search at depth {depth} ---")

                score, move = self.negamax(
                    depth, -float("inf"), float("inf"), player, banned_moves_enabled
                )
                last_score = score

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
        global logger, game_logger
        game_logger.start_new_game()
        logger = game_logger.get_logger()

        self.transposition_table.clear()
        self.evaluator.full_recalc(self.board)
        logger.info("New game signal received. Transposition table has been cleared.")

    def _check_timeout(self):
        if time.time() - self.start_time > self.max_search_time:
            raise TimeoutException()


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
        global logger
        if not logger:
            logger = game_logger.get_logger()
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
    app.run(port=5003, debug=False)
