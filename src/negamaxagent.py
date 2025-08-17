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
from typing import List, Tuple, Optional
from dataclasses import dataclass


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
        self.logger.setLevel(logging.DEBUG)

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
MOVE_DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
# VCF/VCT
MAX_VCF_DEPTH = 12
MAX_VCT_DEPTH = 6
VCF_NODE_LIMIT = 5000  # Maximum nodes per search
VCF_TIME_LIMIT = 3.0  # Maximum time for VCF search
VCT_TIME_LIMIT = 2.0  # Maximum time for VCT search
VCT_PENALTY = 100

"""
The values for opponent's threats ('opp') are now significantly higher than 'mine'
to force the AI to block critical threats instead of making risky offensive moves.
"""
SYNERGY_FACTOR = 0.5
SCORE_TABLE = {
    "FIVE": {"mine": 100_000_000, "opp": 200_000_000},
    "LIVE_FOUR": {"mine": 80_000, "opp": 1_000_000},
    "LIVE_THREE": {"mine": 30_000, "opp": 150_000},
    "RUSH_FOUR": {"mine": 10_000, "opp": 20_000},
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


@dataclass
class VCFResult:
    is_winning: bool
    depth: int
    winning_move: Optional[Tuple[int, int]] = None
    threat_sequence: List[Tuple[int, int]] = None
    nodes_searched: int = 0
    time_elapsed: float = 0.0

    def __post_init__(self):
        if self.threat_sequence is None:
            self.threat_sequence = []


@dataclass
class VCTResult:
    is_winning: bool
    depth: int
    winning_move: Optional[Tuple[int, int]] = None
    threat_sequence: List[Tuple[int, int]] = None
    nodes_searched: int = 0
    time_elapsed: float = 0.0

    def __post_init__(self):
        if self.threat_sequence is None:
            self.threat_sequence = []


class ThreatCache:
    """Cache for threat evaluations to avoid redundant calculations"""

    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, board_hash, position):
        key = (board_hash, position)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, board_hash, position, value):
        if len(self.cache) >= self.max_size:
            # Simple eviction - remove 20% oldest entries
            evict_count = self.max_size // 5
            for _ in range(evict_count):
                self.cache.pop(next(iter(self.cache)))

        key = (board_hash, position)
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class ThreatDetector:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.threat_cache = ThreatCache()

    def find_four_threats(self, board, player, max_threats=10):
        """Find all positions that create four-in-a-row threats (sleep-four patterns)"""
        threats = []

        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != EMPTY:
                    continue
                if len(threats) >= max_threats:
                    break

                # Check cache first
                board_hash = hash(board.tobytes())
                cached = self.threat_cache.get(board_hash, (r, c))
                if cached is not None:
                    if cached:
                        threats.append((r, c))
                    continue

                board[r, c] = player
                is_threat = self._check_four_threat(board, r, c, player)
                board[r, c] = EMPTY

                self.threat_cache.put(board_hash, (r, c), is_threat)
                if is_threat:
                    threats.append((r, c))

        return threats

    def find_live_three_threats(self, board, player, max_threats=10):
        """Find all positions that create live three threats"""
        threats = []

        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != EMPTY:
                    continue
                if len(threats) >= max_threats:
                    break

                board[r, c] = player
                is_threat = self._check_live_three(board, r, c, player)
                board[r, c] = EMPTY

                if is_threat:
                    threats.append((r, c))

        return threats

    def _check_patterns_at(self, board, r, c, player, pattern_names_to_check):
        directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        ]  # Horizontal, Vertical, Diagonals

        for dr, dc in directions:
            line = []
            for i in range(-5, 6):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    line.append(str(board[nr, nc]))
                else:
                    line.append(str(3 - player))

            line_str = "".join(line)
            if player == WHITE:
                line_str = (
                    line_str.replace("2", "X").replace("1", "2").replace("X", "1")
                )

            for pattern_name in pattern_names_to_check:
                regex = PATTERNS_PLAYER.get(pattern_name)
                if regex and regex.search(line_str):
                    return True

        return False

    def _check_four_threat(self, board, r, c, player):
        four_threat_patterns = ["LIVE_FOUR", "RUSH_FOUR"]
        return self._check_patterns_at(board, r, c, player, four_threat_patterns)

    def _check_live_three(self, board, r, c, player):
        three_threat_patterns = ["LIVE_THREE"]
        return self._check_patterns_at(board, r, c, player, three_threat_patterns)

    def _check_urgent_threat_in_direction(self, board, r, c, player, dr, dc):
        """Check if there's a threat in specific direction"""
        count = 1

        for i in range(1, 5):
            nr, nc = r + i * dr, c + i * dc
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                break
            if board[nr, nc] != player:
                break
            count += 1

        for i in range(1, 5):
            nr, nc = r - i * dr, c - i * dc
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                break
            if board[nr, nc] != player:
                break
            count += 1

        return count >= 4

    def get_defensive_moves(self, board, threat_move, player, max_defenses):
        """Get all moves that can defend against a threat"""
        defensive_moves = []
        r, c = threat_move
        seen = set()

        # Temporarily place the threat
        board[r, c] = player

        # Find critical defensive points
        for dr, dc in MOVE_DIRECTIONS:
            # if not self._check_urgent_threat_in_direction(board, r, c, player, dr, dc):
            #     continue
            # Find empty spots in the threat line
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                    continue
                if (nr, nc) == (r, c):
                    continue
                if board[nr, nc] != EMPTY:
                    continue
                if (nr, nc) in seen:
                    continue

                # Check if this defense actually blocks the threat
                board[nr, nc] = 3 - player
                still_threat = self._check_four_threat(board, r, c, player)
                board[nr, nc] = EMPTY

                if not still_threat:
                    defensive_moves.append((nr, nc))
                    seen.add((nr, nc))
                    if len(defensive_moves) >= max_defenses:
                        board[r, c] = EMPTY
                        return defensive_moves

        board[r, c] = EMPTY

        # Remove duplicates and return
        return list(set(defensive_moves))

    def find_double_threats(self, board, player, max_threats=10):
        """Find moves that create multiple threats simultaneously"""
        double_threats = []

        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != EMPTY:
                    continue
                if len(double_threats) >= max_threats:
                    break

                # Count different types of threats
                board[r, c] = player
                threat_count = 0
                if self._check_four_threat(board, r, c, player):
                    threat_count += 2
                if self._check_live_three(board, r, c, player):
                    threat_count += 1

                board[r, c] = EMPTY

                if threat_count >= 2:
                    double_threats.append((r, c))

        return double_threats


class VCFSearcher:
    def __init__(self, board_size=15, agent=None):
        self.board_size = board_size
        self.threat_detector = ThreatDetector(board_size)
        self.search_nodes = 0
        self.max_vcf_depth = 20

        self.agent = agent

        self.search_nodes = 0
        self.max_nodes = VCF_NODE_LIMIT
        self.start_time = 0
        self.time_limit = VCF_TIME_LIMIT

    def search(self, board, player, max_depth=None, time_limit=None):
        if max_depth is None:
            max_depth = self.max_vcf_depth
        if time_limit is None:
            time_limit = self.time_limit

        self.search_nodes = 0
        self.start_time = time.time()
        self.threat_detector.threat_cache.clear()

        result = self._vcf_recursive(
            board.copy(), player, 0, max_depth, [], set(), True
        )
        result.nodes_searched = self.search_nodes
        result.time_elapsed = time.time() - self.start_time
        return result

    def _vcf_recursive(
        self,
        board,
        player,
        depth,
        max_depth,
        threat_sequence,
        visited_states,
        is_attacker_turn=True,
    ):
        # Check limits
        self.search_nodes += 1
        if self.search_nodes > self.max_nodes:
            return VCFResult(False, depth)
        if time.time() - self.start_time > self.time_limit:
            return VCFResult(False, depth)
        if self.agent and hasattr(self.agent, "_check_timeout"):
            try:
                self.agent._check_timeout()
            except:
                return VCFResult(False, depth)

        # Cycle Detection
        if depth >= max_depth:
            return VCFResult(False, depth)
        board_hash = hash(board.tobytes())
        if board_hash in visited_states:
            return VCFResult(False, depth)
        visited_states.add(board_hash)

        if is_attacker_turn:
            # Attacker's turn: need ALL defenses to fail
            four_threats = self.threat_detector.find_four_threats(
                board, player, max_threats=5  # Limit threats examined
            )

            if not four_threats:
                visited_states.discard(board_hash)
                return VCFResult(False, depth)

            # Try each threat (but stop early if winning)
            for threat_move in four_threats:
                r, c = threat_move
                board[r, c] = player

                # Check immediate win
                if self._check_win(board, r, c, player):
                    board[r, c] = EMPTY
                    visited_states.discard(board_hash)
                    return VCFResult(
                        True, depth + 1, threat_move, threat_sequence + [threat_move]
                    )

                # Get defensive moves
                defensive_moves = self.threat_detector.get_defensive_moves(
                    board, threat_move, player, max_defenses=3  # Limit defenses
                )

                if not defensive_moves:
                    # No defense = win
                    board[r, c] = EMPTY
                    visited_states.discard(board_hash)
                    return VCFResult(
                        True, depth + 1, threat_move, threat_sequence + [threat_move]
                    )

                # Check if ALL defenses fail (with early termination)
                all_defenses_fail = True

                for defense in defensive_moves:
                    dr, dc = defense
                    if board[dr, dc] != EMPTY:
                        continue

                    board[dr, dc] = 3 - player

                    # Defender's turn
                    result = self._vcf_recursive(
                        board,
                        player,
                        depth + 2,
                        max_depth,
                        threat_sequence + [threat_move, defense],
                        visited_states.copy(),
                        True,  # Back to attacker
                    )

                    board[dr, dc] = EMPTY

                    if not result.is_winning:
                        # Found successful defense - this threat fails
                        all_defenses_fail = False
                        break  # PRUNING: No need to check other defenses

                board[r, c] = EMPTY

                if all_defenses_fail:
                    # This threat wins!
                    visited_states.discard(board_hash)
                    return VCFResult(
                        True, depth + 1, threat_move, threat_sequence + [threat_move]
                    )

        visited_states.discard(board_hash)
        return VCFResult(False, depth)

    # def _sort_threats_by_priority(self, board, threats, player):
    #     """Sort threats by their strategic value"""
    #     threat_scores = []

    #     for threat in threats:
    #         r, c = threat

    #         # Check connectivity to existing stones
    #         connectivity = 0
    #         for dr in [-1, 0, 1]:
    #             for dc in [-1, 0, 1]:
    #                 if dr == 0 and dc == 0:
    #                     continue
    #                 nr, nc = r + dr, c + dc
    #                 if (
    #                     0 <= nr < self.board_size
    #                     and 0 <= nc < self.board_size
    #                     and board[nr, nc] == player
    #                 ):
    #                     connectivity += 1

    #         threat_scores.append((threat, connectivity * 2))

    #     threat_scores.sort(key=lambda x: x[1], reverse=True)
    #     return [t[0] for t in threat_scores]

    def _check_win(self, board, r, c, player):
        """Check if move creates a win"""
        for dr, dc in MOVE_DIRECTIONS:
            count = 1

            # Count in positive direction
            for i in range(1, 5):
                nr, nc = r + i * dr, c + i * dc
                if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                    break
                if board[nr, nc] != player:
                    break
                count += 1

            # Count in negative direction
            for i in range(1, 5):
                nr, nc = r - i * dr, c - i * dc
                if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                    break
                if board[nr, nc] != player:
                    break
                count += 1

            if count >= 5:
                return True

        return False


class VCTSearcher:
    """Victory by Continuous Threat searcher"""

    def __init__(self, board_size=15, agent=None):
        self.board_size = board_size
        self.threat_detector = ThreatDetector(board_size)
        self.vcf_searcher = VCFSearcher(board_size, agent)

        self.agent = agent

        self.search_nodes = 0
        self.max_nodes = VCF_NODE_LIMIT // 2  # VCT gets fewer nodes
        self.start_time = 0
        self.time_limit = VCT_TIME_LIMIT

    def search(self, board, player, max_depth=None, time_limit=None):
        """
        Search for VCT (Victory by Continuous Threat)
        Uses open threes and threat combinations
        """
        if max_depth is None:
            max_depth = self.max_vct_depth
        if time_limit is None:
            time_limit = self.time_limit

        self.search_nodes = 0
        self.start_time = time.time()

        result = self._vct_recursive(board.copy(), player, 0, max_depth, [], set())
        result.nodes_searched = self.search_nodes
        result.time_elapsed = time.time() - self.start_time
        return result

    def _vct_recursive(
        self, board, player, depth, max_depth, threat_sequence, visited_states
    ):
        """Recursive VCT search"""
        # Check limits
        self.search_nodes += 1
        if self.search_nodes > self.max_nodes:
            return VCTResult(False, depth)
        if time.time() - self.start_time > self.time_limit:
            return VCTResult(False, depth)
        if self.agent and hasattr(self.agent, "_check_timeout"):
            try:
                self.agent._check_timeout()
            except:
                return VCTResult(False, depth)

        if depth >= max_depth:
            return VCTResult(False, depth)

        # Transition to VCF
        vcf_result = self.vcf_searcher.search(board, player, max_depth=4)
        if vcf_result.is_winning:
            return VCTResult(
                True,
                depth + vcf_result.depth,
                vcf_result.winning_move,
                threat_sequence + vcf_result.threat_sequence,
            )

        # Cycle detection
        board_hash = hash(board.tobytes())
        if board_hash in visited_states:
            return VCTResult(False, depth)
        visited_states.add(board_hash)

        # Find different types of threats
        three_threats = self.threat_detector.find_live_three_threats(
            board, player, max_threats=5
        )
        double_threats = self.threat_detector.find_double_threats(board, player)
        all_threats = []
        threat_set = set()
        for threat in double_threats:
            if threat not in threat_set:
                all_threats.append(threat)
                threat_set.add(threat)
        for threat in three_threats:
            if threat not in threat_set and len(all_threats) < 8:
                all_threats.append(threat)
                threat_set.add(threat)
        if not all_threats:
            visited_states.discard(board_hash)
            return VCTResult(False, depth)

        # Sort threats by priority
        for threat_move in all_threats[:6]:  # Limit to 6 best threats
            r, c = threat_move
            board[r, c] = player

            # Check win
            if self.vcf_searcher._check_win(board, r, c, player):
                board[r, c] = EMPTY
                visited_states.discard(board_hash)
                return VCTResult(
                    True, depth + 1, threat_move, threat_sequence + [threat_move]
                )

            # Get defensive moves - more comprehensive for VCT
            defensive_moves = self.threat_detector.get_defensive_moves(
                board, threat_move, player, max_defenses=4
            )

            # For double threats, we might need to consider more defenses
            if threat_move in double_threats:
                additional_defenses = self._get_multi_threat_defenses(
                    board, threat_move, player
                )
                for def_move in additional_defenses:
                    if def_move not in defensive_moves and len(defensive_moves) < 5:
                        defensive_moves.append(def_move)

            if not defensive_moves:
                # No defense = win
                board[r, c] = EMPTY
                visited_states.discard(board_hash)
                return VCTResult(
                    True, depth + 1, threat_move, threat_sequence + [threat_move]
                )

            # Too many defenses = not forcing enough for VCT
            if len(defensive_moves) > 5:
                board[r, c] = EMPTY
                continue

            # Check if all defenses fail
            all_defenses_fail = True

            for defense in defensive_moves[:3]:  # Check top 3 defenses
                dr, dc = defense
                if board[dr, dc] != EMPTY:
                    continue

                board[dr, dc] = 3 - player

                result = self._vct_recursive(
                    board,
                    player,
                    depth + 2,
                    max_depth,
                    threat_sequence + [threat_move, defense],
                    visited_states.copy(),
                )

                board[dr, dc] = EMPTY

                if not result.is_winning:
                    all_defenses_fail = False
                    break  # Pruning: one successful defense is enough

            board[r, c] = EMPTY

            if all_defenses_fail:
                visited_states.discard(board_hash)
                return VCTResult(
                    True, depth + 1, threat_move, threat_sequence + [threat_move]
                )

        visited_states.discard(board_hash)
        return VCTResult(False, depth)

    def _get_multi_threat_defenses(self, board, threat_move, player):
        """
        Get defensive moves specifically for multi-threat situations
        These are positions that might defend against multiple threats at once
        """
        r, c = threat_move
        defenses = []

        # Check intersections of threat lines
        for dr, dc in MOVE_DIRECTIONS:
            # Look for key defensive points
            for i in [-2, -1, 1, 2]:
                nr, nc = r + i * dr, c + i * dc
                if (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and board[nr, nc] == EMPTY
                ):
                    # Check if this position defends multiple threats
                    board[nr, nc] = 3 - player

                    # Quick check if it reduces threat level
                    is_still_double = False
                    board[r, c] = player  # Temporarily place threat

                    if self.threat_detector._check_four_threat(
                        board, r, c, player
                    ) or self.threat_detector._check_live_three(board, r, c, player):
                        # Count remaining threats
                        threat_count = 0
                        if self.threat_detector._check_four_threat(board, r, c, player):
                            threat_count += 2
                        if self.threat_detector._check_live_three(board, r, c, player):
                            threat_count += 1

                        if threat_count >= 2:
                            is_still_double = True

                    board[r, c] = EMPTY
                    board[nr, nc] = EMPTY

                    if not is_still_double:
                        defenses.append((nr, nc))

        return defenses[:3]


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

        self.vcf_searcher = VCFSearcher(board_size, self)
        self.vct_searcher = VCTSearcher(board_size, self)
        self.threat_detector = ThreatDetector(board_size)
        self.vcf_checks = 0
        self.vct_checks = 0
        self.vcf_wins_found = 0
        self.vct_wins_found = 0

        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]

        self.search_generation = 0
        self.current_search_depth = 0
        self.nodes_evaluated_at_root = 0
        self.total_nodes_at_root = 0

        self.swap2_opening_sequence = []
        self.joseki_book = self._load_joseki()

        # Timeout
        self.safety_buffer = 0.2
        self.max_search_time = TIME_LIMIT - self.safety_buffer
        self.panic_mode = False

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

        # 1st stone: Black moves
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

        # 2nd stone: White moves
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

            # choose preferred based on trend
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

        # 3rd stone: Black moves
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

            # choose preferred based on trend
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
        for dr, dc in MOVE_DIRECTIONS:
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
        for dr, dc in MOVE_DIRECTIONS:
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
        for dr, dc in MOVE_DIRECTIONS:
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

    def evaluate_board(self, color_to_play):
        self.evaluator.full_recalc(self.board)
        return self.evaluator.get_current_score(color_to_play)

    def _calculate_synergy_at(self, r, c, player):
        if self.board[r, c] != EMPTY:
            return 0

        affected_lines = self.evaluator.square_to_lines.get((r, c), [])
        if not affected_lines:
            return 0

        opponent = 3 - player
        synergy_score = 0

        board = self.board
        lines = self.evaluator.lines
        patterns_items = list(PATTERNS_PLAYER.items())
        score_table = SCORE_TABLE

        if player == 1:
            trans_player = str.maketrans({"0": "0", "1": "1", "2": "2"})
            trans_opp = str.maketrans({"0": "0", "1": "2", "2": "1"})
            char_player = "1"
            char_opp = "2"
        else:  # player == 2
            trans_player = str.maketrans({"0": "0", "1": "2", "2": "1"})
            trans_opp = str.maketrans({"0": "0", "1": "1", "2": "2"})
            char_player = "2"
            char_opp = "1"

        offensive_cnt = 0
        defensive_cnt = 0
        defensive_sum_opp = 0.0
        offensive_sum_mine = 0.0
        for line_id in affected_lines:
            squares = lines[line_id]

            base = []
            idx = -1
            for i, (rr, cc) in enumerate(squares):
                if rr == r and cc == c:
                    idx = i
                v = board[rr, cc]
                base.append("0" if v == 0 else ("1" if v == 1 else "2"))
            if idx == -1:
                continue
            # My view
            original_stone = base[idx]
            base[idx] = char_player
            s_player_pc = "".join(base).translate(trans_player)

            for name, regex in patterns_items:
                if regex.search(s_player_pc):
                    offensive_cnt += 1
                    offensive_sum_mine += score_table[name]["mine"]
            # Opponent's view
            base[idx] = char_opp
            s_opp_pc = "".join(base).translate(trans_opp)

            for name, regex in patterns_items:
                if regex.search(s_opp_pc):
                    defensive_cnt += 1
                    defensive_sum_opp += score_table[name]["opp"]

            base[idx] = original_stone

        if offensive_cnt >= 2:
            synergy_score += offensive_sum_mine * 0.3
        if offensive_cnt >= 1 and defensive_cnt >= 1:
            synergy_score += (offensive_sum_mine + defensive_sum_opp) * 0.5
        if defensive_cnt >= 2:
            synergy_score += defensive_sum_opp * 0.3
        return synergy_score

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
        if not moves:
            return []

        # Categorize moves by threat type
        vcf_threats = []
        vct_threats = []
        defensive_vcf = []
        defensive_vct = []
        regular_moves = []

        opponent = 3 - player
        for move in moves:
            r, c = move
            is_vcf_threat = False
            is_vct_threat = False
            is_defensive = False

            # Check if move creates VCF threat
            self.board[r, c] = player
            if self.threat_detector._check_four_threat(self.board, r, c, player):
                vcf_threats.append(move)
                is_vcf_threat = True
            elif self.threat_detector._check_live_three(self.board, r, c, player):
                vct_threats.append(move)
                is_vct_threat = True
            self.board[r, c] = EMPTY

            # Check if move blocks opponent's threat
            if not is_vcf_threat and not is_vct_threat:
                self.board[r, c] = opponent
                if self.threat_detector._check_four_threat(self.board, r, c, opponent):
                    defensive_vcf.append(move)
                    is_defensive = True
                elif self.threat_detector._check_live_three(self.board, r, c, opponent):
                    defensive_vct.append(move)
                    is_defensive = True
                self.board[r, c] = EMPTY

            if not is_vcf_threat and not is_vct_threat and not is_defensive:
                regular_moves.append(move)

        # Order moves by priority
        ordered_moves = []

        # 1. Hash move (from transposition table)
        if hash_move and hash_move in moves:
            ordered_moves.append(hash_move)
            moves.remove(hash_move)
        # 2. VCF offensive threats
        ordered_moves.extend(vcf_threats)
        # 3. VCF defensive moves
        ordered_moves.extend(defensive_vcf)
        # 4. VCT offensive threats
        ordered_moves.extend(vct_threats)
        # 5. VCT defensive moves
        ordered_moves.extend(defensive_vct)
        # 6. Killer moves
        if depth < len(self.killer_moves):
            for killer in self.killer_moves[depth]:
                if killer and killer in regular_moves:
                    ordered_moves.append(killer)
                    regular_moves.remove(killer)
        # 7. Regular moves sorted by evaluation
        if regular_moves:
            move_scores = []
            for move in regular_moves:
                r, c = move
                self.board[r, c] = player
                self.evaluator.update_score(self.board, r, c, player)
                score = self.evaluator.get_current_score(
                    player
                ) + self._calculate_synergy_at(r, c, player)
                self.board[r, c] = EMPTY
                self.evaluator.update_score(self.board, r, c, EMPTY)
                move_scores.append((move, score))

            move_scores.sort(key=lambda x: x[1], reverse=True)
            ordered_moves.extend([m[0] for m in move_scores])

        # Apply depth-based pruning
        if (
            depth > 0
            and len(ordered_moves)
            > TOP_K_BY_DEPTH[min(depth - 1, len(TOP_K_BY_DEPTH) - 1)]
        ):
            ordered_moves = ordered_moves[
                : TOP_K_BY_DEPTH[min(depth - 1, len(TOP_K_BY_DEPTH) - 1)]
            ]

        return ordered_moves

    def negamax_with_vcf_vct(self, depth, alpha, beta, player, banned_moves_enabled):
        """
        Enhanced negamax with VCF/VCT integration
        VCF: Victory by Continuous Four
        VCT: Victory by Continuous Three
        """
        self._check_timeout()

        # Check VCF at higher depths
        if depth > 2 and self.current_search_depth > 3:
            self.vcf_checks += 1
            vcf_result = self.vcf_searcher.search(
                self.board, player, min(MAX_VCF_DEPTH, depth)
            )
            if vcf_result.is_winning:
                self.vcf_wins_found += 1
                logger.info(
                    f"[Turn {self.current_turn}] VCF win found at depth {depth}"
                )
                return (
                    SCORE_TABLE["FIVE"]["mine"] - vcf_result.depth,
                    vcf_result.winning_move,
                )

        # Check VCT at even higher depths
        if depth > 4 and self.current_search_depth > 5:
            self.vct_checks += 1
            vct_result = self.vct_searcher.search(
                self.board, player, min(MAX_VCT_DEPTH, depth // 2)
            )
            if vct_result.is_winning:
                self.vct_wins_found += 1
                logger.info(
                    f"[Turn {self.current_turn}] VCT win found at depth {depth}"
                )
                return (
                    SCORE_TABLE["FIVE"]["mine"] - vct_result.depth - VCT_PENALTY,
                    vct_result.winning_move,
                )

        # Standard negamax
        return self.negamax(depth, alpha, beta, player, banned_moves_enabled)

    def negamax(
        self, depth, alpha, beta, player, banned_moves_enabled, current_hash=None
    ):
        self._check_timeout()
        if depth <= 0:
            return self.evaluator.get_current_score(player), None
        if current_hash is None:
            current_hash = self._compute_hash(player)

        tt_entry = self.transposition_table.get(current_hash)
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

        original_alpha = alpha
        best_move, max_score = None, -float("inf")
        hash_move = tt_entry.get("move") if tt_entry else None
        moves = self.get_possible_moves(player, banned_moves_enabled, depth, hash_move)
        if not moves:
            return 0, None

        for i, move in enumerate(moves):
            self._check_timeout()

            r, c = move
            original_total_score = self.evaluator.get_current_score(player)
            original_line_values = {
                line_id: self.evaluator.line_values[line_id]
                for line_id in self.evaluator.square_to_lines.get((r, c), [])
            }  # store original affected line values

            # try to move and search
            new_hash = self._push_hash_state(r, c, player, current_hash)
            self.evaluator.update_score(self.board, r, c, player)
            self.board[r, c] = player
            if self._check_win_by_move(r, c, player):
                self._pop_hash_state()
                self.board[r, c] = EMPTY
                # self.evaluator.total_score = original_total_score
                # self.evaluator.line_values.update(original_line_values)
                return SCORE_TABLE["FIVE"]["mine"] - depth, (r, c)

            # The core PVS logic
            if i == 0:
                # 1. Principal Variation) - full search
                score, _ = self.negamax(
                    depth - 1, -beta, -alpha, 3 - player, banned_moves_enabled, new_hash
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
                if alpha < score and score < beta:
                    score, _ = self.negamax(
                        depth - 1,
                        -float("inf"),
                        float("inf"),
                        3 - player,
                        banned_moves_enabled,
                        new_hash,
                    )
                    score = -score

            # 3. Undo the move and restore the score state EXACTLY
            self._pop_hash_state()
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
            alpha = max(alpha, score)
            if alpha >= beta:
                if depth < len(self.killer_moves):
                    if move != self.killer_moves[depth][0]:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                break

        # Transposition table saving
        flag = "EXACT"
        if max_score <= original_alpha:
            flag = "UPPERBOUND"
        elif max_score >= beta:
            flag = "LOWERBOUND"
        # existing_entry = self.transposition_table.get(board_hash)
        # if not existing_entry or depth >= existing_entry["depth"]:
        self.transposition_table[current_hash] = {
            "score": max_score,
            "depth": depth,
            "flag": flag,
            "move": best_move,
            "age": self.search_generation,
            # "sorted_moves": moves,
        }

        return max_score, best_move

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

    def find_best_move(self, board_state, player, banned_moves_enabled):
        self.board = np.array(board_state)
        self.start_time = time.time()
        self.search_generation += 1
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
        self.panic_mode = False

        best_move_so_far = None
        final_search_depth = 0

        self.vcf_checks = 0
        self.vct_checks = 0
        self.vcf_wins_found = 0
        self.vct_wins_found = 0

        # 1st Priority: Check for VCF win at root
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

        # 2nd Priority: Check for VCF win at root
        # My View
        vcf_result = self.vcf_searcher.search(self.board, player, MAX_VCF_DEPTH)
        if vcf_result.is_winning:
            logger.info(
                f"[Turn {self.current_turn}] VCF win found! Sequence: {vcf_result.threat_sequence}"
            )
            return vcf_result.winning_move, 0
        # Opponent's View
        opponent = 3 - player
        opp_vcf = self.vcf_searcher.search(self.board, opponent, MAX_VCF_DEPTH)
        if opp_vcf.is_winning:
            logger.info(f"[Turn {self.current_turn}] Opponent has VCF! Must defend")
            defensive_moves = self.threat_detector.get_defensive_moves(
                self.board, opp_vcf.winning_move, opponent, max_defenses=3
            )
            if defensive_moves:
                return defensive_moves[0], 0

        # Normal search: IDDFS - Iterative Deepening Depth-First Search
        for depth in range(1, MAX_DEPTH + 1):
            try:
                self.current_search_depth = depth
                logger.info(f"--- Starting search at depth {depth} ---")

                score, move = self.negamax_with_vcf_vct(
                    depth, -float("inf"), float("inf"), player, banned_moves_enabled
                )
                if move:
                    best_move_so_far = move
                final_search_depth = depth

                elapsed_time = time.time() - self.start_time
                logger.info(
                    f"[Turn {self.current_turn}] Depth {depth} finished in {elapsed_time:.2f}s. Best move: {move}, Score: {score}"
                )

                # Panic mode
                if score < -SCORE_TABLE["LIVE_FOUR"]["opp"] and not self.panic_mode:
                    self.panic_mode = True
                    self.max_search_time = min(
                        TIME_LIMIT - 0.5, self.max_search_time * 1.5
                    )
                    logger.info("PANIC MODE: Extending search time")

                # Found forced win
                if score >= SCORE_TABLE["FIVE"]["mine"] - MAX_DEPTH:
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

        if not best_move_so_far and candidate_moves:
            best_move_so_far = candidate_moves[0]
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
        self.panic_mode = False
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
        best_move, search_depth = agent.find_best_move(
            board, color_to_play, banned_moves_enabled
        )
        logger.info(f"Move found: {best_move} at depth {search_depth}")
        return jsonify(
            {
                "move": [int(best_move[0]), int(best_move[1])],
                "search_depth": search_depth,
            }
        )
    except Exception:
        logger.exception("An unhandled error occurred in the get_move endpoint.")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    app.run(port=5003, debug=False)
