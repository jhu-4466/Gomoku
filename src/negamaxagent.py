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
TOP_K_BY_DEPTH = [24, 22, 20, 18, 16]
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
    "LIVE_FOUR": {"mine": 120_000, "opp": 1_000_000},
    "LIVE_THREE": {"mine": 80_000, "opp": 150_000},
    "RUSH_FOUR": {"mine": 10_000, "opp": 50_000},
    "SLEEPY_THREE": {"mine": 5_000, "opp": 7_000},
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
    "LIVE_TWO": re.compile(r"0110|01010"),
    "SLEEPY_TWO": re.compile(r"2110|0112|21010|01012"),
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

    def _get_candidate_points(self, board):
        if not np.any(board):
            center = self.board_size // 2
            return {(center, center)}

        candidate_points = set()
        occupied_stones = np.argwhere(board != EMPTY)
        for r, c in occupied_stones:
            for dr, dc in MOVE_DIRECTIONS:
                for i in range(1, 3):
                    nr, nc = r + i * dr, c + i * dc
                    if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                        break
                    if board[nr, nc] == EMPTY:
                        candidate_points.add((nr, nc))
                    else:
                        break

        return sorted(candidate_points)

    def find_four_threats(self, board, player, current_hash, max_threats=10):
        """Find all positions that create four-in-a-row threats (live-four patterns)"""
        threats = []

        candidate_points = self._get_candidate_points(board)
        for r, c in candidate_points:
            if len(threats) >= max_threats:
                break

            cached = self.threat_cache.get(current_hash, (r, c))
            if cached is not None:
                if cached:
                    threats.append((r, c))
                continue

            board[r, c] = player
            is_threat = self._check_four_threat(board, r, c, player)
            board[r, c] = EMPTY

            self.threat_cache.put(current_hash, (r, c), is_threat)
            if is_threat:
                threats.append((r, c))

        return threats

    def _count_threats_at(self, board, r, c, player):
        board[r, c] = player

        live_threes = 0
        rush_fours = 0
        live_fours = 0
        live_twos = 0
        sleepy_threes = 0

        for dr, dc in MOVE_DIRECTIONS:
            line_str = ""
            for i in range(-5, 6):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    line_str += str(board[nr, nc])
                else:
                    line_str += "3"

            if player == WHITE:
                line_str = (
                    line_str.replace("1", "X").replace("2", "1").replace("X", "2")
                )

            if PATTERNS_PLAYER["LIVE_THREE"].search(line_str):
                live_threes += 1
            if PATTERNS_PLAYER["RUSH_FOUR"].search(line_str):
                rush_fours += 1
            if PATTERNS_PLAYER["LIVE_FOUR"].search(line_str):
                live_fours += 1
            if PATTERNS_PLAYER["LIVE_TWO"].search(line_str):
                live_twos += 1
            if PATTERNS_PLAYER["SLEEPY_THREE"].search(line_str):
                sleepy_threes += 1

        board[r, c] = EMPTY
        return live_threes, rush_fours, live_fours, live_twos, sleepy_threes

    def find_vct_threats(self, board, player, max_threats=15):
        threats_by_type = {
            "CRITICAL": [],  # R4+R4 or L3+R4 or L3+L3 (100% win)
            "MAJOR": [],  # Single L3
        }

        candidate_points = self._get_candidate_points(board)
        for r, c in candidate_points:
            live_threes, rush_fours, live_fours, live_twos, sleepy_threes = (
                self._count_threats_at(board, r, c, player)
            )

            if (
                rush_fours >= 2
                or (rush_fours >= 1 and live_threes >= 1)
                or live_threes >= 2
                or live_fours >= 1
            ):
                threats_by_type["CRITICAL"].append((r, c))
                break
            elif live_threes >= 1:
                threats_by_type["MAJOR"].append((r, c))

        critical_threats = threats_by_type["CRITICAL"]
        major_threats = threats_by_type["MAJOR"]
        return critical_threats[:5], major_threats[:10]

    def _check_patterns_at(self, board, r, c, player, pattern_names_to_check):
        for dr, dc in MOVE_DIRECTIONS:
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
        four_threat_patterns = ["LIVE_FOUR"]
        return self._check_patterns_at(board, r, c, player, four_threat_patterns)

    # def _check_live_three(self, board, r, c, player):
    #     three_threat_patterns = ["LIVE_THREE"]
    #     return self._check_patterns_at(board, r, c, player, three_threat_patterns)


class VCFSearcher:
    """
    VCF: Victory by Continuous Four(Forcing moves)
    To check if there is a winning move routes by VCF.
    """

    def __init__(self, board_size=15, agent=None):
        self.board_size = board_size
        self.threat_detector = ThreatDetector(board_size)
        self.search_nodes = 0
        self.max_vcf_depth = 20

        self.agent = agent

        self.search_nodes = 0
        self.start_time = 0

    def search(self, board, player, max_depth=None, time_limit=None):
        if max_depth is None:
            max_depth = MAX_VCF_DEPTH
        if time_limit is None:
            time_limit = VCF_TIME_LIMIT

        self.search_nodes = 0
        self.start_time = time.time()
        self.threat_detector.threat_cache.clear()
        initial_hash = np.uint64(0)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != EMPTY:
                    initial_hash ^= zobrist_table[r, c, board[r, c]]
        result = self._vcf_recursive(board.copy(), player, 0, [], set(), initial_hash)
        result.nodes_searched = self.search_nodes
        result.time_elapsed = time.time() - self.start_time

        logger.info(
            f"[VCF] VCF search completed for player {player}: is_winning={result.is_winning}"
        )
        if result.is_winning:
            logger.info(
                f"[VCF] Player {player} has a winning move {result.winning_move}!"
            )
        return result

    def _get_defensive_replies(self, board, attack_move, attacker_player):
        r, c = attack_move
        defenses = set()

        for dr, dc in MOVE_DIRECTIONS:
            line_stones = [(r, c)]

            for i in range(1, 5):
                nr, nc = r + i * dr, c + i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and board[nr, nc] == attacker_player
                ):
                    break
                line_stones.append((nr, nc))

            for i in range(1, 5):
                nr, nc = r - i * dr, c - i * dc
                if not (
                    0 <= nr < self.board_size
                    and 0 <= nc < self.board_size
                    and board[nr, nc] == attacker_player
                ):
                    break
                line_stones.append((nr, nc))

            if len(line_stones) >= 4:
                min_r, min_c = min(line_stones)
                max_r, max_c = max(line_stones)

                p1_r, p1_c = min_r - dr, min_c - dc
                if (
                    0 <= p1_r < self.board_size
                    and 0 <= p1_c < self.board_size
                    and board[p1_r, p1_c] == EMPTY
                ):
                    defenses.add((p1_r, p1_c))

                p2_r, p2_c = max_r + dr, max_c + dc
                if (
                    0 <= p2_r < self.board_size
                    and 0 <= p2_c < self.board_size
                    and board[p2_r, p2_c] == EMPTY
                ):
                    defenses.add((p2_r, p2_c))

        return list(defenses)

    def _vcf_recursive(
        self,
        board,
        player,
        depth,
        threat_sequence,
        visited_states,
        current_hash,
    ):
        # Check limits
        self.search_nodes += 1
        if (
            depth >= MAX_VCF_DEPTH
            or self.search_nodes > VCF_NODE_LIMIT
            or time.time() - self.start_time > VCF_TIME_LIMIT
        ):
            return VCFResult(is_winning=False, depth=depth)
        if self.agent and hasattr(self.agent, "_check_timeout"):
            try:
                self.agent._check_timeout()
            except TimeoutException:
                return VCFResult(is_winning=False, depth=depth)

        if current_hash in visited_states:
            return VCFResult(is_winning=False, depth=depth)
        visited_states.add(current_hash)

        attacking_moves = self.threat_detector.find_four_threats(
            board, player, current_hash
        )
        if not attacking_moves:
            return VCFResult(False, depth)

        for attack in attacking_moves:
            board[attack[0], attack[1]] = player
            new_threat_sequence = threat_sequence + [attack]
            hash_after_attack = (
                current_hash ^ zobrist_table[attack[0], attack[1], player]
            )

            # if win directly
            if self._check_win(board, attack[0], attack[1], player):
                board[attack[0], attack[1]] = EMPTY
                return VCFResult(
                    is_winning=True,
                    depth=depth,
                    winning_move=attack,
                    threat_sequence=new_threat_sequence,
                )
            # Otherwise, search defensive pos.
            defensive_replies = self._get_defensive_replies(board, attack, player)
            if not defensive_replies:
                board[attack[0], attack[1]] = EMPTY
                return VCFResult(
                    is_winning=True,
                    depth=depth,
                    winning_move=attack,
                    threat_sequence=new_threat_sequence,
                )
            # Continuously attack and defense
            all_defenses_countered = True
            for defense in defensive_replies:
                board[defense[0], defense[1]] = 3 - player
                hash_after_defense = (
                    hash_after_attack
                    ^ zobrist_table[defense[0], defense[1], 3 - player]
                )
                result = self._vcf_recursive(
                    board,
                    player,
                    depth + 1,
                    new_threat_sequence,
                    visited_states.copy(),
                    hash_after_defense,
                )
                board[defense[0], defense[1]] = EMPTY
                if not result.is_winning:
                    all_defenses_countered = False
                    break
            if all_defenses_countered:
                board[attack[0], attack[1]] = EMPTY
                return VCFResult(
                    is_winning=True,
                    depth=depth,
                    winning_move=attack,
                    threat_sequence=new_threat_sequence,
                )

            # Revert
            board[attack[0], attack[1]] = EMPTY

        return VCFResult(is_winning=False, depth=depth)

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

    def _check_immediate_vct_win(self, board, player):
        candidate_moves = self.threat_detector._get_candidate_points(board)

        scored = []
        for r, c in candidate_moves:
            board[r, c] = player
            live_threes, rush_fours, live_fours, _, _ = (
                self.threat_detector._count_threats_at(board, r, c, player)
            )
            board[r, c] = EMPTY
            scored.append(((live_fours, rush_fours, live_threes), (r, c)))
        scored.sort(reverse=True, key=lambda x: x[0])
        for (live_fours, rush_fours, live_threes), (r, c) in scored[
            : min(20, len(scored))
        ]:
            is_vct_win = False
            if (
                live_fours >= 1
                or rush_fours >= 2
                or (rush_fours >= 1 and live_threes >= 1)
                or live_threes >= 2
            ):
                is_vct_win = True
            if is_vct_win:
                logger.info(
                    f"[Turn] VCT win found at ({r},{c}): L4={live_fours}, R4={rush_fours}, L3={live_threes}"
                )
                return (r, c)

        return None

    def search(self, board, player, max_depth=None, time_limit=None):
        """
        Search for VCT (Victory by Continuous Threat)
        Uses open threes and threat combinations
        """
        if max_depth is None:
            max_depth = MAX_VCT_DEPTH
        if time_limit is None:
            time_limit = VCT_TIME_LIMIT

        self.search_nodes = 0
        self.start_time = time.time()

        initial_hash = np.uint64(0)
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != EMPTY:
                    initial_hash ^= zobrist_table[r, c, board[r, c]]

        immediate_winning_move = self._check_immediate_vct_win(board, player)
        if immediate_winning_move:
            logger.info(
                f"[VCT] Immediate VCT win found at {immediate_winning_move[0], immediate_winning_move[1]}"
            )
            return VCTResult(
                is_winning=True,
                depth=0,
                winning_move=immediate_winning_move,
                threat_sequence=[immediate_winning_move],
                nodes_searched=1,
                time_elapsed=time.time() - self.start_time,
            )

        result = self._vct_recursive(board.copy(), player, 0, [], set(), initial_hash)
        result.nodes_searched = self.search_nodes
        result.time_elapsed = time.time() - self.start_time

        logger.info(
            f"[VCT] VCT search completed for player {player}: is_winning={result.is_winning}"
        )
        if result.is_winning:
            logger.info(f"[VCT] Winning move: {result.winning_move}")

        return result

    def _vct_recursive(
        self,
        board,
        player,
        depth,
        threat_sequence,
        visited_states,
        current_hash,
    ):
        """Recursive VCT search"""
        # Check limits
        self.search_nodes += 1
        if (
            depth >= MAX_VCT_DEPTH
            or self.search_nodes > self.max_nodes
            or time.time() - self.start_time > VCT_TIME_LIMIT
        ):
            return VCTResult(is_winning=False, depth=depth)

        if self.agent and hasattr(self.agent, "_check_timeout"):
            try:
                self.agent._check_timeout()
            except TimeoutException:
                return VCTResult(is_winning=False, depth=depth)

        if current_hash in visited_states:
            return VCTResult(is_winning=False, depth=depth)
        visited_states.add(current_hash)

        # Check if win directly
        immediate_win = self._check_immediate_vct_win(board, player)
        if immediate_win:
            return VCTResult(
                is_winning=True,
                depth=depth,
                winning_move=immediate_win,
                threat_sequence=threat_sequence + [immediate_win],
            )

        # # Transition to VCF, if vcf successfully, means that this round successes.
        # if depth >= 2:
        #     vcf_result = self.vcf_searcher.search(
        #         board, player, max_depth=MAX_VCF_DEPTH // 2
        #     )
        #     if vcf_result.is_winning:
        #         return VCTResult(
        #             is_winning=True,
        #             depth=depth + vcf_result.depth,
        #             winning_move=vcf_result.winning_move,
        #             threat_sequence=threat_sequence + vcf_result.threat_sequence,
        #         )

        critical_threats, major_threats = self.threat_detector.find_vct_threats(
            board, player
        )
        if critical_threats:
            winning_move = critical_threats[0]
            new_threat_sequence = threat_sequence + [winning_move]
            return VCTResult(
                is_winning=True,
                depth=depth,
                winning_move=winning_move,
                threat_sequence=new_threat_sequence,
            )
        if not major_threats:
            return VCTResult(is_winning=False, depth=depth)
        for attack in major_threats:
            r_att, c_att = attack
            board[r_att, c_att] = player
            hash_after_attack = current_hash ^ zobrist_table[r_att, c_att, player]
            new_threat_sequence = threat_sequence + [attack]

            defensive_replies = self._get_vct_defensive_replies(board, attack, player)
            if not defensive_replies:
                board[r_att, c_att] = EMPTY
                return VCTResult(
                    is_winning=True,
                    depth=depth,
                    winning_move=attack,
                    threat_sequence=new_threat_sequence,
                )
            all_defenses_countered = True
            for defense in defensive_replies:
                r_def, c_def = defense
                board[r_def, c_def] = 3 - player
                hash_after_defense = (
                    hash_after_attack ^ zobrist_table[r_def, c_def, 3 - player]
                )

                result = self._vct_recursive(
                    board,
                    player,
                    depth + 1,
                    new_threat_sequence,
                    visited_states.copy(),
                    hash_after_defense,
                )

                board[r_def, c_def] = EMPTY

                if not result.is_winning:
                    all_defenses_countered = False
                    break
            if all_defenses_countered:
                board[r_att, c_att] = EMPTY
                return VCTResult(
                    is_winning=True,
                    depth=depth,
                    winning_move=attack,
                    threat_sequence=new_threat_sequence,
                )

            board[r_att, c_att] = EMPTY

        return VCTResult(is_winning=False, depth=depth)

    def _get_vct_defensive_replies(self, board, attack_move, attacker_player):
        """
        Get defensive replies for a given threat move
        """
        if not attack_move or len(attack_move) != 2:
            return []

        r_att, c_att = attack_move
        if not (0 <= r_att < self.board_size and 0 <= c_att < self.board_size):
            return []

        defenses = set()

        board[attack_move[0], attack_move[1]] = attacker_player
        for dr, dc in MOVE_DIRECTIONS:
            line_str = ""
            center_idx = -1
            line_positions = []

            for i in range(-5, 6):
                if i == 0:
                    center_idx = len(line_str)

                nr, nc = attack_move[0] + i * dr, attack_move[1] + i * dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    piece_value = str(board[nr, nc])
                    line_str += piece_value
                    line_positions.append((nr, nc))
                else:
                    line_str += "3"
                    line_positions.append(None)

            original_line_str = line_str
            if attacker_player == WHITE:
                line_str = (
                    line_str.replace("1", "X").replace("2", "1").replace("X", "2")
                )

            threat_patterns_to_check = [
                ("LIVE_FOUR", PATTERNS_PLAYER["LIVE_FOUR"]),
                ("RUSH_FOUR", PATTERNS_PLAYER["RUSH_FOUR"]),
                ("LIVE_THREE", PATTERNS_PLAYER["LIVE_THREE"]),
            ]
            total_matches_found = 0
            for pattern_name, pattern_regex in threat_patterns_to_check:
                matches_found = 0
                for match in re.finditer(pattern_regex, line_str):
                    matches_found += 1
                    total_matches_found += 1
                    match_start = match.start()
                    match_end = match.end()
                    matched_pattern = match.group(0)

                    if not (match_start <= center_idx < match_end):
                        continue

                    defense_positions = self._find_defense_positions_for_pattern(
                        matched_pattern, match_start, line_positions, pattern_name
                    )

                    for pos in defense_positions:
                        defenses.add(pos)

        board[attack_move[0], attack_move[1]] = EMPTY

        valid_defenses = []
        for r, c in defenses:
            is_in_bounds = 0 <= r < self.board_size and 0 <= c < self.board_size
            is_empty = board[r, c] == EMPTY if is_in_bounds else False
            if is_in_bounds and is_empty:
                valid_defenses.append((int(r), int(c)))
        return valid_defenses

    def _find_defense_positions_for_pattern(
        self, matched_pattern, match_start_idx, line_positions, pattern_name
    ):
        defense_positions = []

        if pattern_name == "LIVE_FOUR":
            if matched_pattern == "011110":
                for i in [0, 5]:
                    if matched_pattern[i] == "0":
                        abs_idx = match_start_idx + i
                        if (
                            abs_idx < len(line_positions)
                            and line_positions[abs_idx] is not None
                        ):
                            defense_positions.append(line_positions[abs_idx])
        elif pattern_name == "RUSH_FOUR":
            for i, char in enumerate(matched_pattern):
                if char == "0":
                    abs_idx = match_start_idx + i
                    if (
                        abs_idx < len(line_positions)
                        and line_positions[abs_idx] is not None
                    ):
                        defense_positions.append(line_positions[abs_idx])
        elif pattern_name in ["LIVE_THREE"]:
            if matched_pattern == "01110":
                for i, char in enumerate(matched_pattern):
                    if char == "0":
                        abs_idx = match_start_idx + i
                        if (
                            abs_idx < len(line_positions)
                            and line_positions[abs_idx] is not None
                        ):
                            defense_positions.append(line_positions[abs_idx])
            elif matched_pattern == "011010" or matched_pattern == "010110":
                for i, char in enumerate(matched_pattern):
                    if i == 0:
                        continue
                    if char == "0":
                        abs_idx = match_start_idx + i
                        if (
                            abs_idx < len(line_positions)
                            and line_positions[abs_idx] is not None
                        ):
                            defense_positions.append(line_positions[abs_idx])
                        break

        return defense_positions


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

    def get_possible_moves(
        self, player, banned_moves_enabled, depth, hash_move, current_hash
    ):
        """
        Uses a sophisticated, multi-layered sorting approach,
        including static threat analysis, to achieve
        Threat Space Search.
        """
        moves = self._get_candidate_moves(player, banned_moves_enabled)
        if not moves:
            empty_cells = np.argwhere(self.board == EMPTY)
            return [tuple(cell) for cell in empty_cells]

        opponent = 3 - player
        my_win_moves = [m for m in moves if self._check_win_by_move(m[0], m[1], player)]
        if my_win_moves:
            return my_win_moves
        opponent_win_moves = [
            m for m in moves if self._check_win_by_move(m[0], m[1], opponent)
        ]
        if opponent_win_moves:
            return opponent_win_moves

        # --- Threat-based move ordering ---
        move_scores = []
        for move in moves:
            r, c = move
            self.board[r, c] = player
            self.evaluator.update_score(self.board, r, c, player)
            score = self.evaluator.get_current_score(
                player
            ) + self._calculate_synergy_at(r, c, player)
            self.board[r, c] = EMPTY
            self.evaluator.update_score(self.board, r, c, EMPTY)  # Revert score
            move_scores.append((move, score))
        sorted_moves = sorted(move_scores, key=lambda x: x[1], reverse=True)

        # --- Final prioritized list construction ---
        final_ordered_list = []
        # 1. Hash Move from Transposition Table
        if hash_move and hash_move in moves:
            final_ordered_list.append(hash_move)
        # 2. Killer Moves
        killers = self.killer_moves[depth]
        if killers[0] and killers[0] in moves and killers[0] not in final_ordered_list:
            final_ordered_list.append(killers[0])
        if killers[1] and killers[1] in moves and killers[1] not in final_ordered_list:
            final_ordered_list.append(killers[1])
        # 3. All other moves
        for move_tuple in sorted_moves:
            move = move_tuple[0]
            if move not in final_ordered_list:
                final_ordered_list.append(move)
        # 4. Top-K Move Pruning to reduce the branching factor.
        if depth > 0:
            absolute_depth = self.current_search_depth - depth
            if absolute_depth < 0:
                absolute_depth = 0  # Safeguard

            if absolute_depth < len(TOP_K_BY_DEPTH):
                top_k = TOP_K_BY_DEPTH[absolute_depth]
            else:
                top_k = TOP_K_BY_DEPTH[
                    -1
                ]  # Use the last value as a default for deeper nodes.

            if len(final_ordered_list) > top_k:
                return final_ordered_list[:top_k]

        return final_ordered_list

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
        moves = self.get_possible_moves(
            player, banned_moves_enabled, depth, hash_move, current_hash
        )
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
                    score += 1_000_000  # Reward faster wins
                else:
                    score -= 1_000_000  # Penalize faster losses

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

        return list(moves)

    def _blocks_threats(self, opponent_player, banned_moves_enabled, is_extra=False):
        block_moves_4 = set()
        block_moves_3 = set()

        candidate_moves = self._get_candidate_moves(
            opponent_player, banned_moves_enabled
        )
        for r, c in candidate_moves:
            self.board[r, c] = opponent_player

            live_fours = 0
            live_threes = 0
            live_twos = 0
            rush_fours = 0

            for dr, dc in MOVE_DIRECTIONS:
                line_str_list = []
                for i in range(-7, 8):
                    nr, nc = r + i * dr, c + i * dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        line_str_list.append(str(self.board[nr, nc]))
                    else:
                        line_str_list.append("3")

                line_str = "".join(line_str_list)
                if opponent_player == WHITE:
                    normalized_line = (
                        line_str.replace("1", "X").replace("2", "1").replace("X", "2")
                    )
                else:
                    normalized_line = line_str

                if PATTERNS_PLAYER["LIVE_FOUR"].search(normalized_line):
                    logger.debug(
                        f"Found LIVE_FOUR threat at {(r, c)} in direction {(dr, dc)} with line {normalized_line}"
                    )
                    block_moves_4.add((r, c))
                elif PATTERNS_PLAYER["LIVE_THREE"].search(normalized_line):
                    logger.debug(
                        f"Found LIVE_THREE threat at {(r, c)} in direction {(dr, dc)} with line {normalized_line}"
                    )
                    if not is_extra:
                        block_moves_3.add((r, c))
                        continue
                    live_threes += 1
                elif (
                    PATTERNS_PLAYER["RUSH_FOUR"].search(normalized_line)
                    and not is_extra
                ):
                    rush_fours += 1
                    logger.debug(
                        f"Found RUSH_FOUR threat at {(r, c)} in direction {(dr, dc)} with line {normalized_line}, num = {rush_fours}"
                    )
                    if rush_fours >= 2:
                        self.board[r, c] = EMPTY
                        return [(r, c)]  # Immediate block needed
                # if PATTERNS_PLAYER["LIVE_TWO"].search(normalized_line):
                #     live_twos += 1

                if live_threes >= 1 and rush_fours >= 1:
                    logger.debug(
                        f"Found LIVE_THREE and RUSH_FOUR threat at {(r, c)} in direction {(dr, dc)} with line {normalized_line}"
                    )
                    self.board[r, c] = EMPTY
                    return [(r, c)]
                if live_threes >= 2:
                    logger.debug(
                        f"Found double LIVE_THREE threat at {(r, c)} in direction {(dr, dc)} with line {normalized_line}"
                    )
                    self.board[r, c] = EMPTY
                    return [(r, c)]

            self.board[r, c] = EMPTY

        logger.info(
            f"[Turn {self.current_turn}] Candidate Block Moves: {block_moves_4, block_moves_3}."
        )
        return list(block_moves_4) if block_moves_4 else list(block_moves_3)

    def find_best_move(self, board_state, player, banned_moves_enabled):
        self.board = np.array(board_state)
        self.start_time = time.time()
        self.search_generation += 1
        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]

        best_move_so_far = None
        final_search_depth = 0
        current_hash = self._compute_hash(player)

        # 1st Priority: Check for immediate win, have to win if you can win
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

        # VCF/VCT Check at root only
        logger.info(f"--- Starting VCF/VCT check ---")
        # Opponent's VCF
        opp_vcf = self.vcf_searcher.search(self.board, opponent)
        if opp_vcf.is_winning:
            logger.info(
                f"[Turn {self.current_turn}] Opponent's VCF win detected. Finding defense."
            )
            defensive_moves = self.vcf_searcher._get_defensive_replies(
                self.board, opp_vcf.winning_move, opponent
            )
            if defensive_moves:
                best_defense_move = None
                best_defense_score = -float("inf")
                for move in defensive_moves:
                    r, c = move

                    self.board[r, c] = opponent
                    self.evaluator.update_score(self.board, r, c, opponent)
                    score = self.evaluator.get_current_score(
                        opponent
                    ) + self._calculate_synergy_at(r, c, opponent)
                    self.board[r, c] = EMPTY
                    self.evaluator.update_score(self.board, r, c, EMPTY)

                    if score > best_defense_score:
                        best_defense_score = score
                        best_defense_move = move

                best_move_so_far = best_defense_move
                final_search_depth = 0
                logger.info(
                    f"[Turn {self.current_turn}] VCF - Found necessary defense {best_move_so_far[0], best_move_so_far[1]} that score is {best_defense_score}. Halting."
                )
                return best_move_so_far, final_search_depth
            else:
                logger.warning(
                    f"[Turn {self.current_turn}] Opponent has an unstoppable VCF."
                )
        # My VCF
        my_vcf = self.vcf_searcher.search(self.board, player, MAX_VCF_DEPTH)
        if my_vcf.is_winning:
            logger.info(
                f"[Turn {self.current_turn}] My VCF win found at root during depth {0}. Halting search."
            )
            best_move_so_far = my_vcf.winning_move
            final_search_depth = 0
            return best_move_so_far, final_search_depth
        # 3. Opponent's VCT
        opp_vct = self.vct_searcher.search(self.board, opponent, MAX_VCT_DEPTH)
        if opp_vct.is_winning:
            logger.info(
                f"[Turn {self.current_turn}] Opponent's VCT win detected. Finding defense."
            )
            defensive_moves = self._blocks_threats(
                opponent, banned_moves_enabled, is_extra=False
            )
            if defensive_moves:
                best_defense_move = None
                best_defense_score = -float("inf")
                for move in defensive_moves:
                    r, c = move

                    self.board[r, c] = opponent
                    self.evaluator.update_score(self.board, r, c, opponent)
                    score = self.evaluator.get_current_score(
                        opponent
                    ) + self._calculate_synergy_at(r, c, opponent)
                    self.board[r, c] = EMPTY
                    self.evaluator.update_score(self.board, r, c, EMPTY)

                    if score > best_defense_score:
                        best_defense_score = score
                        best_defense_move = move

                best_move_so_far = best_defense_move
                final_search_depth = 0
                logger.info(
                    f"[Turn {self.current_turn}] VCT - Found necessary defense {best_move_so_far[0], best_move_so_far[1]} that score is {best_defense_score}. Halting."
                )
                return best_move_so_far, final_search_depth
            else:
                logger.warning(
                    f"[Turn {self.current_turn}] No defensive moves found for opponent's VCT!"
                )
                logger.info(
                    "[DEBUG] VCT defense detection failed, continuing to normal search..."
                )
        # Extra defense for single LIVE_THREE
        block_moves = self._blocks_threats(
            opponent, banned_moves_enabled, is_extra=True
        )
        if block_moves:
            logger.info(
                f"[Turn {self.current_turn}] Extra defense: Blocking opponent's potential threats at {block_moves}. Skipping search."
            )
            best_defense_move = None
            best_defense_score = -float("inf")
            for move in block_moves:
                r, c = move

                self.board[r, c] = opponent
                self.evaluator.update_score(self.board, r, c, opponent)
                score = self.evaluator.get_current_score(
                    opponent
                ) + self._calculate_synergy_at(r, c, opponent)
                self.board[r, c] = EMPTY
                self.evaluator.update_score(self.board, r, c, EMPTY)

                if score > best_defense_score:
                    best_defense_score = score
                    best_defense_move = move

            best_move_so_far = best_defense_move
            final_search_depth = 0
            return best_move_so_far, final_search_depth
        # 4. My VCT
        my_vct = self.vct_searcher.search(self.board, player, MAX_VCT_DEPTH)
        if my_vct.is_winning:
            logger.info(
                f"[Turn {self.current_turn}] My VCT win found at root. Halting search."
            )
            best_move_so_far = my_vct.winning_move
            final_search_depth = 0
            return best_move_so_far, final_search_depth

        # Normal search: IDDFS - Iterative Deepening Depth-First Search
        for depth in range(1, MAX_DEPTH + 1):
            try:
                self.current_search_depth = depth
                # Negamax search
                logger.info(f"--- Starting search at depth {depth} ---")
                score, move = self.negamax(
                    depth, -float("inf"), float("inf"), player, banned_moves_enabled
                )
                if move:
                    best_move_so_far = move
                final_search_depth = depth

                elapsed_time = time.time() - self.start_time
                logger.info(
                    f"[Turn {self.current_turn}] Depth {depth} finished in {elapsed_time:.2f}s. Best move: {move[0], move[1]}, Score: {score}"
                )

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
                f"Normal Phase: Opening book move found: {opening_move[0], opening_move[1]}. Returning immediately."
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
        logger.info(f"Move found: {best_move[0], best_move[1]} at depth {search_depth}")
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
