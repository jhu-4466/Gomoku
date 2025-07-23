# -*- coding: utf-8 -*-

import time
import json
import re
from collections import defaultdict
import numpy as np
from flask import Flask, request, jsonify


# --- Configuration ---
GRID_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
# EXPERT COMMENT: Adjusted TIME_LIMIT to be slightly less than typical HTTP timeouts.
TIME_LIMIT = 29.5  # Time limit for the AI to make a move, in seconds.
MAX_DEPTH = 50  # Max search depth for IDDFS
MIN_DEPTH = 3  # The minimum depth the AI must complete, regardless of time.

"""
The values for opponent's threats ('opp') are now significantly higher than 'mine'
to force the AI to block critical threats instead of making risky offensive moves.
"""
SCORE_TABLE = {
    "FIVE": {"mine": 100_000_000, "opp": 100_000_000},
    "LIVE_FOUR": {"mine": 10_000_000, "opp": 50_000_000},
    "RUSH_FOUR": {"mine": 1_000_000, "opp": 2_000_000},
    "DOUBLE_THREE": {"mine": 1_000_000, "opp": 10_000_000},
    "LIVE_THREE": {"mine": 500_000, "opp": 1_500_000},
    "SLEEPY_THREE": {"mine": 5_000, "opp": 10_000},
    "LIVE_TWO": {"mine": 1_000, "opp": 2_000},
    "SLEEPY_TWO": {"mine": 100, "opp": 200},
    "SINGLE": {"mine": 10, "opp": 20},
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
zobrist_table = np.random.randint(
    1, 2**63 - 1, (GRID_SIZE, GRID_SIZE, 3), dtype=np.uint64
)
zobrist_player_turn = np.random.randint(1, 2**63 - 1, dtype=np.uint64)

app = Flask(__name__)


# Custom exception for reliable timeout handling.
class TimeoutException(Exception):
    pass


class NegamaxAgent:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.transposition_table = {}
        self.start_time = 0
        self.board = np.zeros((board_size, board_size), dtype=int)

        self.killer_moves = [[None, None] for _ in range(MAX_DEPTH + 1)]
        self.history_heuristic = defaultdict(int)
        self.search_generation = 0
        self.last_depth_start_time = 0

        self.current_search_depth = 0
        self.nodes_evaluated_at_root = 0
        self.total_nodes_at_root = 0

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
        if self.board[r, c] != EMPTY:
            return False, ""

        self.board[r, c] = player

        fours, threes, overline = 0, 0, False
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
                overline = True
                break

            threes_in_line, fours_in_line = self._count_patterns_at(
                r, c, dr, dc, player
            )
            threes += threes_in_line
            fours += fours_in_line

        self.board[r, c] = EMPTY
        if overline:
            return True, "Overline"
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

    def evaluate_board(self, player_to_move):
        my_patterns = self._find_patterns_fast(player_to_move)
        op_patterns = self._find_patterns_fast(3 - player_to_move)

        my_score = sum(SCORE_TABLE[p]["mine"] * c for p, c in my_patterns.items())
        op_score = sum(SCORE_TABLE[p]["opp"] * c for p, c in op_patterns.items())

        if my_patterns.get("LIVE_THREE", 0) >= 2:
            my_score += SCORE_TABLE["DOUBLE_THREE"]["mine"]
        if op_patterns.get("LIVE_THREE", 0) >= 2:
            op_score += SCORE_TABLE["DOUBLE_THREE"]["opp"]

        return my_score - op_score

    def _find_patterns_fast(self, player):
        patterns = defaultdict(int)
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
        NEW: A lightweight function to statically evaluate the threat
        of a single move for sorting purposes. It is much faster than
        a full board evaluation.
        """
        score = 0
        opponent = 3 - player

        # Offensive score
        self.board[r, c] = player
        my_patterns = self._find_patterns_fast(player)
        score += my_patterns.get("LIVE_FOUR", 0) * 100000
        score += my_patterns.get("RUSH_FOUR", 0) * 10000
        score += my_patterns.get("LIVE_THREE", 0) * 5000
        self.board[r, c] = EMPTY

        # Defensive score
        self.board[r, c] = opponent
        op_patterns = self._find_patterns_fast(opponent)
        score += op_patterns.get("LIVE_FOUR", 0) * 75000
        score += op_patterns.get("RUSH_FOUR", 0) * 7500
        score += op_patterns.get("LIVE_THREE", 0) * 2500
        self.board[r, c] = EMPTY

        return score

    def get_possible_moves(self, player, banned_moves_enabled, depth, hash_move):
        """
        Uses a sophisticated, multi-layered sorting approach,
        including static threat analysis, to achieve
        Threat Space Search.
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

        # --- Threat-based move ordering ---
        move_scores = {
            m: self._rate_move_statically(m[0], m[1], player)
            + self.history_heuristic.get(m, 0)
            for m in moves
        }

        sorted_moves = sorted(moves, key=lambda m: move_scores.get(m, 0), reverse=True)

        # --- Final prioritized list construction ---
        final_ordered_list = []
        # 1. Hash Move from Transposition Table
        if hash_move and hash_move in moves:
            final_ordered_list.append(hash_move)
        # 2. Urgent: Opponent's winning moves
        for move in opponent_win_moves:
            if move not in final_ordered_list:
                final_ordered_list.append(move)
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
            return self.quiescence_search(alpha, beta, player, 2), None

        if depth >= 3 and np.any(self.board):
            score, _ = self.negamax(
                depth - 3, -beta, -beta + 1, 3 - player, banned_moves_enabled
            )
            score = -score
            if score >= beta:
                return beta, None

        best_move, max_score = None, -float("inf")
        hash_move = tt_entry.get("move") if tt_entry else None
        moves = self.get_possible_moves(player, banned_moves_enabled, depth, hash_move)

        if depth == self.current_search_depth:
            self.total_nodes_at_root = len(moves)
        if not moves:
            return 0, None

        for i, move in enumerate(moves):
            r, c = move
            if depth == self.current_search_depth:
                self.nodes_evaluated_at_root = i + 1

            self.board[r, c] = player

            if self._check_win_by_move(r, c, player):
                score = SCORE_TABLE["FIVE"]["mine"] - (MAX_DEPTH - depth)
            else:
                reduction = 0
                extension = 0

                if move == hash_move:
                    extension = 1

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

            self.board[r, c] = EMPTY

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

        flag = "EXACT"
        if max_score <= original_alpha:
            flag = "UPPERBOUND"
        elif max_score >= beta:
            flag = "LOWERBOUND"

        self.transposition_table[board_hash] = {
            "score": max_score,
            "depth": depth,
            "flag": flag,
            "move": best_move,
            "age": self.search_generation,
        }
        return max_score, best_move

    # --- Quiescence Search and other helpers remain unchanged ---
    def quiescence_search(self, alpha, beta, player, q_depth):
        if time.time() - self.start_time > TIME_LIMIT:
            raise TimeoutException()

        if q_depth == 0:
            return self.evaluate_board(player)

        stand_pat_score = self.evaluate_board(player)
        if stand_pat_score >= beta:
            return beta
        alpha = max(alpha, stand_pat_score)

        moves = self._get_qsearch_moves(player)

        for move in moves:
            r, c = move
            self.board[r, c] = player
            score = -self.quiescence_search(-beta, -alpha, 3 - player, q_depth - 1)
            self.board[r, c] = EMPTY
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

        for depth in range(1, MAX_DEPTH + 1):
            try:
                self.current_search_depth = depth
                print(f"--- Starting search at depth {depth} ---")

                score, move = self.negamax(
                    depth, -float("inf"), float("inf"), player, banned_moves_enabled
                )

                best_move_so_far = move
                elapsed_time = time.time() - self.start_time
                print(
                    f"Depth {depth} finished in {elapsed_time:.2f}s. Best move: {move}, Score: {score}"
                )

                if abs(score) >= SCORE_TABLE["FIVE"]["mine"] - MAX_DEPTH:
                    print("Terminal sequence found. Halting search.")
                    break

            except TimeoutException:
                print(f"Timeout! Search at depth {depth} was forcefully interrupted.")
                print(f"Returning best move from last completed depth ({depth-1}).")
                break

        if not best_move_so_far:
            print(
                "Search failed or timed out at depth 1. Falling back to first possible move."
            )
            possible_moves = self.get_possible_moves(
                player, banned_moves_enabled, 0, None
            )
            if possible_moves:
                best_move_so_far = possible_moves[0]

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
            moves = agent.get_possible_moves(
                color_to_play, banned_moves_enabled, 0, None
            )
            return (
                jsonify({"move": [int(moves[0][0]), int(moves[0][1])]})
                if moves
                else jsonify({"move": None})
            )
    else:
        moves = agent.get_possible_moves(1, False, 0, None)
        return (
            jsonify({"move": [int(moves[0][0]), int(moves[0][1])]})
            if moves
            else jsonify({"move": None})
        )


if __name__ == "__main__":
    app.run(port=5003, debug=False)
