from flask import Flask, request, jsonify
import random

app = Flask(__name__)
GRID_SIZE = 15


# GamePhase class is now defined globally to be accessible by the Flask app logic.
class GamePhase:
    NORMAL = 0
    SWAP2_P1_PLACE_3 = 1
    SWAP2_P2_CHOOSE_ACTION = 2  # Note: App sends "P2_CHOOSE" string for this
    SWAP2_P2_PLACE_2 = 3
    SWAP2_P1_CHOOSE_COLOR = 4  # Note: App sends "P1_CHOOSE" string for this


# A simple opening book for Swap2 P1's first 3 moves.
OPENING_BOOK = [
    [(7, 7), (7, 8), (8, 7)],  # Central pattern
    [(7, 7), (8, 8), (6, 6)],  # Diagonal pattern
    [(7, 7), (7, 6), (6, 8)],  # Another solid central pattern
]


# --- Banned Move & Evaluation Logic ---
def _get_char(board, r, c):
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        return board[r][c]
    return -1


def _check_overline(board, r, c, p):
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        count = 1
        for i in range(1, 6):
            if _get_char(board, r + i * dr, c + i * dc) == p:
                count += 1
            else:
                break
        for i in range(1, 6):
            if _get_char(board, r - i * dr, c - i * dc) == p:
                count += 1
            else:
                break
        if count > 5:
            return True
    return False


def _is_banned_move(board, r, c):
    if board[r][c] != 0:
        return False
    board[r][c] = 1
    if _check_overline(board, r, c, 1):
        board[r][c] = 0
        return True
    threes, fours = 0, 0
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        line_str = "".join(
            map(str, [_get_char(board, r + i * dr, c + i * dc) for i in range(-4, 5)])
        )
        # Synchronized logic with the main app to prevent inconsistencies.
        line_str = line_str.replace("2", "0")
        if "01110" in line_str:
            threes += 1
        if "1111" in line_str:
            fours += 1
    board[r][c] = 0
    return threes >= 2 or fours >= 2


def check_line_at(board, r, c, p, length):
    if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == 0):
        return False
    board[r][c] = p
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        count = 1
        for i in range(1, length):
            if (
                0 <= r + i * dr < 15
                and 0 <= c + i * dc < 15
                and board[r + i * dr][c + i * dc] == p
            ):
                count += 1
            else:
                break
        for i in range(1, length):
            if (
                0 <= r - i * dr < 15
                and 0 <= c - i * dc < 15
                and board[r - i * dr][c - i * dc] == p
            ):
                count += 1
            else:
                break
        if count >= length:
            board[r][c] = 0
            return True
    board[r][c] = 0
    return False


def count_open_threes(board, p):
    """A simple heuristic to count open-three threats for a player."""
    count = 0
    opponent = 3 - p
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 0:
                board[r][c] = p
                for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    line = [
                        _get_char(board, r + i * dr, c + i * dc) for i in range(-4, 5)
                    ]
                    line_str = "".join(map(str, line)).replace(str(opponent), "X")
                    if "0" + str(p) * 3 + "0" in line_str:
                        count += 1
                board[r][c] = 0  # backtrack
    return count


@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    board = data["board"]
    color_to_play = data.get("color_to_play")
    banned_moves_enabled = data.get("banned_moves_enabled", False)
    game_phase = data.get("game_phase")

    # --- Handle Swap2 Choices ---
    if game_phase == "P2_CHOOSE":
        if count_open_threes(board, 1) > 0:
            return jsonify({"choice": "TAKE_BLACK"})
        if count_open_threes(board, 1) == 0 and count_open_threes(board, 2) == 0:
            return jsonify({"choice": "PLACE_2"})
        else:
            return jsonify({"choice": "TAKE_WHITE"})

    if game_phase == "P1_CHOOSE":
        if count_open_threes(board, 1) >= count_open_threes(board, 2):
            return jsonify({"choice": "CHOOSE_BLACK"})
        else:
            return jsonify({"choice": "CHOOSE_WHITE"})

    # --- Handle All Move Placements ---
    if game_phase == GamePhase.SWAP2_P1_PLACE_3:
        num_stones_on_board = sum(row.count(1) + row.count(2) for row in board)
        if num_stones_on_board < 3:
            opening_pattern = OPENING_BOOK[0]
            move = opening_pattern[num_stones_on_board]
            return jsonify({"move": move})

    elif game_phase == GamePhase.SWAP2_P2_PLACE_2:
        pass  # Fall through to standard move logic

    # --- Standard Baseline Logic ---
    opponent_color = 3 - color_to_play
    empty_cells = [
        (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if board[r][c] == 0
    ]
    if not empty_cells:
        return jsonify({"move": None})

    valid_moves = [
        m
        for m in empty_cells
        if not (
            color_to_play == 1
            and banned_moves_enabled
            and _is_banned_move(board, m[0], m[1])
        )
    ]
    if not valid_moves:
        valid_moves = empty_cells

    # 1. Win
    for r, c in valid_moves:
        if check_line_at(board, r, c, color_to_play, 5):
            return jsonify({"move": [r, c]})
    # 2. Block opponent's win
    for r, c in valid_moves:
        if check_line_at(board, r, c, opponent_color, 5):
            return jsonify({"move": [r, c]})
    # 3. Create an open four
    for r, c in valid_moves:
        if check_line_at(board, r, c, color_to_play, 4):
            return jsonify({"move": [r, c]})
    # 4. Block opponent's open four
    for r, c in valid_moves:
        if check_line_at(board, r, c, opponent_color, 4):
            return jsonify({"move": [r, c]})
    # 5. Block opponent's three
    for r, c in valid_moves:
        if check_line_at(board, r, c, opponent_color, 3):
            return jsonify({"move": [r, c]})

    # 6. Fallback: Play a random valid move
    return jsonify({"move": random.choice(valid_moves)})


if __name__ == "__main__":
    app.run(port=5002, debug=True)
