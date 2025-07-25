from flask import Flask, request, jsonify
import random

app = Flask(__name__)
GRID_SIZE = 15


# GamePhase class is defined for clarity, though RandomAgent doesn't use it for complex logic.
class GamePhase:
    NORMAL = 0
    SWAP2_P1_PLACE_3 = 1
    SWAP2_P2_CHOOSE_ACTION = 2
    SWAP2_P2_PLACE_2 = 3
    SWAP2_P1_CHOOSE_COLOR = 4


# --- Banned Move Logic (Synchronized with main app and baseline agent) ---
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
        # This logic is now identical to the baseline agent and main app's logic.
        line_str = line_str.replace("2", "0")
        if "01110" in line_str:
            threes += 1
        if "1111" in line_str:
            fours += 1
    board[r][c] = 0
    return threes >= 2 or fours >= 2


@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    if not data or "board" not in data:
        return jsonify({"error": "Invalid input"}), 400

    board = data["board"]
    color_to_play = data.get("color_to_play")
    banned_moves_enabled = data.get("banned_moves_enabled", False)
    game_phase = data.get("game_phase")

    # --- Handle Swap2 Choices ---
    if game_phase == GamePhase.SWAP2_P2_CHOOSE_ACTION:
        choices = ["TAKE_BLACK", "TAKE_WHITE", "PLACE_2"]
        return jsonify({"choice": random.choice(choices)})
    if game_phase == GamePhase.SWAP2_P1_CHOOSE_COLOR:
        choices = ["CHOOSE_BLACK", "CHOOSE_WHITE"]
        return jsonify({"choice": random.choice(choices)})

    # --- Handle All Move Placements ---
    empty_cells = [
        (r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if board[r][c] == 0
    ]
    if not empty_cells:
        return jsonify({"move": None, "message": "Board is full"})

    # Filter out banned moves if applicable
    if color_to_play == 1 and banned_moves_enabled:
        valid_moves = [m for m in empty_cells if not _is_banned_move(board, m[0], m[1])]
        # If all moves are banned, it's a loss for black, but for robustness,
        # we allow it to make a move from any empty cell to prevent a crash.
        if not valid_moves:
            valid_moves = empty_cells
    else:
        valid_moves = empty_cells

    if valid_moves:
        return jsonify({"move": random.choice(valid_moves)})
    else:
        # This case should ideally not be reached.
        return jsonify({"move": None, "message": "No valid moves available"})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
