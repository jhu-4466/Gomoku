from flask import Flask, request, jsonify
import random

app = Flask(__name__)
GRID_SIZE = 15


# --- Banned Move Logic (Copied from main app for self-containment) ---
def _get_char(board, r, c):
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        return board[r][c]
    return -1  # Represents board edge


def _check_overline(board, r, c):
    p = board[r][c]
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

    board[r][c] = 1  # Temporarily place the stone

    if _check_overline(board, r, c):
        board[r][c] = 0
        return True

    threes, fours = 0, 0
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        line_str = "".join(
            map(str, [_get_char(board, r + i * dr, c + i * dc) for i in range(-4, 5)])
        )
        # Replace opponent stones with a non-interfering character for pattern matching
        line_str = line_str.replace("2", "X")

        if "01110" in line_str:
            threes += 1
        if "1111" in line_str:
            fours += 1

    board[r][c] = 0  # Reset the board
    return threes >= 2 or fours >= 2


# --- API Endpoint ---
@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    if not data or "board" not in data:
        return jsonify({"error": "Invalid input"}), 400

    board = data["board"]
    player = data.get("player")
    banned_moves_enabled = data.get("banned_moves_enabled", False)

    empty_cells = []
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] == 0:
                empty_cells.append((r, c))

    if not empty_cells:
        return jsonify({"move": None, "message": "Board is full"}), 200

    valid_moves = []
    # If it's Black's turn and banned moves are on, filter the moves
    if player == 1 and banned_moves_enabled:
        for r, c in empty_cells:
            if not _is_banned_move(board, r, c):
                valid_moves.append((r, c))
    else:
        valid_moves = empty_cells

    if valid_moves:
        chosen_move = random.choice(valid_moves)
        print(f"Random Strategy chose: {chosen_move}")
        return jsonify({"move": chosen_move})
    else:
        # This can happen if all available moves are banned
        return jsonify({"move": random.choice(empty_cells)})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
