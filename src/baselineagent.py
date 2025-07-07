from flask import Flask, request, jsonify
import random

app = Flask(__name__)
GRID_SIZE = 15


# --- Banned Move Logic (Copied and adapted) ---
def _get_char(board, r, c):
    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
        return board[r][c]
    return -1


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
    board[r][c] = 1
    if _check_overline(board, r, c):
        board[r][c] = 0
        return True
    threes, fours = 0, 0
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        line_str = "".join(
            map(str, [_get_char(board, r + i * dr, c + i * dc) for i in range(-4, 5)])
        )
        line_str = line_str.replace("2", "X")
        if "01110" in line_str:
            threes += 1
        if "1111" in line_str:
            fours += 1
    board[r][c] = 0
    return threes >= 2 or fours >= 2


# --- Helper for Baseline Logic ---
def check_line_at(board, r, c, player, length):
    if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == 0):
        return False
    board[r][c] = player
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for i in range(1, length):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr][nc] == player:
                count += 1
            else:
                break
        for i in range(1, length):
            nr, nc = r - i * dr, c - i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr][nc] == player:
                count += 1
            else:
                break
        if count >= length:
            board[r][c] = 0
            return True
    board[r][c] = 0
    return False


# --- API Endpoint ---
@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    board = data["board"]
    player = data["player"]
    opponent = 3 - player
    banned_moves_enabled = data.get("banned_moves_enabled", False)

    empty_cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 0:
                empty_cells.append((r, c))

    if not empty_cells:
        return jsonify({"move": None, "message": "Board is full"}), 200

    # Filter out banned moves if applicable
    valid_moves = []
    if player == 1 and banned_moves_enabled:
        for r, c in empty_cells:
            if not _is_banned_move(board, r, c):
                valid_moves.append((r, c))
    else:
        valid_moves = empty_cells

    if not valid_moves:  # If all moves are banned, fall back to any empty cell
        valid_moves = empty_cells

    # 1. Offensive Check: Can I win now?
    for r, c in valid_moves:
        if check_line_at(board, r, c, player, 5):
            return jsonify({"move": [r, c]})
    # 2. Defensive Check: Must I block an opponent's win?
    for r, c in valid_moves:
        if check_line_at(board, r, c, opponent, 5):
            return jsonify({"move": [r, c]})
    # 3. Defensive Check: Must I block an opponent's four?
    for r, c in valid_moves:
        if check_line_at(board, r, c, opponent, 4):
            return jsonify({"move": [r, c]})
    # 4. Defensive Check: Must I block an opponent's three?
    for r, c in valid_moves:
        if check_line_at(board, r, c, opponent, 3):
            return jsonify({"move": [r, c]})
    # 5. Fallback Strategy: Play a random valid move
    random.shuffle(valid_moves)
    chosen_move = valid_moves[0]
    return jsonify({"move": chosen_move})


if __name__ == "__main__":
    app.run(port=5002, debug=True)
