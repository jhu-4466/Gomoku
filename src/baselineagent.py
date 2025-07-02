from flask import Flask, request, jsonify
import random

app = Flask(__name__)

GRID_SIZE = 19


def check_line_at(board, r, c, player, length):
    """
    Checks if placing a stone for 'player' at (r, c) creates a line of 'length'.
    Returns True if it does, False otherwise.
    """
    if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == 0):
        return False

    board[r][c] = player  # Temporarily place the stone for checking

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonals
    for dr, dc in directions:
        count = 1
        # Count in the positive direction
        for i in range(1, length):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr][nc] == player:
                count += 1
            else:
                break
        # Count in the negative direction
        for i in range(1, length):
            nr, nc = r - i * dr, c - i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr][nc] == player:
                count += 1
            else:
                break

        if count >= length:
            board[r][c] = 0  # Reset the board
            return True

    board[r][c] = 0  # Reset the board
    return False


@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    board = data["board"]
    player = data["player"]
    opponent = 3 - player

    empty_cells = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if board[r][c] == 0:
                empty_cells.append((r, c))

    # 1. Offensive Check: Can I win now? (Make a line of 5)
    for r, c in empty_cells:
        if check_line_at(board, r, c, player, 5):
            return jsonify({"move": [r, c]})
    # 2. Defensive Check: Must I block an opponent's win? (Opponent makes a line of 5)
    for r, c in empty_cells:
        if check_line_at(board, r, c, opponent, 5):
            return jsonify({"move": [r, c]})
    # 3. Defensive Check: Must I block an opponent's four? (THIS IS A CRITICAL DEFENSE)
    for r, c in empty_cells:
        if check_line_at(board, r, c, opponent, 4):
            return jsonify({"move": [r, c]})
    # 4. Defensive Check: Must I block an opponent's three? (THIS IS THE USER-SUGGESTED IMPROVEMENT)
    for r, c in empty_cells:
        if check_line_at(board, r, c, opponent, 3):
            return jsonify({"move": [r, c]})
    # 5. Fallback Strategy: No immediate threats, play a random move
    if empty_cells:
        random.shuffle(empty_cells)
        chosen_move = empty_cells[0]
        return jsonify({"move": chosen_move})

    return jsonify({"move": None, "message": "Board is full"}), 200


if __name__ == "__main__":
    app.run(port=5002, debug=True)
