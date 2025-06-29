from flask import Flask, request, jsonify
import random

app = Flask(__name__)

GRID_SIZE = 19


def check_win_at(board, r, c, player):
    if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and board[r][c] == 0):
        return False

    board[r][c] = player  # temp position for checking
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for i in range(1, 5):
            nr, nc = r + i * dr, c + i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr][nc] == player:
                count += 1
            else:
                break
        for i in range(1, 5):
            nr, nc = r - i * dr, c - i * dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and board[nr][nc] == player:
                count += 1
            else:
                break

        if count >= 5:
            board[r][c] = 0
            return True

    board[r][c] = 0
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

    # if it is possible to win
    for r, c in empty_cells:
        if check_win_at(board, r, c, player):
            print(f"Baseline Strategy: Found winning move at {(r, c)}")
            return jsonify({"move": [r, c]})

    # has to defense
    for r, c in empty_cells:
        if check_win_at(board, r, c, opponent):
            print(f"Baseline Strategy: Found blocking move at {(r, c)}")
            return jsonify({"move": [r, c]})

    # 3. back up radom startegy
    if empty_cells:
        chosen_move = random.choice(empty_cells)
        print(f"Baseline Strategy: Fallback to random move: {chosen_move}")
        return jsonify({"move": chosen_move})

    return jsonify({"move": None, "message": "Board is full"}), 200


if __name__ == "__main__":
    app.run(port=5002, debug=True)
