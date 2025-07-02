from flask import Flask, request, jsonify
import numpy as np
import random

app = Flask(__name__)


@app.route("/get_move", methods=["POST"])
def get_move():
    data = request.get_json()
    if not data or "board" not in data:
        return jsonify({"error": "Invalid input"}), 400

    board = data["board"]

    empty_cells = []
    for r in range(len(board)):
        for c in range(len(board[r])):
            if board[r][c] == 0:
                empty_cells.append((r, c))

    if empty_cells:
        chosen_move = random.choice(empty_cells)
        print(f"Random Strategy chose: {chosen_move}")
        return jsonify({"move": chosen_move})
    else:
        return jsonify({"move": None, "message": "Board is full"}), 200


if __name__ == "__main__":
    app.run(port=5001, debug=True)
