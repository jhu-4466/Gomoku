# -*- coding: utf-8 -*-

import subprocess
import os
import time
import argparse
import threading
import queue
from flask import Flask, request, jsonify


# --- The AgentWrapper Class (Integrated) ---
# This class handles the low-level, persistent communication with the AI process.
class AgentWrapper:
    def __init__(self, exe_path, board_size=15):
        """Launches the Agent engine subprocess and prepares for communication."""
        self.process = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Read/write in text mode (UTF-8)
            bufsize=1,  # Line-buffered
            universal_newlines=True,
        )
        self.board_size = board_size
        self._stdout_queue = queue.Queue()
        # Start a separate thread to continuously read the engine's output to avoid blocking
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self):
        """Background thread: continuously reads engine output and puts it in a queue."""
        while True:
            # Check if the process is still running before trying to read from stdout
            if self.process.poll() is not None:
                break
            line = self.process.stdout.readline()
            if not line:
                break
            self._stdout_queue.put(line.strip())

    def _send_command(self, cmd):
        """Sends a single line command to the engine and flushes the buffer."""
        print(f"CMD > {cmd}")  # Log the command being sent
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _get_response(self, timeout=31):
        """
        Gets the next valid move coordinate from the engine, ignoring all other
        informational (DEBUG, MESSAGE, ERROR) or acknowledgment (OK) messages.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._stdout_queue.get(timeout=1)
                print(f"RSP < {response}")

                if "," in response:
                    parts = response.split(",")
                    if (
                        len(parts) == 2
                        and parts[0].strip().isdigit()
                        and parts[1].strip().isdigit()
                    ):
                        print(f"VALID MOVE PARSED: {response}")
                        return response

            except queue.Empty:
                continue

        raise queue.Empty(
            f"Engine did not respond with a valid move coordinate within the {timeout}s period."
        )

    def _wait_for_ok(self, timeout=30):
        """Waits specifically for an OK response, ignoring everything else."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._stdout_queue.get(timeout=1)
                print(f"RSP < {response}")
                if response.upper().startswith("OK"):
                    return  # Success
            except queue.Empty:
                continue
        raise queue.Empty("Engine did not respond with OK.")

    def start_game(self):
        """Sends the START command and waits for the OK acknowledgement."""
        self._send_command(f"START {self.board_size}")
        self._wait_for_ok()

    def set_timeout(self, ms):
        """Sends the INFO timeout_turn command to set the time limit per turn."""
        self._send_command(f"INFO timeout_turn {ms}")

    def set_rule(self, rule_value):
        """
        Sends the INFO rule command to set the game rules (e.g., 1 for Renju).
        """
        self._send_command(f"INFO rule {rule_value}")
        # According to the protocol, we don't need to wait for an OK for INFO commands.

    def begin(self):
        """Sends the BEGIN command to get the AI's first move."""
        self._send_command("BEGIN")
        return self._get_response()

    def turn(self, x, y):
        """Sends the TURN command to get the AI's next move."""
        self._send_command(f"TURN {x},{y}")
        return self._get_response()


# --- Helper Function for Move Adjustment ---
def find_nearest_empty_spot(board, x, y):
    board_size = len(board)
    if not board_size or not len(board[0]):
        return None

    clamped_x = max(0, min(x, board_size - 1))
    clamped_y = max(0, min(y, board_size - 1))

    if board[clamped_x][clamped_y] == 0:
        return [clamped_x, clamped_y]
    for r in range(1, board_size):

        def check_and_return(cx, cy):
            if 0 <= cx < board_size and 0 <= cy < board_size:
                if board[cx][cy] == 0:
                    return [cx, cy]
            return None

        for i in range(-r, r + 1):
            # top
            spot = check_and_return(clamped_x + i, clamped_y - r)
            if spot:
                return spot

            # bottom
            spot = check_and_return(clamped_x + i, clamped_y + r)
            if spot:
                return spot

        for i in range(-r + 1, r):
            # left
            spot = check_and_return(clamped_x - r, clamped_y + i)
            if spot:
                return spot

            # right
            spot = check_and_return(clamped_x + r, clamped_y + i)
            if spot:
                return spot

    return None


# --- Flask Web Server Setup ---
app = Flask(__name__)
gomoku_engine_wrapper = None
GRID_SIZE = 15


@app.route("/get_move", methods=["POST"])
def get_move():
    """Endpoint that handles move requests from the GUI."""
    global gomoku_engine_wrapper
    if not gomoku_engine_wrapper:
        return jsonify({"error": "Engine wrapper not initialized."}), 500

    data = request.get_json()
    board = data["board"]
    move_history = data.get("move_history", [])
    is_new_game = data.get("new_game", False)
    banned_moves_enabled = data.get("banned_moves_enabled", False)

    start_time = time.time()
    try:
        move_str = ""
        # STEP 1: Always initialize the engine and set timeout on a new game signal.
        if is_new_game:
            print("--- New Game Signal. Sending START, setting rules, and timeout. ---")
            gomoku_engine_wrapper.start_game()
            gomoku_engine_wrapper.set_timeout(29500)
            if banned_moves_enabled:
                gomoku_engine_wrapper.set_rule(1)

        # STEP 2: Prompt for the actual move.
        if not move_history:
            # AI is Player 1, so it needs the BEGIN command.
            print("--- AI is Player 1. Sending BEGIN to get first move. ---")
            move_str = gomoku_engine_wrapper.begin()
        else:
            # AI is Player 2 or it's a later turn.
            last_move = move_history[-1]["move"]
            print(
                f"--- Sending TURN for opponent's move: {[last_move[0], last_move[1]]} ---"
            )
            move_str = gomoku_engine_wrapper.turn(last_move[0], last_move[1])

        thinking_time = time.time() - start_time
        print(f"AI thinking time: {thinking_time:.2f}s")

        # STEP 3: Parse and VALIDATE the response.
        if move_str:
            parts = move_str.split(",")
            move = [int(parts[0].strip()), int(parts[1].strip())]

            # --- Check for out-of-bounds move and adjust if necessary ---
            if move[0] >= GRID_SIZE or move[1] >= GRID_SIZE:
                print(
                    f"WARNING: AI returned out-of-bounds move {move}. Finding nearest empty spot..."
                )
                adjusted_move = find_nearest_empty_spot(board, move[0], move[1])

                if adjusted_move:
                    move = adjusted_move
                    print(f"SUCCESS: Adjusted move to {move}.")
                else:
                    # This case happens if the board is full
                    error_msg = f"FATAL: AI returned out-of-bounds move {move} but no empty spots were found."
                    print(error_msg)
                    return jsonify({"error": error_msg}), 500

            return jsonify({"move": move, "search_depth": -1})

        error_msg = f"Engine returned unexpected empty output: '{move_str}'"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

    except queue.Empty as e:
        error_msg = f"FATAL: Engine timed out. {e}"
        print(error_msg)
        return jsonify({"error": error_msg, "recommend_restart": True}), 500
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a web API wrapper for a Gomoku AI executable."
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the Gomoku AI executable (e.g., ai.exe)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5004,
        help="Port number for this wrapper server to run on (default: 5004)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=15,
        help="Size of the Gomoku board (default: 15)",
    )
    args = parser.parse_args()

    # Update the global GRID_SIZE before initializing the engine
    GRID_SIZE = args.size

    AI_EXECUTABLE_PATH = os.path.abspath(args.model_path)
    SERVER_PORT = args.port

    if not os.path.exists(AI_EXECUTABLE_PATH):
        print(
            f"Error: Model file not found at the specified path '{AI_EXECUTABLE_PATH}'."
        )
        exit(1)

    print(f"Initializing Gomoku Engine Wrapper for '{AI_EXECUTABLE_PATH}'...")
    gomoku_engine_wrapper = AgentWrapper(AI_EXECUTABLE_PATH, board_size=GRID_SIZE)

    print(
        f"--- Wrapper Initialized. Starting server on http://127.0.0.1:{SERVER_PORT} ---"
    )
    app.run(port=SERVER_PORT, debug=False)
