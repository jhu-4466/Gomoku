# -*- coding: utf-8 -*-

import subprocess
import os
import time
import argparse
import threading
import queue
from flask import Flask, request, jsonify


# --- The RAPFIWrapper Class (Integrated) ---
# This class handles the low-level, persistent communication with the AI process.
class RAPFIWrapper:
    def __init__(self, exe_path, board_size=15):
        """Launches the RAPFI engine subprocess and prepares for communication."""
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
            line = self.process.stdout.readline()
            if not line:
                break
            self._stdout_queue.put(line.strip())

    def _send_command(self, cmd):
        """Sends a single line command to the engine and flushes the buffer."""
        print(f"CMD > {cmd}")  # Log the command being sent
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _get_response(self, timeout=30):
        """Gets the next meaningful line from the engine, skipping info messages."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._stdout_queue.get(timeout=1)
                print(f"RSP < {response}")
                if not response.upper().startswith(
                    "MESSAGE"
                ) and not response.upper().startswith("OK"):
                    return response  # Return the actual move or error
            except queue.Empty:
                continue
        raise queue.Empty(
            "Engine did not respond with a move within the timeout period."
        )

    def _wait_for_ok(self, timeout=10):
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

    def begin(self):
        """Sends the BEGIN command to get the AI's first move."""
        self._send_command("BEGIN")
        return self._get_response()

    def turn(self, x, y):
        """Sends the TURN command to get the AI's next move."""
        self._send_command(f"TURN {x},{y}")
        return self._get_response()


# --- Flask Web Server Setup ---
app = Flask(__name__)
gomoku_engine_wrapper = None


@app.route("/get_move", methods=["POST"])
def get_move():
    """Endpoint that handles move requests from the GUI."""
    global gomoku_engine_wrapper
    if not gomoku_engine_wrapper:
        return jsonify({"error": "Engine wrapper not initialized."}), 500

    data = request.get_json()
    move_history = data.get("move_history", [])
    is_new_game = data.get("new_game", False)

    start_time = time.time()
    try:
        move_str = ""
        # STEP 1: Always initialize the engine on a new game signal.
        if is_new_game:
            print("--- New Game Signal. Sending START and waiting for OK. ---")
            gomoku_engine_wrapper.start_game()

        # STEP 2: Prompt for the actual move.
        if not move_history:
            # AI is Player 1, so it needs the BEGIN command.
            print("--- AI is Player 1. Sending BEGIN to get first move. ---")
            move_str = gomoku_engine_wrapper.begin()
        else:
            # AI is Player 2 or it's a later turn.
            last_move = move_history[-1]["move"]
            print(f"--- Sending TURN for opponent's move: {last_move} ---")
            move_str = gomoku_engine_wrapper.turn(last_move[0], last_move[1])

        thinking_time = time.time() - start_time
        print(f"AI thinking time: {thinking_time:.2f}s")

        # STEP 3: Parse the response.
        if move_str and "," in move_str:
            parts = move_str.split(",")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                move = [int(parts[0]), int(parts[1])]
                return jsonify({"move": move, "search_depth": -1})

        error_msg = f"Engine returned unexpected output: '{move_str}'"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

    except queue.Empty as e:
        error_msg = f"Engine timed out. {e}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500
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
    args = parser.parse_args()

    AI_EXECUTABLE_PATH = os.path.abspath(args.model_path)
    SERVER_PORT = args.port

    if not os.path.exists(AI_EXECUTABLE_PATH):
        print(
            f"Error: Model file not found at the specified path '{AI_EXECUTABLE_PATH}'."
        )
        exit(1)

    print(f"Initializing Gomoku Engine Wrapper for '{AI_EXECUTABLE_PATH}'...")
    gomoku_engine_wrapper = RAPFIWrapper(AI_EXECUTABLE_PATH)

    print(
        f"--- Wrapper Initialized. Starting server on http://127.0.0.1:{SERVER_PORT} ---"
    )
    app.run(port=SERVER_PORT, debug=False)
