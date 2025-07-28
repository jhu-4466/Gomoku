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

    # --- MODIFIED FUNCTION ---
    def _get_response(
        self, timeout=29.8
    ):  # Increased timeout to be safely above engine's timeout
        """
        Gets the next valid move coordinate from the engine, ignoring all other
        informational (DEBUG, MESSAGE, ERROR) or acknowledgment (OK) messages.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._stdout_queue.get(timeout=1)
                print(f"RSP < {response}")

                # --- NEW LOGIC ---
                # Specifically look for a response in the "x,y" format.
                # This is the single most important change.
                if "," in response:
                    parts = response.split(",")
                    if (
                        len(parts) == 2
                        and parts[0].strip().isdigit()
                        and parts[1].strip().isdigit()
                    ):
                        print(f"VALID MOVE PARSED: {response}")
                        return response  # Return the valid move string

                # If it's not a move, it's logged and ignored. The loop continues.

            except queue.Empty:
                # This is not an error, just the queue being temporarily empty.
                # The outer while loop will handle the overall timeout.
                continue

        # If the while loop finishes, a true timeout has occurred.
        raise queue.Empty(
            f"Engine did not respond with a valid move coordinate within the {timeout}s period."
        )

    def _wait_for_ok(self, timeout=29.8):
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
        # We don't necessarily need to wait for an OK here, as it's an informational command

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
        # STEP 1: Always initialize the engine and set timeout on a new game signal.
        if is_new_game:
            print("--- New Game Signal. Sending START and setting timeout. ---")
            gomoku_engine_wrapper.start_game()
            gomoku_engine_wrapper.set_timeout(
                29500
            )  # Set engine timeout to 29.5 seconds

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
        # The parsing logic is now simpler because _get_response guarantees a valid format or throws a timeout.
        if move_str:
            parts = move_str.split(",")
            move = [int(parts[0].strip()), int(parts[1].strip())]
            return jsonify({"move": move, "search_depth": -1})

        # This part should ideally not be reached unless _get_response returns an empty string, which it won't.
        error_msg = f"Engine returned unexpected empty output: '{move_str}'"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

    except queue.Empty as e:
        error_msg = f"FATAL: Engine timed out. {e}"
        print(error_msg)
        # In case of a timeout, the engine state is now unknown.
        # It's safest to force a restart on the next call.
        # A more advanced implementation could kill and restart the subprocess here.
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
