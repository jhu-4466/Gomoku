# -*- coding: utf-8 -*-

import subprocess
import os
import time
import argparse
import threading
import queue
from flask import Flask, request, jsonify


# --- Game Phase Constants ---
class GamePhase:
    NORMAL = 0
    SWAP2_P1_PLACE_3 = 1
    SWAP2_P2_CHOOSE_ACTION = 2
    SWAP2_P2_PLACE_2 = 3
    SWAP2_P1_CHOOSE_COLOR = 4


# --- The RAPFIWrapper Class (Corrected) ---
class RAPFIWrapper:
    def __init__(self, exe_path, board_size=15):
        """Launches the RAPFI engine subprocess and prepares for communication."""
        self.process = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self.board_size = board_size
        self.swap2_cache = []

        self._stdout_queue = queue.Queue()
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self):
        """Background thread: continuously reads engine output and puts it in a queue."""
        while True:
            if self.process.poll() is not None:
                break
            line = self.process.stdout.readline()
            if not line:
                break
            self._stdout_queue.put(line.strip())

    def _send_command(self, cmd):
        """Sends a single line command to the engine and flushes the buffer."""
        print(f"CMD > {cmd}")
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _get_raw_line(self, timeout=29.8):
        """Gets the next line of output from the engine."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._stdout_queue.get(timeout=1)
                print(f"RSP < {response}")
                return response
            except queue.Empty:
                continue
        raise queue.Empty(f"Engine did not respond within the {timeout}s period.")

    def _get_data_response(self, timeout=29.8):
        """
        Gets the next meaningful data line from the engine, ignoring all 'OK' messages.
        This is a robust way to get actual data instead of acknowledgements.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self._get_raw_line(timeout=timeout - (time.time() - start_time))
            if response.upper().startswith("OK"):
                continue  # Ignore OK and wait for the next line
            return response  # Return the actual data
        raise queue.Empty("Engine did not respond with data (only OK or nothing).")

    def _get_move_response(self, timeout=31):
        """Gets a valid move coordinate, using the data response logic."""
        response = self._get_data_response(timeout)
        if "," in response:
            parts = response.split(",")
            if (
                len(parts) == 2
                and parts[0].strip().isdigit()
                and parts[1].strip().isdigit()
            ):
                print(f"VALID MOVE PARSED: {response}")
                return response
        raise TypeError(f"Expected a move coordinate, but got: {response}")

    def _wait_for_ok(self, timeout=30):
        """Waits specifically for an OK response, ignoring everything else."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self._get_raw_line(timeout=1)
            if response.upper().startswith("OK"):
                return
        raise queue.Empty("Engine did not respond with OK.")

    def _get_p2_choice(self, timeout=30):
        response = self._get_data_response()

        # Case 1: The AI responds with 'SWAP'
        if response.upper() == "SWAP":
            print("PARSED P2 CHOICE: TAKE_BLACK")
            return {"action": "SWAP"}

        # Case 3: The AI responds with two moves (e.g., "8,8 8,6")
        parts = response
        if len(parts) == 2:
            try:
                # Assumes a helper function parse_move_str exists
                move1 = parse_move_str(parts[0])
                move2 = parse_move_str(parts[1])
                print(f"PARSED P2 CHOICE: PLACE_2, Moves: {move1}, {move2}")
                self.swap2_cache = [move1, move2]
                return {"action": "PLACE_2"}
            except Exception:
                raise ValueError(
                    f"Could not parse two moves from AI response: '{response}'"
                )

        # Case 2: The AI responds with a single move (e.g., "8,8")
        if len(parts) == 1:
            try:
                move = parse_move_str(parts[0])
                print(f"PARSED P2 CHOICE: TAKE_WHITE, Move: {move}")
                self.swap2_cache = [move]
                return {"action": "TAKE_WHITE"}
            except Exception:
                raise ValueError(
                    f"Could not parse a single move from AI response: '{response}'"
                )

        # If the response matches no known pattern, raise an error
        raise ValueError(f"Unexpected response format for P2 choice: '{response}'")

    def _get_p1_color_choice(self, timeout=30):
        response = self._get_data_response()

        # Case 1: The AI responds with 'SWAP'
        if response.upper() == "SWAP":
            print("PARSED P2 CHOICE: TAKE_WHITE")
            return {"action": "SWAP"}

        # Split the response by spaces to check for one or two coordinates
        parts = response.split()
        # Case 2: The AI responds with a single move (e.g., "8,8")
        if len(parts) == 1:
            try:
                move = parse_move_str(parts[0])
                print(f"PARSED P2 CHOICE: TAKE_WHITE, Move: {move}")
                self.swap2_cache = [move]
                return {"action": "TAKE_BLACK"}
            except Exception:
                raise ValueError(
                    f"Could not parse a single move from AI response: '{response}'"
                )

        # If the response matches no known pattern, raise an error
        raise ValueError(f"Unexpected response format for P2 choice: '{response}'")

    def start_game(self):
        self._send_command(f"START {self.board_size}")
        self._wait_for_ok()

    def set_timeout(self, ms):
        self._send_command(f"INFO timeout_turn {ms}")

    def set_rule(self, rule_value):
        self._send_command(f"INFO rule {rule_value}")

    def swap_p1_place_3(self):
        self._send_command("SWAP2BOARD")
        self._send_command("DONE")

        response = self._get_data_response()
        parts = response.split()
        self.swap2_cache = [parse_move_str(part) for part in parts if "," in part]

    def swap2_p2_choose_action(self, move_history):
        """
        Sends the entire board state to the engine using BOARD and DONE commands.
        """
        move_strings = [f"{m['move'][0]},{m['move'][1]}" for m in move_history]
        self._send_command("SWAP2BOARD")
        for move_str in move_strings:
            self._send_command(move_str)
        self._send_command("DONE")
        return self._get_p2_choice()

    def swap2_p1_choose_color(self, move_history):
        """
        Sends the current board state to the engine and gets the color choice.
        """
        move_strings = [f"{m['move'][0]},{m['move'][1]}" for m in move_history]
        self._send_command("SWAP2BOARD")
        for move_str in move_strings:
            self._send_command(move_str)
        self._send_command("DONE")
        return self._get_data_response()

    def turn(self, x, y):
        self._send_command(f"TURN {x},{y}")
        return self._get_move_response()


# --- Flask Web Server Setup ---
app = Flask(__name__)
gomoku_engine_wrapper = None


def parse_move_str(move_str):
    parts = move_str.split(",")
    return [int(parts[0].strip()), int(parts[1].strip())]


@app.route("/get_move", methods=["POST"])
def get_move():
    global gomoku_engine_wrapper
    if not gomoku_engine_wrapper:
        return jsonify({"error": "Engine wrapper not initialized."}), 500

    data = request.get_json()
    move_history = data.get("move_history", [])
    game_phase = data.get("game_phase", GamePhase.NORMAL)
    is_new_game = data.get("new_game", False)
    swap2_enabled = data.get("swap2_enabled", False)
    banned_moves_enabled = data.get("banned_moves_enabled", False)

    start_time = time.time()
    try:
        if is_new_game:
            print("--- New Game Signal. Sending START, setting rules, and timeout. ---")
            gomoku_engine_wrapper.start_game()
            gomoku_engine_wrapper.set_timeout(29500)
            if swap2_enabled or banned_moves_enabled:
                gomoku_engine_wrapper.set_rule(4)

        response_data = {}
        if swap2_enabled:
            if game_phase == GamePhase.SWAP2_P1_PLACE_3:
                print("--- AI is Player 1 (Swap2). Getting 3 initial moves. ---")
                if not gomoku_engine_wrapper.swap2_cache:
                    gomoku_engine_wrapper.swap_p1_place_3()
                move_str = gomoku_engine_wrapper.swap2_cache.pop(0)
                move = parse_move_str(move_str)
                response_data = {"move": move}
                move_history.append(",".join(move))
            elif game_phase == GamePhase.SWAP2_P2_CHOOSE_ACTION:
                print("--- AI is Player 2 (Swap2). Choosing action. ---")
                p2_choice = gomoku_engine_wrapper.swap2_p2_choose_action(move_history)
                response_data = {"choice": p2_choice}
            elif game_phase == GamePhase.SWAP2_P2_PLACE_2:
                print("--- AI is Player 2 (Swap2). Placing 2 stones. ---")
                move_str = gomoku_engine_wrapper.swap2_cache.pop(0)
                move = parse_move_str(move_str)
                response_data = {"move": move}
                move_history.append(",".join(move))
            elif game_phase == GamePhase.SWAP2_P1_CHOOSE_COLOR:
                print("--- AI is Player 1 (Swap2). Choosing color. ---")
                gomoku_engine_wrapper._get_p1_color_choice(move_history)
                choice = gomoku_engine_wrapper.get_swap2_color_choice()
                response_data = {"color_choice": choice}
            elif gomoku_engine_wrapper.swap2_cache:
                move_str = gomoku_engine_wrapper.swap2_cache.pop(0)
                move = parse_move_str(move_str)
                response_data = {"move": move}
                move_history.append(",".join(move))

            return jsonify(response_data)

        # --- NORMAL GAMEPLAY LOGIC ---
        print("--- Normal Gameplay Turn ---")
        if not move_history:
            print("--- AI is Player 1. Sending BEGIN to get first move. ---")
            move_str = (
                gomoku_engine_wrapper.swap_p1_place_3()[0]
                if swap2_enabled
                else gomoku_engine_wrapper.turn(7, 7)
            )
        else:
            last_move = move_history[-1]["move"]
            print(f"--- Sending TURN for opponent's move: {last_move} ---")
            move_str = gomoku_engine_wrapper.turn(last_move[0], last_move[1])

        thinking_time = time.time() - start_time
        print(f"AI thinking time: {thinking_time:.2f}s")
        if move_str:
            move = parse_move_str(move_str)
            return jsonify({"move": move})

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
