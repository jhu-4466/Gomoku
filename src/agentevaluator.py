# -*- coding: utf-8 -*-
import json
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

# --- Constants ---
GRID_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2
AGENT_TO_EVALUATE = "Negamax AI"


class AgentEvaluator:
    """
    Analyzes a single Gomoku game from a history file and evaluates
    the performance of the participating agents.
    """

    def __init__(self, history_data):
        self.history = history_data
        self.stats = {}  # Will be populated for the agent we are evaluating

    def _get_line_patterns(self, board, r, c, player):
        """
        A simplified pattern checker to find threats around a point.
        """
        patterns = {"FIVE": 0, "LIVE_FOUR": 0, "RUSH_FOUR": 0, "LIVE_THREE": 0}
        opponent = 3 - player

        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            line_str = ""
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    stone = board[nr, nc]
                    if stone == player:
                        line_str += "O"
                    elif stone == opponent:
                        line_str += "X"
                    else:
                        line_str += "_"
                else:
                    line_str += "X"

            if "OOOOO" in line_str:
                patterns["FIVE"] += 1
            if "_OOOO_" in line_str:
                patterns["LIVE_FOUR"] += 1
            if "XOOOO_" in line_str or "_OOOOX" in line_str:
                patterns["RUSH_FOUR"] += 1
            if "_OOO_" in line_str:
                patterns["LIVE_THREE"] += 1

        return patterns

    def analyze_move(self, board_before, move_info, player_name, player_color):
        """Analyzes a single move's strategic contribution."""
        r, c = move_info["move"]
        opponent_color = 3 - player_color

        # Assess Defensive Value
        board_before[r, c] = opponent_color
        opponent_threats = self._get_line_patterns(board_before, r, c, opponent_color)
        board_before[r, c] = EMPTY
        if opponent_threats["FIVE"] > 0:
            self.stats[player_name]["defense"]["Blocked Fives"] += 1
        if opponent_threats["LIVE_FOUR"] > 0:
            self.stats[player_name]["defense"]["Blocked Live Fours"] += 1
        if opponent_threats["LIVE_THREE"] > 0:
            self.stats[player_name]["defense"]["Blocked Live Threes"] += 1

        # Assess Offensive Value
        board_after = board_before.copy()
        board_after[r, c] = player_color
        our_threats = self._get_line_patterns(board_after, r, c, player_color)
        if our_threats["LIVE_THREE"] > 0:
            self.stats[player_name]["offense"]["Live Threes Created"] += 1

    def run_analysis(self):
        """Replays the game, analyzes each move, and returns structured results."""
        p1_name = self.history["player_setup"]["p1_name"]
        p2_name = self.history["player_setup"]["p2_name"]
        agent_name = None

        if p1_name == AGENT_TO_EVALUATE:
            agent_name = p1_name
        elif p2_name == AGENT_TO_EVALUATE:
            agent_name = p2_name
        else:
            return None  # Agent not in this game

        # Initialize stats for the agent in this game
        self.stats[agent_name] = {
            "offense": defaultdict(int),
            "defense": defaultdict(int),
            "total_moves": 0,
            "total_thinking_time": 0.0,
            "timed_moves": 0,
            "search_depths": [],  # --- NEW: List to store search depths ---
        }

        board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        for move_info in self.history["move_history"]:
            move_player_id = move_info["player_id"]
            move_player_name = self.history["player_setup"][f"p{move_player_id}_name"]

            if move_player_name == agent_name:
                self.stats[agent_name]["total_moves"] += 1
                if "thinking_time" in move_info:
                    self.stats[agent_name]["total_thinking_time"] += move_info[
                        "thinking_time"
                    ]
                    self.stats[agent_name]["timed_moves"] += 1

                # --- NEW: Record search depth if available ---
                if "search_depth" in move_info:
                    self.stats[agent_name]["search_depths"].append(
                        move_info["search_depth"]
                    )

                self.analyze_move(
                    board.copy(), move_info, move_player_name, move_info["color"]
                )

            r, c = move_info["move"]
            board[r, c] = move_info["color"]

        return self.get_report_data()

    def get_report_data(self):
        """Returns a dictionary with the analysis results for this game."""
        if not self.stats:
            return None

        agent_stats = self.stats[AGENT_TO_EVALUATE]
        winner_name = self.history.get("winner_name", "N/A")

        result = "Draw"
        if winner_name == AGENT_TO_EVALUATE:
            result = "Win"
        elif winner_name != "None" and winner_name != "N/A" and winner_name != "Draw":
            result = "Loss"

        avg_time = (
            (agent_stats["total_thinking_time"] / agent_stats["timed_moves"])
            if agent_stats["timed_moves"] > 0
            else 0
        )

        depths = agent_stats["search_depths"]
        avg_depth = np.mean(depths) if depths else 0
        max_depth = np.max(depths) if depths else 0

        return {
            "Game ID": self.history["game_id"],
            "Agent": AGENT_TO_EVALUATE,
            "Opponent": (
                self.history["player_setup"]["p1_name"]
                if self.history["player_setup"]["p2_name"] == AGENT_TO_EVALUATE
                else self.history["player_setup"]["p2_name"]
            ),
            "Winner": winner_name,
            "Result": result,
            "Agent Total Moves": agent_stats["total_moves"],
            "Agent Avg. Time (s)": round(avg_time, 4),
            "Agent Avg. Depth": round(avg_depth, 2),  # --- NEW COLUMN ---
            "Agent Max Depth": int(max_depth),  # --- NEW COLUMN ---
            "Blocked Fives": agent_stats["defense"]["Blocked Fives"],
            "Blocked Live Fours": agent_stats["defense"]["Blocked Live Fours"],
            "Blocked Live Threes": agent_stats["defense"]["Blocked Live Threes"],
            "Live Threes Created": agent_stats["offense"]["Live Threes Created"],
        }


class BatchEvaluator:
    """
    Manages the evaluation of all game history files in a directory
    and generates a summary report.
    """

    def __init__(self, history_dir):
        self.history_dir = history_dir
        self.all_game_reports = []

    def run_batch_analysis(self):
        print(f"Starting batch analysis in directory: '{self.history_dir}'...")
        if not os.path.isdir(self.history_dir):
            print(f"Error: Directory not found at '{self.history_dir}'")
            return

        history_files = [f for f in os.listdir(self.history_dir) if f.endswith(".json")]
        if not history_files:
            print("No game history files (.json) found.")
            return

        for filename in sorted(history_files):
            filepath = os.path.join(self.history_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    game_data = json.load(f)

                evaluator = AgentEvaluator(game_data)
                report_data = evaluator.run_analysis()

                if report_data:
                    self.all_game_reports.append(report_data)
                    print(f"  - Analyzed {filename}")

            except Exception as e:
                print(f"  - Failed to analyze {filename}: {e}")

    def generate_excel_report(self):
        if not self.all_game_reports:
            print("No data to generate report.")
            return

        details_df = pd.DataFrame(self.all_game_reports)

        # Create Summary Row
        total_games = len(details_df)
        wins = (details_df["Result"] == "Win").sum()
        losses = (details_df["Result"] == "Loss").sum()
        draws = (details_df["Result"] == "Draw").sum()
        win_rate = (wins / total_games * 100) if total_games > 0 else 0

        total_moves = details_df["Agent Total Moves"].sum()
        total_time = (
            details_df["Agent Avg. Time (s)"] * details_df["Agent Total Moves"]
        ).sum()
        overall_avg_time = (total_time / total_moves) if total_moves > 0 else 0

        total_weighted_depth = (
            details_df["Agent Avg. Depth"] * details_df["Agent Total Moves"]
        ).sum()
        overall_avg_depth = (
            (total_weighted_depth / total_moves) if total_moves > 0 else 0
        )
        overall_max_depth = (
            details_df["Agent Max Depth"].max() if not details_df.empty else 0
        )

        summary_row = {
            "Game ID": "SUMMARY",
            "Agent": AGENT_TO_EVALUATE,
            "Opponent": "All",
            "Winner": f"{wins} Wins, {losses} Losses, {draws} Draws",
            "Result": f"{win_rate:.2f}% Win Rate",
            "Agent Total Moves": details_df["Agent Total Moves"].sum(),
            "Agent Avg. Time (s)": round(overall_avg_time, 4),
            "Agent Avg. Depth": round(overall_avg_depth, 2),
            "Agent Max Depth": "N/A",
            "Blocked Fives": round(details_df["Blocked Fives"].mean(), 2),
            "Blocked Live Fours": round(details_df["Blocked Live Fours"].mean(), 2),
            "Blocked Live Threes": round(details_df["Blocked Live Threes"].mean(), 2),
            "Live Fours Created": round(details_df["Live Fours Created"].mean(), 2),
            "Live Threes Created": round(details_df["Live Threes Created"].mean(), 2),
        }

        # Append summary row
        summary_df = pd.DataFrame([summary_row])
        final_df = pd.concat([details_df, summary_df], ignore_index=True)

        # Create output directory and save file
        output_dir = "evaluation"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(
            output_dir, f"evaluation_report_{timestamp}.xlsx"
        )

        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="Evaluation_Report", index=False)

            # Auto-adjust column widths for better readability
            worksheet = writer.sheets["Evaluation_Report"]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = max_length + 2
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"\nEvaluation report successfully generated: {output_filename}")


if __name__ == "__main__":
    history_dir = sys.argv[1] if len(sys.argv) > 1 else "game_history"

    batch_evaluator = BatchEvaluator(history_dir)
    batch_evaluator.run_batch_analysis()
    batch_evaluator.generate_excel_report()
