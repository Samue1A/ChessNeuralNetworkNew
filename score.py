import chess
import chess.pgn
import chess.engine

PGN_FILE = "old_vs_new_pgns.txt"
STOCKFISH_PATH = "../../n1/.ipynb_checkpoints/eval/stockfish/stockfish/stockfish-windows-x86-64-avx2.exe"
print('hi')
ANALYSIS_DEPTH = 18   # increase if you want stronger adjudication

import re
text = open("old_vs_new_pgns.txt", "r", encoding="utf-8").read()
print("Event header count:", len(re.findall(r"(?m)^\\[Event\\s+\".*\"\\]$", text)))
print("Round header count:", len(re.findall(r"(?m)^\\[Round\\s+\".*\"\\]$", text)))


new_points = 0
old_points = 0
total_games = 0

with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
    with open(PGN_FILE) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            total_games += 1
            board = game.board()

            # play all moves to reach final position
            for move in game.mainline_moves():
                board.push(move)

            info = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
            score = info["score"].white().score(mate_score=100000)

            white_name = game.headers.get("White", "")
            black_name = game.headers.get("Black", "")

            # Pure sign decision
            if score >= 0:
                winner = "White"
            else:
                winner = "Black"

            if winner == "White":
                if "NEW" in white_name:
                    new_points += 1
                else:
                    old_points += 1
            else:
                if "NEW" in black_name:
                    new_points += 1
                else:
                    old_points += 1

            print(f"Game {total_games}: eval={score} -> {winner} wins")

print("\nFINAL SCORE (Stockfish final-position adjudication)")
print("NEW:", new_points)
print("OLD:", old_points)
print("Total games:", total_games)
