# old_vs_new_to_txt.py
import time
import random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import chess
import chess.pgn
import chess.polyglot


# ======================
# CONFIG
# ======================
OLD_CKPT = "../weights/option2_newCNN.pth"
NEW_CKPT = "../weights/cnn_L_256x5_h512_256.pth"

OUT_TXT  = "old_vs_new_pgns.txt"
APPEND   = False

N_GAMES   = 20
DEPTH_OLD = 5
DEPTH_NEW = 5

# If you want to force the *first* move, set this (e.g. "e4").
# NOTE: if it doesn't match the opening line selected, we will skip that line and resample.
FORCE_FIRST_MOVE_SAN = "e4"  # or None

# --- Deterministic search (no random sampling, no jitter) ---
MOVE_ORDER_JITTER = 0.0
ROOT_EPS_NOISE    = 0.0
ROOT_TOPK         = 1
ROOT_TEMP         = 0.0

# --- GM opening DB: use one full line per game, then start engines from there ---
USE_DB_OPENING   = True
DB_NPZ_PATH      = "Games.npz"
DB_USE_FULL_LINE = True      # <-- what you asked for
DB_REQUIRE_MATCH_FORCED_FIRST = True  # if FORCE_FIRST_MOVE_SAN is set
BASE_SEED        = 1337

TT_MAX_ITEMS = 500_000

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)


# ======================
# REPRO SEEDING
# ======================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# GM OPENING DATABASE (FULL LINE SAMPLER)
# ======================
class ChessDatabase:
    """
    Expects NPZ with key 'positions' that is a list of lines,
    each line is a list of moves (SAN or UCI).
    """
    def __init__(self, npz_file: str):
        data = np.load(npz_file, allow_pickle=True)
        self.lines = data["positions"].tolist()

    def __len__(self):
        return len(self.lines)

    def get_line(self, idx: int):
        return self.lines[idx]


# ======================
# MODELS
# ======================
class CNN(nn.Module):
    def __init__(self, in_ch: int, conv_channels=(128, 128, 128), fc_hidden=(256,), p_drop=0.0):
        super().__init__()
        conv = []
        prev = in_ch
        for c in conv_channels:
            conv += [nn.Conv2d(prev, c, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            prev = c
        self.conv = nn.Sequential(*conv)
        self.pool = nn.AdaptiveAvgPool2d(1)
        head = []
        prev = conv_channels[-1]
        for h in fc_hidden:
            head += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if p_drop and p_drop > 0:
                head += [nn.Dropout(p_drop)]
            prev = h
        head += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class CNNn(nn.Module):
    def __init__(self, in_ch: int, conv_channels=(128, 128, 128), fc_hidden=(256,), p_drop=0.0):
        super().__init__()
        conv = []
        prev = in_ch
        for c in conv_channels:
            conv += [
                nn.Conv2d(prev, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
            prev = c
        self.conv = nn.Sequential(*conv)
        self.pool = nn.AdaptiveAvgPool2d(1)
        head = []
        prev = conv_channels[-1]
        for h in fc_hidden:
            head += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if p_drop and p_drop > 0:
                head += [nn.Dropout(p_drop)]
            prev = h
        head += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ======================
# LOADERS
# ======================
def load_old_CNN(
    CNN_class,
    ckpt_path,
    device,
    strict=True,
    default_in_ch=17,
    default_conv_channels=(192, 192, 192, 192),
    default_fc_hidden=(256,),
    default_p_drop=0.1,
):
    p = torch.load(ckpt_path, map_location="cpu")
    state = p["model_state"] if isinstance(p, dict) and "model_state" in p else p
    cfg = (p.get("model_cfg", {}) if isinstance(p, dict) else {}) or {}

    in_ch = cfg.get("in_ch") or (p.get("input_dim") if isinstance(p, dict) else None) or default_in_ch
    conv_channels = cfg.get("conv_channels") or (p.get("conv_channels") if isinstance(p, dict) else None) or default_conv_channels
    fc_hidden = cfg.get("fc_hidden") or (p.get("arch") if isinstance(p, dict) else None) or default_fc_hidden
    p_drop = cfg.get("p_drop") or (p.get("p_drop") if isinstance(p, dict) else None) or default_p_drop

    model = CNN_class(int(in_ch), tuple(conv_channels), tuple(fc_hidden), float(p_drop)).to(device)
    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


def load_new_CNNn(
    CNNn_class,
    ckpt_path,
    device,
    strict=True,
    default_in_ch=17,
    default_conv_channels=(192, 192, 192, 192),
    default_fc_hidden=(256,),
    default_p_drop=0.1,
):
    p = torch.load(ckpt_path, map_location="cpu")
    state = p["model_state"] if isinstance(p, dict) and "model_state" in p else p
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not look like a state_dict: {type(state)}")

    cfg = (p.get("model_cfg", {}) if isinstance(p, dict) else {}) or {}

    in_ch = cfg.get("in_ch") or cfg.get("input_dim") or (p.get("input_dim") if isinstance(p, dict) else None)
    if in_ch is None:
        for k in ("conv.0.weight", "conv.0.0.weight"):
            if k in state:
                in_ch = int(state[k].shape[1])
                break
    if in_ch is None:
        in_ch = default_in_ch

    conv_channels = cfg.get("conv_channels") or (p.get("conv_channels") if isinstance(p, dict) else None)
    if conv_channels is None:
        # supports CNNn layout (Conv, BN, ReLU repeating => conv indices 0,3,6,...)
        chans = []
        i = 0
        while True:
            key = f"conv.{i}.weight"
            if key not in state:
                break
            chans.append(int(state[key].shape[0]))
            i += 3
        conv_channels = tuple(chans) if len(chans) else default_conv_channels
    else:
        conv_channels = tuple(conv_channels)

    fc_hidden = cfg.get("fc_hidden") or cfg.get("arch") or (p.get("arch") if isinstance(p, dict) else None)
    if fc_hidden is None:
        head_ws = []
        for k, v in state.items():
            if k.startswith("head.") and k.endswith(".weight") and getattr(v, "ndim", None) == 2:
                try:
                    idx = int(k.split(".")[1])
                except Exception:
                    idx = 10**9
                head_ws.append((idx, v))
        head_ws.sort(key=lambda t: t[0])
        if len(head_ws) >= 2:
            fc_hidden = tuple(int(w.shape[0]) for _, w in head_ws[:-1])
        else:
            fc_hidden = default_fc_hidden
    else:
        fc_hidden = tuple(fc_hidden)

    p_drop = cfg.get("p_drop")
    if p_drop is None and isinstance(p, dict):
        p_drop = p.get("p_drop")
    if p_drop is None:
        p_drop = default_p_drop

    model = CNNn_class(
        int(in_ch),
        conv_channels=tuple(conv_channels),
        fc_hidden=tuple(fc_hidden),
        p_drop=float(p_drop),
    ).to(device)

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model


# ======================
# ENCODING (17 planes)
# ======================
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_icpram_bitmap_plus(board: chess.Board) -> np.ndarray:
    planes = []
    for color, sign in ((chess.WHITE, +1), (chess.BLACK, -1)):
        for pt in PIECE_ORDER:
            m = np.zeros((8, 8), dtype=np.int8)
            for sq in board.pieces(pt, color):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                m[r, c] = sign
            planes.append(m)

    wk = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    wq = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    bk = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    bq = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    planes.append(np.full((8, 8), wk, dtype=np.int8))
    planes.append(np.full((8, 8), wq, dtype=np.int8))
    planes.append(np.full((8, 8), bk, dtype=np.int8))
    planes.append(np.full((8, 8), bq, dtype=np.int8))

    stm = +1 if board.turn == chess.WHITE else -1
    planes.append(np.full((8, 8), stm, dtype=np.int8))

    return np.stack(planes, axis=0)


# ======================
# ORDERED MOVES
# ======================
_PV = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

def move_score(board: chess.Board, move: chess.Move) -> float:
    s = 0.0
    if move.promotion is not None:
        s += 10_000 + _PV.get(move.promotion, 0)
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v = _PV[victim.piece_type] if victim else 1
        a = _PV[attacker.piece_type] if attacker else 0
        s += 1_000 + 10 * v - a
    if board.gives_check(move):
        s += 500
    if MOVE_ORDER_JITTER > 0:
        s += random.uniform(-MOVE_ORDER_JITTER, MOVE_ORDER_JITTER)
    return s

def ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: move_score(board, m), reverse=True)
    return moves


# ======================
# MINIMAX + TT
# ======================
def make_tt():
    TT = OrderedDict()
    def tt_get(key):
        v = TT.get(key)
        if v is not None:
            TT.move_to_end(key)
        return v
    def tt_put(key, entry):
        old = TT.get(key)
        if old is not None and entry[0] < old[0]:
            TT.move_to_end(key)
            return
        TT[key] = entry
        TT.move_to_end(key)
        if len(TT) > TT_MAX_ITEMS:
            TT.popitem(last=False)
    def tt_clear():
        TT.clear()
    return tt_get, tt_put, tt_clear


@torch.inference_mode()
def eval_with(model, board: chess.Board) -> float:
    x_np = board_to_icpram_bitmap_plus(board)
    x = torch.from_numpy(x_np).unsqueeze(0).float().to(device, non_blocking=True)
    return float(model(x).item())


def make_minimax(model, tt_get, tt_put):
    def minimax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        key = chess.polyglot.zobrist_hash(board)

        entry = tt_get(key)
        if entry is not None:
            d0, flag, val = entry
            if d0 >= depth:
                if flag == "EXACT":
                    return val
                if flag == "LOWER":
                    alpha = max(alpha, val)
                if flag == "UPPER":
                    beta = min(beta, val)
                if beta <= alpha:
                    return val

        if depth == 0 or board.is_game_over(claim_draw=True):
            v = eval_with(model, board)
            tt_put(key, (depth, "EXACT", v))
            return v

        a0, b0 = alpha, beta
        push, pop = board.push, board.pop

        if board.turn == chess.WHITE:
            best = -np.inf
            for mv in ordered_moves(board):
                push(mv)
                v = minimax(board, depth - 1, alpha, beta)
                pop()
                best = max(best, v)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
        else:
            best = np.inf
            for mv in ordered_moves(board):
                push(mv)
                v = minimax(board, depth - 1, alpha, beta)
                pop()
                best = min(best, v)
                beta = min(beta, best)
                if beta <= alpha:
                    break

        flag = "EXACT"
        if best <= a0:
            flag = "UPPER"
        elif best >= b0:
            flag = "LOWER"

        tt_put(key, (depth, flag, float(best)))
        return float(best)

    return minimax


# ======================
# ROOT PICK (deterministic)
# ======================
def pick_move_root(board: chess.Board, minimax_fn, depth: int) -> chess.Move | None:
    moves = ordered_moves(board)
    if not moves:
        return None

    push, pop = board.push, board.pop
    alpha, beta = -np.inf, np.inf

    if board.turn == chess.WHITE:
        best = -np.inf
        best_mv = None
        for mv in moves:
            push(mv)
            v = minimax_fn(board, depth - 1, alpha, beta)
            pop()
            if v > best:
                best = v
                best_mv = mv
            alpha = max(alpha, best)
        return best_mv
    else:
        best = np.inf
        best_mv = None
        for mv in moves:
            push(mv)
            v = minimax_fn(board, depth - 1, alpha, beta)
            pop()
            if v < best:
                best = v
                best_mv = mv
            beta = min(beta, best)
        return best_mv


# ======================
# APPLY ONE FULL OPENING LINE (SAN or UCI) TO BOARD + PGN
# ======================
def apply_opening_line(board: chess.Board, node, line_moves):
    san_history = []
    for mv_str in line_moves:
        if board.is_game_over(claim_draw=True):
            break

        mv = None
        # try SAN
        try:
            mv = board.parse_san(mv_str)
        except Exception:
            mv = None

        # try UCI
        if mv is None:
            try:
                m2 = chess.Move.from_uci(mv_str)
                if m2 in board.legal_moves:
                    mv = m2
            except Exception:
                mv = None

        if mv is None or mv not in board.legal_moves:
            # stop the line at first illegal move
            break

        san = board.san(mv)
        node = node.add_variation(mv)
        board.push(mv)
        san_history.append(san)

    return node, san_history


# ======================
# PLAY ONE GAME
# ======================
def play_one_game(game_idx: int, new_is_white: bool, model_old, model_new, opening_line) -> chess.pgn.Game:
    seed_everything(BASE_SEED + 1000 * game_idx + (1 if new_is_white else 0))

    tt_get_old, tt_put_old, tt_clear_old = make_tt()
    tt_get_new, tt_put_new, tt_clear_new = make_tt()
    tt_clear_old()
    tt_clear_new()

    minimax_old = make_minimax(model_old, tt_get_old, tt_put_old)
    minimax_new = make_minimax(model_new, tt_get_new, tt_put_new)

    board = chess.Board()
    forced = FORCE_FIRST_MOVE_SAN

    game = chess.pgn.Game()
    game.headers["Event"] = "NEW vs OLD"
    game.headers["Site"] = "Local"
    game.headers["Date"] = time.strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_idx)
    game.headers["DepthOld"] = str(DEPTH_OLD)
    game.headers["DepthNew"] = str(DEPTH_NEW)
    game.headers["ForcedFirst"] = str(forced or "")
    game.headers["Seed"] = str(BASE_SEED + 1000 * game_idx + (1 if new_is_white else 0))
    game.headers["OpeningLineLen"] = str(len(opening_line) if opening_line else 0)

    white_name = f"NEW(d={DEPTH_NEW})" if new_is_white else f"OLD(d={DEPTH_OLD})"
    black_name = f"OLD(d={DEPTH_OLD})" if new_is_white else f"NEW(d={DEPTH_NEW})"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name

    node = game

    # If forcing first move, play it before the line (and require line matches if configured)
    san_history = []
    if forced:
        mv0 = board.parse_san(forced)
        san0 = board.san(mv0)
        node = node.add_variation(mv0)
        board.push(mv0)
        san_history.append(san0)

    # Apply the FULL opening line (from database) starting from current position
    if opening_line:
        node, san_from_line = apply_opening_line(board, node, opening_line)
        san_history.extend(san_from_line)

    # Now deterministic minimax from that resulting position
    while not board.is_game_over(claim_draw=True):
        if board.turn == chess.WHITE:
            mm = minimax_new if new_is_white else minimax_old
            depth = DEPTH_NEW if new_is_white else DEPTH_OLD
        else:
            mm = minimax_old if new_is_white else minimax_new
            depth = DEPTH_OLD if new_is_white else DEPTH_NEW

        mv = pick_move_root(board, mm, depth)
        if mv is None or mv not in board.legal_moves:
            raise RuntimeError(f"Illegal/None move at ply {board.ply()} | FEN: {board.fen()} | mv={mv}")

        node = node.add_variation(mv)
        board.push(mv)

    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["PlyCount"] = str(board.ply())
    return game


# ======================
# MAIN
# ======================
def main():
    model_old = load_old_CNN(CNN, OLD_CKPT, device, strict=True)
    model_new = load_new_CNNn(CNNn, NEW_CKPT, device, strict=True)

    # Prepare opening lines: shuffle once, then take a different line each game.
    opening_lines = [None] * N_GAMES
    if USE_DB_OPENING and Path(DB_NPZ_PATH).exists():
        db = ChessDatabase(DB_NPZ_PATH)
        if len(db) == 0:
            print("WARN: DB has 0 lines. Continuing without DB openings.")
        else:
            idxs = list(range(len(db)))
            rng = random.Random(BASE_SEED)   # deterministic shuffle independent of global RNG
            rng.shuffle(idxs)

            # Fill games with unique lines (wrap-around if DB smaller than N_GAMES)
            filled = 0
            ptr = 0
            attempts = 0
            while filled < N_GAMES and attempts < N_GAMES * 50:
                attempts += 1
                idx = idxs[ptr % len(idxs)]
                ptr += 1
                line = db.get_line(idx)

                # If we require forced-first match: line must start with that SAN exactly.
                if FORCE_FIRST_MOVE_SAN and DB_REQUIRE_MATCH_FORCED_FIRST:
                    if not line or line[0] != FORCE_FIRST_MOVE_SAN:
                        continue

                opening_lines[filled] = line
                filled += 1

            if filled < N_GAMES:
                print(f"WARN: Could only assign {filled}/{N_GAMES} opening lines that match constraints. Remaining games use no opening.")
    else:
        if USE_DB_OPENING:
            print(f"WARN: USE_DB_OPENING=True but '{DB_NPZ_PATH}' not found. Continuing without DB openings.")

    out_path = Path(OUT_TXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if APPEND else "w"
    with out_path.open(mode, encoding="utf-8") as f:
        for g in range(1, N_GAMES + 1):
            new_is_white = (g % 2 == 1)
            line = opening_lines[g - 1]

            game = play_one_game(g, new_is_white, model_old, model_new, line)

            f.write(str(game).strip() + "\n")
            f.write("\n" + ("=" * 80) + "\n\n")
            f.flush()

            print(
                f"[{g}/{N_GAMES}] {game.headers['Result']} | NEW as {'White' if new_is_white else 'Black'} "
                f"| plies={game.headers['PlyCount']} | opening_len={game.headers.get('OpeningLineLen','0')}"
            )

    print("Wrote:", out_path.resolve())


if __name__ == "__main__":
    main()
