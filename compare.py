# ==========================================================
# OLD vs NEW MOVE RANKING TEST (Standalone Script)
# ==========================================================

import random
import numpy as np
import chess
import chess.polyglot
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ---------------- CONFIG ----------------
OLD_CKPT = "../weights/option2_newCNN.pth"
NEW_CKPT = "../weights/option3_newResLayer.pth"

N_POSITIONS = 200
SEARCH_DEPTH = 4   # use 3–4 for speed
TT_MAX = 200_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# MODELS
# ==========================================================

class CNN(nn.Module):
    def __init__(self, in_ch, conv_channels=(192,192,192,192),
                 fc_hidden=(256,), p_drop=0.1):
        super().__init__()

        conv = []
        prev = in_ch
        for c in conv_channels:
            conv += [nn.Conv2d(prev, c, 3, padding=1),
                     nn.ReLU(inplace=True)]
            prev = c
        self.conv = nn.Sequential(*conv)

        self.pool = nn.AdaptiveAvgPool2d(1)

        head = []
        prev = conv_channels[-1]
        for h in fc_hidden:
            head += [nn.Linear(prev, h),
                     nn.ReLU(inplace=True)]
            if p_drop > 0:
                head += [nn.Dropout(p_drop)]
            prev = h

        head += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.head(x)



class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class CNNn(nn.Module):
    def __init__(self, in_ch, trunk_ch=192, n_blocks=8, fc_hidden=(256,), p_drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, trunk_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(trunk_ch) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)

        head = []
        prev = trunk_ch
        for h in fc_hidden:
            head += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        head += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ==========================================================
# LOADERS
# ==========================================================

def load_old():
    ckpt = torch.load(OLD_CKPT, map_location="cpu")
    state = ckpt["model_state"]
    model = CNN(17, (192,192,192,192), (256,), 0.1).to(device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_new():
    ckpt = torch.load(NEW_CKPT, map_location="cpu")
    state = ckpt["model_state"]
    model = CNNn(17, 192, 8, (256,), 0.1).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

model_old = load_old()
model_new = load_new()

# ==========================================================
# ENCODING
# ==========================================================

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board):
    planes = []
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        for pt in PIECE_ORDER:
            m = np.zeros((8,8), dtype=np.float32)
            for sq in board.pieces(pt, color):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                m[r,c] = sign
            planes.append(m)

    # castling
    planes.append(np.full((8,8), board.has_kingside_castling_rights(chess.WHITE)))
    planes.append(np.full((8,8), board.has_queenside_castling_rights(chess.WHITE)))
    planes.append(np.full((8,8), board.has_kingside_castling_rights(chess.BLACK)))
    planes.append(np.full((8,8), board.has_queenside_castling_rights(chess.BLACK)))

    # side to move
    planes.append(np.full((8,8), 1 if board.turn else -1))

    return np.stack(planes)

@torch.inference_mode()
def eval_with(model, board):
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(device)
    return float(model(x).item())

# ==========================================================
# SEARCH
# ==========================================================

def make_minimax(model):
    TT = OrderedDict()

    def tt_get(key):
        v = TT.get(key)
        if v is not None:
            TT.move_to_end(key)
        return v

    def tt_put(key, entry):
        TT[key] = entry
        TT.move_to_end(key)
        if len(TT) > TT_MAX:
            TT.popitem(last=False)

    def minimax(board, depth, alpha, beta):
        key = chess.polyglot.zobrist_hash(board)
        entry = tt_get(key)
        if entry and entry[0] >= depth:
            return entry[1]

        if depth == 0 or board.is_game_over():
            v = eval_with(model, board)
            tt_put(key, (depth, v))
            return v

        if board.turn:
            best = -1e9
            for mv in board.legal_moves:
                board.push(mv)
                v = minimax(board, depth-1, alpha, beta)
                board.pop()
                best = max(best, v)
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
        else:
            best = 1e9
            for mv in board.legal_moves:
                board.push(mv)
                v = minimax(board, depth-1, alpha, beta)
                board.pop()
                best = min(best, v)
                beta = min(beta, v)
                if beta <= alpha:
                    break

        tt_put(key, (depth, best))
        return best

    return minimax

minimax_old = make_minimax(model_old)
minimax_new = make_minimax(model_new)

def pick_move(minimax_fn, board, depth):
    best_mv = None
    if board.turn:
        best = -1e9
        for mv in board.legal_moves:
            board.push(mv)
            v = minimax_fn(board, depth-1, -1e9, 1e9)
            board.pop()
            if v > best:
                best = v
                best_mv = mv
    else:
        best = 1e9
        for mv in board.legal_moves:
            board.push(mv)
            v = minimax_fn(board, depth-1, -1e9, 1e9)
            board.pop()
            if v < best:
                best = v
                best_mv = mv
    return best_mv

# ==========================================================
# RANK TEST
# ==========================================================

def rank_test(model, minimax_fn, name):
    top1 = 0
    ranks = []

    for i in range(N_POSITIONS):
        board = chess.Board()

        for _ in range(random.randint(6,20)):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

        if board.is_game_over():
            continue

        best_move = pick_move(minimax_fn, board, SEARCH_DEPTH)
        if best_move is None:
            continue

        moves = list(board.legal_moves)
        scores = []

        for mv in moves:
            board.push(mv)
            scores.append(eval_with(model, board))
            board.pop()

        scores = np.array(scores)

        if board.turn:
            order = np.argsort(-scores)
        else:
            order = np.argsort(scores)

        ranked_moves = [moves[i] for i in order]
        rank = ranked_moves.index(best_move)
        ranks.append(rank)

        if rank == 0:
            top1 += 1

    print("\nResults for", name)
    print("Top-1 accuracy:", top1/len(ranks))
    print("Avg rank:", np.mean(ranks))


# ==========================================================
# RUN
# ==========================================================

rank_test(model_old, minimax_old, "OLD")
rank_test(model_new, minimax_new, "NEW")
