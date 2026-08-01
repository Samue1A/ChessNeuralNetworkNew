from __future__ import annotations

import io
import math
import queue
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess
import chess.pgn
import chess.polyglot

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


DOWNLOADS = Path.home() / "Downloads"
DEFAULT_WEIGHTS = Path("C:/Users/samue/OneDrive/Desktop/projects/ML/chess-learn/n6/weights/new_labels2.pth")
DEFAULT_PGN_DIR = Path("C:/Users/samue/Downloads/chess_play_outputs/human_vs_dual1_improved.pgn")

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
POLICY_DIM = 4672
VALUE_CP_SCALE = 400.0
MATE_SCORE = 100_000.0

PIECE_VALUE_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

UNICODE_PIECES = {
    "P": "\u2659",
    "N": "\u2658",
    "B": "\u2657",
    "R": "\u2656",
    "Q": "\u2655",
    "K": "\u2654",
    "p": "\u265f",
    "n": "\u265e",
    "b": "\u265d",
    "r": "\u265c",
    "q": "\u265b",
    "k": "\u265a",
}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x, inplace=True)
        return x


class ChessNet(nn.Module):
    def __init__(self, in_ch: int = 19, filters: int = 128, num_blocks: int = 10, policy_dim: int = POLICY_DIM):
        super().__init__()
        self.input_conv = nn.Conv2d(in_ch, filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(filters)
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(num_blocks)])

        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_dim)

    def forward_features(self, x):
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)
        return x

    def forward_raw(self, x, legal_mask=None):
        x = self.forward_features(x)

        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = self.value_fc2(v)

        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        if legal_mask is not None:
            p = p.masked_fill(~legal_mask, float("-inf"))
        return v, p


def load_dual_model(ckpt_path: str | Path, device: torch.device) -> tuple[ChessNet, dict]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
        in_ch = int(ckpt.get("input_channels", ckpt.get("in_ch", 19)))
        filters = int(ckpt.get("filters", 128))
        num_blocks = int(ckpt.get("num_blocks", 10))
        policy_dim = int(ckpt.get("policy_dim", POLICY_DIM))
        meta = dict(ckpt)
    else:
        state = ckpt
        in_ch, filters, num_blocks, policy_dim = 19, 128, 10, POLICY_DIM
        meta = {}

    if state is None:
        raise KeyError("Could not find model weights in checkpoint.")

    model = ChessNet(in_ch=in_ch, filters=filters, num_blocks=num_blocks, policy_dim=policy_dim).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta


def board_to_icpram_bitmap_plus(board: chess.Board, channels_first: bool = True) -> np.ndarray:
    planes: list[np.ndarray] = []

    for color, sign in ((chess.WHITE, +1.0), (chess.BLACK, -1.0)):
        for piece_type in PIECE_ORDER:
            plane = np.zeros((8, 8), dtype=np.float32)
            for sq in board.pieces(piece_type, color):
                row = 7 - chess.square_rank(sq)
                col = chess.square_file(sq)
                plane[row, col] = sign
            planes.append(plane)

    planes.append(np.full((8, 8), 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.turn == chess.WHITE else -1.0, dtype=np.float32))

    ep = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        ep[row, col] = 1.0
    planes.append(ep)

    planes.append(np.full((8, 8), min(board.halfmove_clock, 100) / 100.0, dtype=np.float32))

    x = np.stack(planes, axis=0)
    if not channels_first:
        x = np.transpose(x, (1, 2, 0))
    return x


def move_to_index(move: chess.Move) -> int:
    if move.promotion and move.promotion != chess.QUEEN:
        promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
        return 4096 + move.from_square * 3 + promo_offset[move.promotion]
    return move.from_square * 64 + move.to_square


def get_legal_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_DIM, dtype=np.bool_)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < POLICY_DIM:
            mask[idx] = True
    return mask


def cp_for_side(cp_white: float, side: chess.Color) -> float:
    return float(cp_white if side == chess.WHITE else -cp_white)


def material_cp_white(board: chess.Board) -> int:
    total = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUE_CP.get(piece.piece_type, 0)
        total += value if piece.color == chess.WHITE else -value
    return total


def nonking_piece_count(board: chess.Board) -> int:
    return sum(1 for piece in board.piece_map().values() if piece.piece_type != chess.KING)


def is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if piece is None or piece.piece_type != chess.PAWN or piece.color != color:
        return False
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    enemy = not color
    files = [f for f in (file_idx - 1, file_idx, file_idx + 1) if 0 <= f <= 7]
    ranks = range(rank_idx + 1, 8) if color == chess.WHITE else range(rank_idx - 1, -1, -1)
    for f in files:
        for r in ranks:
            p = board.piece_at(chess.square(f, r))
            if p is not None and p.color == enemy and p.piece_type == chess.PAWN:
                return False
    return True


def passed_pawn_danger(board: chess.Board, color: chess.Color) -> float:
    danger = 0.0
    for sq in board.pieces(chess.PAWN, color):
        if not is_passed_pawn(board, sq, color):
            continue
        rank = chess.square_rank(sq)
        progress = rank if color == chess.WHITE else 7 - rank
        if progress >= 4:
            danger += (progress - 3) ** 2
            if chess.square_file(sq) in (3, 4):
                danger += 0.75
    return danger


def move_is_dangerous_passer_push(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    to_rank = chess.square_rank(move.to_square)
    return (piece.color == chess.WHITE and to_rank >= 5) or (piece.color == chess.BLACK and to_rank <= 2)


def move_flags(board: chess.Board, move: chess.Move) -> str:
    flags = []
    if board.is_capture(move):
        flags.append("capture")
    if board.gives_check(move):
        flags.append("check")
    if move.promotion:
        flags.append("promo")
    if move_is_dangerous_passer_push(board, move):
        flags.append("passer")
    child = board.copy(stack=True)
    child.push(move)
    if child.is_repetition(2):
        flags.append("repeat")
    return ", ".join(flags)


def make_child(board: chess.Board, move: chess.Move) -> chess.Board:
    child = board.copy(stack=True)
    child.push(move)
    return child


@dataclass
class CandidateInfo:
    move: chess.Move
    san: str
    policy_rank: int
    policy_logit: float
    flags: str
    static_cp: float
    depth2_cp: float
    side_score: float
    best_reply: str
    material_after: int
    conversion_score: float
    is_played: bool = False
    rank: int = 0
    loss_cp: float = 0.0


@dataclass
class PositionAnalysis:
    board_fen: str
    side_to_move: chess.Color
    root_value_cp: float
    played_move: Optional[chess.Move]
    played_san: str
    candidates: list[CandidateInfo]
    summary: str
    elapsed: float


class ModelDiagnostics:
    def __init__(self, weights_path: str | Path, root_top_n: int = 12, child_top_n: int = 6):
        self.weights_path = Path(weights_path)
        self.device = select_device()
        self.model, self.meta = load_dual_model(self.weights_path, self.device)
        self.root_top_n = int(root_top_n)
        self.child_top_n = int(child_top_n)
        self.eval_cache: OrderedDict[tuple[int, int], tuple[float, np.ndarray]] = OrderedDict()
        self.max_cache = 200_000

    def key(self, board: chess.Board) -> tuple[int, int]:
        return (chess.polyglot.zobrist_hash(board), min(board.halfmove_clock, 100))

    def terminal_value(self, board: chess.Board) -> Optional[float]:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            if board.is_repetition(3):
                return 0.0
            return None
        if outcome.winner is None:
            return 0.0
        score = MATE_SCORE - min(board.ply(), 10_000)
        return score if outcome.winner == chess.WHITE else -score

    def cache_put(self, key: tuple[int, int], value: tuple[float, np.ndarray]) -> None:
        self.eval_cache[key] = value
        self.eval_cache.move_to_end(key)
        if len(self.eval_cache) > self.max_cache:
            self.eval_cache.popitem(last=False)

    @torch.inference_mode()
    def evaluate_many(self, boards: list[chess.Board], batch_size: int = 128) -> list[tuple[float, np.ndarray]]:
        results: list[tuple[float, np.ndarray] | None] = [None] * len(boards)
        missing: list[tuple[int, chess.Board, tuple[int, int]]] = []

        for i, board in enumerate(boards):
            terminal = self.terminal_value(board)
            if terminal is not None:
                results[i] = (float(terminal), np.full(POLICY_DIM, -np.inf, dtype=np.float32))
                continue
            key = self.key(board)
            cached = self.eval_cache.get(key)
            if cached is not None:
                self.eval_cache.move_to_end(key)
                results[i] = cached
            else:
                missing.append((i, board, key))

        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            xs = np.stack([board_to_icpram_bitmap_plus(board) for _, board, _ in chunk], axis=0)
            masks = np.stack([get_legal_mask(board) for _, board, _ in chunk], axis=0)
            x = torch.from_numpy(xs).to(self.device, dtype=torch.float32, non_blocking=True)
            legal_mask = torch.from_numpy(masks).to(self.device, dtype=torch.bool, non_blocking=True)
            v_raw, p_logits = self.model.forward_raw(x, legal_mask)
            values = v_raw.squeeze(-1).detach().float().cpu().numpy() * VALUE_CP_SCALE
            logits = p_logits.detach().float().cpu().numpy()
            for row, (i, _board, key) in enumerate(chunk):
                item = (float(values[row]), logits[row])
                self.cache_put(key, item)
                results[i] = item

        return [item for item in results if item is not None]

    def evaluate(self, board: chess.Board) -> tuple[float, np.ndarray]:
        return self.evaluate_many([board])[0]

    def ordered_moves_from_logits(self, board: chess.Board, logits: np.ndarray) -> list[chess.Move]:
        scored = []
        for mv in board.legal_moves:
            idx = move_to_index(mv)
            score = float(logits[idx]) if 0 <= idx < len(logits) and np.isfinite(logits[idx]) else -1e9
            if mv.promotion:
                score += 0.45
            if board.is_capture(mv):
                score += 0.28
            if board.gives_check(mv):
                score += 0.25
            if move_is_dangerous_passer_push(board, mv):
                score += 0.20
            scored.append((score, mv))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [mv for _, mv in scored]

    def select_candidates(self, board: chess.Board, ordered: list[chess.Move], top_n: int, include: Optional[chess.Move] = None) -> list[chess.Move]:
        selected = list(ordered[: min(top_n, len(ordered))])
        seen = set(selected)
        for mv in ordered:
            if board.is_capture(mv) or board.gives_check(mv) or mv.promotion or move_is_dangerous_passer_push(board, mv):
                if mv not in seen:
                    selected.append(mv)
                    seen.add(mv)
        if include is not None and include not in seen and include in board.legal_moves:
            selected.append(include)
        return selected

    def conversion_score(self, board: chess.Board, move: chess.Move, searched_score: float, root_turn: chess.Color) -> float:
        side_mult = 1.0 if root_turn == chess.WHITE else -1.0
        side_score = searched_score * side_mult
        before_count = nonking_piece_count(board)
        before_enemy_passer = passed_pawn_danger(board, not root_turn)
        before_material = material_cp_white(board) * side_mult
        captured = board.piece_at(move.to_square)

        child = make_child(board, move)
        after_count = nonking_piece_count(child)
        after_enemy_passer = passed_pawn_danger(child, not root_turn)
        after_material = material_cp_white(child) * side_mult
        repeat_penalty = 1.0 if child.is_repetition(2) else 0.0

        opponent_forcing = 0
        for reply in child.legal_moves:
            if child.is_capture(reply) or child.gives_check(reply) or reply.promotion or move_is_dangerous_passer_push(child, reply):
                opponent_forcing += 1

        capture_value = PIECE_VALUE_CP.get(captured.piece_type, 0) if captured is not None else 0
        score = 0.18 * side_score
        score += 20.0 * (before_count - after_count)
        score += 0.08 * (after_material - before_material)
        score += 0.05 * capture_value
        score += 28.0 * (before_enemy_passer - after_enemy_passer)
        score -= 7.0 * opponent_forcing
        score -= 180.0 * repeat_penalty
        if board.gives_check(move) and side_score > 250:
            score -= 10.0
        return float(score)

    def analyze_position(
        self,
        board: chess.Board,
        played_move: Optional[chess.Move] = None,
        root_top_n: Optional[int] = None,
        child_top_n: Optional[int] = None,
    ) -> PositionAnalysis:
        start = time.perf_counter()
        root_top_n = self.root_top_n if root_top_n is None else int(root_top_n)
        child_top_n = self.child_top_n if child_top_n is None else int(child_top_n)
        board = board.copy(stack=True)
        root_turn = board.turn

        root_value, root_logits = self.evaluate(board)
        root_ordered = self.ordered_moves_from_logits(board, root_logits)
        policy_rank_by_move = {mv: i + 1 for i, mv in enumerate(root_ordered)}
        root_candidates = self.select_candidates(board, root_ordered, root_top_n, include=played_move)
        if not root_candidates:
            return PositionAnalysis(board.fen(), root_turn, root_value, played_move, "", [], "No legal moves.", 0.0)

        child_boards = [make_child(board, mv) for mv in root_candidates]
        child_evals = self.evaluate_many(child_boards)

        leaf_boards: list[chess.Board] = []
        leaf_owner: list[int] = []
        leaf_reply: list[chess.Move] = []
        depth2_values: list[Optional[float]] = [None] * len(root_candidates)
        best_replies: list[str] = [""] * len(root_candidates)

        for i, child in enumerate(child_boards):
            terminal = self.terminal_value(child)
            if terminal is not None:
                depth2_values[i] = terminal
                best_replies[i] = "(terminal)"
                continue
            child_logits = child_evals[i][1]
            replies_ordered = self.ordered_moves_from_logits(child, child_logits)
            replies = self.select_candidates(child, replies_ordered, child_top_n)
            if not replies:
                depth2_values[i] = child_evals[i][0]
                continue
            for reply in replies:
                leaf_boards.append(make_child(child, reply))
                leaf_owner.append(i)
                leaf_reply.append(reply)

        leaf_evals = self.evaluate_many(leaf_boards)
        grouped: dict[int, list[tuple[float, chess.Move]]] = defaultdict(list)
        for owner, reply, (value, _logits) in zip(leaf_owner, leaf_reply, leaf_evals):
            grouped[owner].append((float(value), reply))

        for i, child in enumerate(child_boards):
            if depth2_values[i] is not None:
                continue
            options = grouped.get(i, [])
            if not options:
                depth2_values[i] = child_evals[i][0]
                continue
            if child.turn == chess.WHITE:
                value, reply = max(options, key=lambda item: item[0])
            else:
                value, reply = min(options, key=lambda item: item[0])
            depth2_values[i] = value
            best_replies[i] = child.san(reply)

        infos: list[CandidateInfo] = []
        for i, mv in enumerate(root_candidates):
            idx = move_to_index(mv)
            policy_logit = float(root_logits[idx]) if 0 <= idx < len(root_logits) and np.isfinite(root_logits[idx]) else -1e9
            child = child_boards[i]
            depth2_cp = float(depth2_values[i] if depth2_values[i] is not None else child_evals[i][0])
            san = board.san(mv)
            infos.append(
                CandidateInfo(
                    move=mv,
                    san=san,
                    policy_rank=policy_rank_by_move.get(mv, 999),
                    policy_logit=policy_logit,
                    flags=move_flags(board, mv),
                    static_cp=float(child_evals[i][0]),
                    depth2_cp=depth2_cp,
                    side_score=cp_for_side(depth2_cp, root_turn),
                    best_reply=best_replies[i],
                    material_after=material_cp_white(child),
                    conversion_score=self.conversion_score(board, mv, depth2_cp, root_turn),
                    is_played=(played_move is not None and mv == played_move),
                )
            )

        infos.sort(key=lambda item: item.side_score, reverse=True)
        best_side = infos[0].side_score
        for rank, info in enumerate(infos, start=1):
            info.rank = rank
            info.loss_cp = best_side - info.side_score

        played_san = board.san(played_move) if played_move is not None and played_move in board.legal_moves else ""
        summary = self.explain(board, root_value, infos, played_move, played_san)
        return PositionAnalysis(
            board_fen=board.fen(),
            side_to_move=root_turn,
            root_value_cp=float(root_value),
            played_move=played_move,
            played_san=played_san,
            candidates=infos,
            summary=summary,
            elapsed=time.perf_counter() - start,
        )

    def explain(
        self,
        board: chess.Board,
        root_value: float,
        infos: list[CandidateInfo],
        played_move: Optional[chess.Move],
        played_san: str,
    ) -> str:
        side = "White" if board.turn == chess.WHITE else "Black"
        side_eval = cp_for_side(root_value, board.turn)
        best = infos[0] if infos else None
        played = next((item for item in infos if item.is_played), None)

        lines = []
        lines.append(f"Side to move: {side}")
        lines.append(f"Static value before move: {root_value:+.1f} cp from White POV ({side} sees {side_eval:+.1f} cp).")
        lines.append("")

        if best is None:
            lines.append("No candidates were available.")
            return "\n".join(lines)

        if played is None:
            lines.append(f"No played move was supplied. Top diagnostic choice: {best.san} ({best.depth2_cp:+.1f} cp White POV).")
        else:
            lines.append(f"Played move: {played_san}")
            lines.append(f"Diagnostic best: {best.san}")
            lines.append(f"Played rank: #{played.rank} of {len(infos)}; estimated loss vs best: {played.loss_cp:.1f} cp for the side to move.")
            lines.append(f"Played policy rank: #{played.policy_rank}; best policy rank: #{best.policy_rank}.")
            if played.best_reply:
                lines.append(f"Model reply check: after {played.san}, opponent's best sampled reply was {played.best_reply}.")
            if played.loss_cp <= 25:
                lines.append("Interpretation: the played move was basically tied with the best sampled move.")
            elif played.loss_cp <= 90:
                lines.append("Interpretation: the played move was plausible but the diagnostic search preferred another line.")
            else:
                lines.append("Interpretation: this is a candidate turning point; the sampled reply-aware eval disliked the played move.")

            if played.policy_rank <= 3 and played.rank > 3:
                lines.append("This looks like a policy/eval disagreement: the policy liked the move, but replies made it less attractive.")
            if played.policy_rank > 8 and played.rank <= 3:
                lines.append("This looks like a search rescue: the policy did not naturally like it, but reply-aware eval did.")
            if "repeat" in played.flags and side_eval > 120:
                lines.append("Repetition warning: this move repeats while the side to move appears better.")

        enemy_passer = passed_pawn_danger(board, not board.turn)
        own_passer = passed_pawn_danger(board, board.turn)
        if enemy_passer >= 2.0:
            lines.append(f"Passed-pawn warning: opponent passer danger is high ({enemy_passer:.1f}).")
        if own_passer >= 2.0:
            lines.append(f"Own passer note: side to move has dangerous passer potential ({own_passer:.1f}).")

        if side_eval >= 180:
            conversion_best = max(infos, key=lambda item: item.conversion_score)
            lines.append("")
            lines.append("Conversion mode should matter here because the side to move appears ahead.")
            lines.append(f"Cleanest sampled conversion move: {conversion_best.san} (conversion score {conversion_best.conversion_score:+.1f}).")
            if played is not None and played.move != conversion_best.move and played.loss_cp <= 90:
                lines.append("A practical engine might choose the cleaner conversion move even if raw eval is close.")

        lines.append("")
        lines.append("Top sampled candidates:")
        for info in infos[:5]:
            marker = "played" if info.is_played else ""
            lines.append(
                f"{info.rank}. {info.san:8s} depth2 {info.depth2_cp:+7.1f} cp, "
                f"side {info.side_score:+7.1f}, policy #{info.policy_rank}, reply {info.best_reply or '-'} {marker}"
            )
        return "\n".join(lines)

    def analyze_game(
        self,
        initial_board: chess.Board,
        moves: list[chess.Move],
        model_side: Optional[chess.Color],
        root_top_n: int,
        child_top_n: int,
        progress=None,
    ) -> str:
        board = initial_board.copy(stack=True)
        rows = []
        total = 0
        for ply, mv in enumerate(moves, start=1):
            should_analyze = model_side is None or board.turn == model_side
            san = board.san(mv)
            if should_analyze:
                total += 1
                analysis = self.analyze_position(board, mv, root_top_n=root_top_n, child_top_n=child_top_n)
                played = next((item for item in analysis.candidates if item.is_played), None)
                best = analysis.candidates[0] if analysis.candidates else None
                if played is not None and best is not None:
                    rows.append((played.loss_cp, ply, san, played, best, analysis.root_value_cp, board.turn))
            board.push(mv)
            if progress:
                progress(ply, len(moves))

        rows.sort(key=lambda item: item[0], reverse=True)
        side_label = "both sides" if model_side is None else ("White" if model_side == chess.WHITE else "Black")
        lines = [f"Analyzed {total} decisions for {side_label}.", ""]
        lines.append("Largest diagnostic disagreements:")
        for loss, ply, san, played, best, root_value, turn in rows[:20]:
            move_no = (ply + 1) // 2
            prefix = f"{move_no}." if turn == chess.WHITE else f"{move_no}..."
            side = "White" if turn == chess.WHITE else "Black"
            lines.append(
                f"{prefix} {san:8s} | {side:5s} | loss {loss:6.1f} cp | "
                f"best {best.san:8s} | played rank #{played.rank}, policy #{played.policy_rank} | "
                f"before {root_value:+.1f} cp"
            )
        lines.append("")
        lines.append("Treat this as a model diagnostic, not a Stockfish verdict. It tells you where this model/search stack disagreed with the played move.")
        return "\n".join(lines)


class DiagnosticApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dual Model Chess Diagnostics")
        self.geometry("1240x780")
        self.minsize(1060, 660)

        self.analyzer: Optional[ModelDiagnostics] = None
        self.work_queue: queue.Queue = queue.Queue()
        self.current_game: Optional[chess.pgn.Game] = None
        self.initial_board = chess.Board()
        self.moves: list[chess.Move] = []
        self.sans: list[str] = []
        self.positions: list[chess.Board] = [chess.Board()]
        self.current_ply = 0
        self.last_analysis: Optional[PositionAnalysis] = None
        self.analysis_arrow: Optional[chess.Move] = None

        self.weights_var = tk.StringVar(value=str(DEFAULT_WEIGHTS))
        self.status_var = tk.StringVar(value="Load a PGN and wait for the model to load.")
        self.flip_var = tk.BooleanVar(value=False)
        self.model_side_var = tk.StringVar(value="Auto")
        self.root_top_var = tk.IntVar(value=12)
        self.child_top_var = tk.IntVar(value=6)

        self.build_ui()
        self.after(100, self.poll_queue)
        self.start_model_load()

    def build_ui(self):
        top = ttk.Frame(self, padding=(8, 6))
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Weights").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.weights_var, width=62).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Reload Model", command=self.start_model_load).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load PGN", command=self.load_pgn_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="Flip", variable=self.flip_var, command=self.refresh_board).pack(side=tk.LEFT, padx=8)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        middle = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=0)
        main.add(middle, weight=1)
        main.add(right, weight=2)

        self.canvas = tk.Canvas(left, width=512, height=512, bg="#f0d9b5", highlightthickness=0)
        self.canvas.pack()

        nav = ttk.Frame(left)
        nav.pack(fill=tk.X, pady=8)
        ttk.Button(nav, text="|<", width=4, command=lambda: self.goto_ply(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="<", width=4, command=lambda: self.goto_ply(max(0, self.current_ply - 1))).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text=">", width=4, command=lambda: self.goto_ply(min(len(self.moves), self.current_ply + 1))).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text=">|", width=4, command=lambda: self.goto_ply(len(self.moves))).pack(side=tk.LEFT, padx=2)

        self.position_label = ttk.Label(left, text="")
        self.position_label.pack(fill=tk.X)

        opts = ttk.LabelFrame(left, text="Diagnostic settings", padding=8)
        opts.pack(fill=tk.X, pady=8)
        ttk.Label(opts, text="Root moves").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(opts, from_=4, to=40, textvariable=self.root_top_var, width=5).grid(row=0, column=1, sticky="w")
        ttk.Label(opts, text="Reply moves").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(opts, from_=2, to=24, textvariable=self.child_top_var, width=5).grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="Model side").grid(row=2, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.model_side_var, values=["Auto", "White", "Black", "Both"], width=8, state="readonly").grid(row=2, column=1, sticky="w")

        ttk.Button(left, text="Analyze Selected Move", command=self.analyze_selected_move).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Analyze Current Position", command=self.analyze_current_position).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Analyze All Model Moves", command=self.analyze_all_model_moves).pack(fill=tk.X, pady=3)

        ttk.Label(middle, text="Moves").pack(anchor="w")
        self.move_list = tk.Listbox(middle, exportselection=False, width=28)
        self.move_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(middle, orient=tk.VERTICAL, command=self.move_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.move_list.configure(yscrollcommand=scroll.set)
        self.move_list.bind("<<ListboxSelect>>", self.on_move_select)

        ttk.Label(right, text="Candidate diagnostics").pack(anchor="w")
        columns = ("rank", "move", "flags", "policy", "static", "depth2", "side", "loss", "reply", "conv")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=12)
        widths = {
            "rank": 48,
            "move": 80,
            "flags": 120,
            "policy": 70,
            "static": 82,
            "depth2": 82,
            "side": 82,
            "loss": 70,
            "reply": 90,
            "conv": 70,
        }
        headings = {
            "rank": "#",
            "move": "Move",
            "flags": "Flags",
            "policy": "Policy",
            "static": "Static",
            "depth2": "Depth2",
            "side": "Side",
            "loss": "Loss",
            "reply": "Reply",
            "conv": "Convert",
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor=tk.CENTER)
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", self.on_candidate_select)

        ttk.Label(right, text="Explanation").pack(anchor="w", pady=(8, 0))
        self.summary = tk.Text(right, wrap=tk.WORD, height=22)
        self.summary.pack(fill=tk.BOTH, expand=True)

        self.refresh_board()

    def set_status(self, text: str):
        self.status_var.set(text)
        self.update_idletasks()

    def start_model_load(self):
        weights = self.weights_var.get().strip()
        self.set_status("Loading model...")

        def worker():
            try:
                analyzer = ModelDiagnostics(weights, root_top_n=self.root_top_var.get(), child_top_n=self.child_top_var.get())
                self.work_queue.put(("model_loaded", analyzer, None))
            except Exception as exc:
                self.work_queue.put(("error", None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def poll_queue(self):
        try:
            while True:
                kind, payload, err = self.work_queue.get_nowait()
                if kind == "model_loaded":
                    self.analyzer = payload
                    meta = self.analyzer.meta
                    epoch = meta.get("epoch", "?") if isinstance(meta, dict) else "?"
                    self.set_status(f"Model loaded on {self.analyzer.device}; checkpoint epoch {epoch}.")
                elif kind == "analysis":
                    self.last_analysis = payload
                    self.populate_analysis(payload)
                    self.set_status(f"Analysis done in {payload.elapsed:.2f}s.")
                elif kind == "report":
                    self.write_summary(payload)
                    self.set_status("Game report done.")
                elif kind == "progress":
                    self.set_status(payload)
                elif kind == "error":
                    self.set_status("Error.")
                    messagebox.showerror("Error", str(err))
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    def load_pgn_dialog(self):
        initial = DEFAULT_PGN_DIR if DEFAULT_PGN_DIR.exists() else DOWNLOADS
        path = filedialog.askopenfilename(
            title="Open PGN",
            initialdir=str(initial),
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")],
        )
        if path:
            self.load_pgn(Path(path))

    def load_pgn(self, path: Path):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                raise ValueError("No game found in PGN.")
        except Exception as exc:
            messagebox.showerror("Could not load PGN", str(exc))
            return

        self.current_game = game
        self.initial_board = game.board()
        self.moves = list(game.mainline_moves())
        self.positions = [self.initial_board.copy(stack=True)]
        self.sans = []
        board = self.initial_board.copy(stack=True)
        for mv in self.moves:
            self.sans.append(board.san(mv))
            board.push(mv)
            self.positions.append(board.copy(stack=True))

        self.move_list.delete(0, tk.END)
        for i, san in enumerate(self.sans, start=1):
            move_no = (i + 1) // 2
            prefix = f"{move_no}. " if i % 2 == 1 else f"{move_no}... "
            self.move_list.insert(tk.END, prefix + san)

        white = game.headers.get("White", "?")
        black = game.headers.get("Black", "?")
        result = game.headers.get("Result", "*")
        self.write_summary(f"Loaded {path}\n\n{white} vs {black}  {result}\nMoves: {len(self.moves)} plies")
        self.goto_ply(0)
        self.set_status(f"Loaded PGN: {path.name}")

    def current_board(self) -> chess.Board:
        return self.positions[self.current_ply].copy(stack=True)

    def goto_ply(self, ply: int):
        if not self.positions:
            return
        self.current_ply = max(0, min(int(ply), len(self.moves)))
        self.move_list.selection_clear(0, tk.END)
        if self.current_ply > 0:
            idx = self.current_ply - 1
            self.move_list.selection_set(idx)
            self.move_list.see(idx)
        self.analysis_arrow = None
        self.refresh_board()

    def on_move_select(self, _event=None):
        sel = self.move_list.curselection()
        if sel:
            self.goto_ply(sel[0] + 1)

    def on_candidate_select(self, _event=None):
        if self.last_analysis is None:
            return
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(self.tree.item(sel[0], "values")[0]) - 1
        if 0 <= idx < len(self.last_analysis.candidates):
            sorted_candidates = sorted(self.last_analysis.candidates, key=lambda item: item.rank)
            self.analysis_arrow = sorted_candidates[idx].move
            self.refresh_board()

    def refresh_board(self):
        board = self.current_board() if self.positions else chess.Board()
        last_move = self.moves[self.current_ply - 1] if self.current_ply > 0 and self.current_ply - 1 < len(self.moves) else None
        self.draw_board(board, last_move=last_move, arrow=self.analysis_arrow)
        side = "White" if board.turn == chess.WHITE else "Black"
        self.position_label.config(text=f"Ply {self.current_ply}/{len(self.moves)} | {side} to move | {board.fen()}")

    def square_rect(self, square: chess.Square) -> tuple[int, int, int, int]:
        size = 64
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if self.flip_var.get():
            col = 7 - file_idx
            row = rank_idx
        else:
            col = file_idx
            row = 7 - rank_idx
        x0, y0 = col * size, row * size
        return x0, y0, x0 + size, y0 + size

    def square_center(self, square: chess.Square) -> tuple[int, int]:
        x0, y0, x1, y1 = self.square_rect(square)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def draw_board(self, board: chess.Board, last_move: Optional[chess.Move] = None, arrow: Optional[chess.Move] = None):
        self.canvas.delete("all")
        light = "#f0d9b5"
        dark = "#b58863"
        last = "#f6e27f"
        arrow_color = "#2e7d32"
        played_color = "#d98c27"

        for rank in range(8):
            for file_idx in range(8):
                sq = chess.square(file_idx, rank)
                x0, y0, x1, y1 = self.square_rect(sq)
                color = light if (file_idx + rank) % 2 == 0 else dark
                if last_move and sq in (last_move.from_square, last_move.to_square):
                    color = last
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)

        if last_move is not None:
            x0, y0 = self.square_center(last_move.from_square)
            x1, y1 = self.square_center(last_move.to_square)
            self.canvas.create_line(x0, y0, x1, y1, fill=played_color, width=4, arrow=tk.LAST)

        if arrow is not None:
            x0, y0 = self.square_center(arrow.from_square)
            x1, y1 = self.square_center(arrow.to_square)
            self.canvas.create_line(x0, y0, x1, y1, fill=arrow_color, width=5, arrow=tk.LAST)

        for sq, piece in board.piece_map().items():
            x, y = self.square_center(sq)
            symbol = UNICODE_PIECES[piece.symbol()]
            fill = "#f8f8f8" if piece.color == chess.WHITE else "#111111"
            outline = "#111111" if piece.color == chess.WHITE else "#f2f2f2"
            self.canvas.create_text(x + 1, y + 1, text=symbol, font=("Segoe UI Symbol", 38), fill=outline)
            self.canvas.create_text(x, y, text=symbol, font=("Segoe UI Symbol", 38), fill=fill)

        files = "abcdefgh"
        for i in range(8):
            file_label = files[7 - i] if self.flip_var.get() else files[i]
            rank_label = str(i + 1) if self.flip_var.get() else str(8 - i)
            self.canvas.create_text(i * 64 + 6, 504, text=file_label, anchor="sw", fill="#333", font=("Segoe UI", 8))
            self.canvas.create_text(4, i * 64 + 4, text=rank_label, anchor="nw", fill="#333", font=("Segoe UI", 8))

    def require_analyzer(self) -> Optional[ModelDiagnostics]:
        if self.analyzer is None:
            messagebox.showinfo("Model loading", "The model is not loaded yet.")
            return None
        self.analyzer.root_top_n = self.root_top_var.get()
        self.analyzer.child_top_n = self.child_top_var.get()
        return self.analyzer

    def analyze_selected_move(self):
        analyzer = self.require_analyzer()
        if analyzer is None:
            return
        if self.current_ply == 0:
            board = self.positions[0].copy(stack=True)
            played = None
        else:
            board = self.positions[self.current_ply - 1].copy(stack=True)
            played = self.moves[self.current_ply - 1]
        self.set_status("Analyzing selected move...")

        def worker():
            try:
                result = analyzer.analyze_position(board, played, self.root_top_var.get(), self.child_top_var.get())
                self.work_queue.put(("analysis", result, None))
            except Exception as exc:
                self.work_queue.put(("error", None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def analyze_current_position(self):
        analyzer = self.require_analyzer()
        if analyzer is None:
            return
        board = self.current_board()
        self.set_status("Analyzing current position...")

        def worker():
            try:
                result = analyzer.analyze_position(board, None, self.root_top_var.get(), self.child_top_var.get())
                self.work_queue.put(("analysis", result, None))
            except Exception as exc:
                self.work_queue.put(("error", None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def infer_model_side(self) -> Optional[chess.Color]:
        choice = self.model_side_var.get()
        if choice == "White":
            return chess.WHITE
        if choice == "Black":
            return chess.BLACK
        if choice == "Both":
            return None
        if self.current_game is None:
            return None
        white = self.current_game.headers.get("White", "").lower()
        black = self.current_game.headers.get("Black", "").lower()
        white_model = any(token in white for token in ("dual", "model", "nn"))
        black_model = any(token in black for token in ("dual", "model", "nn"))
        if white_model and not black_model:
            return chess.WHITE
        if black_model and not white_model:
            return chess.BLACK
        return None

    def analyze_all_model_moves(self):
        analyzer = self.require_analyzer()
        if analyzer is None:
            return
        if not self.moves:
            messagebox.showinfo("No PGN", "Load a PGN first.")
            return
        initial = self.initial_board.copy(stack=True)
        moves = list(self.moves)
        model_side = self.infer_model_side()
        root_top = self.root_top_var.get()
        child_top = self.child_top_var.get()
        self.set_status("Analyzing game...")

        def progress(ply: int, total: int):
            if ply % 2 == 0 or ply == total:
                self.work_queue.put(("progress", f"Analyzing game: {ply}/{total} plies...", None))

        def worker():
            try:
                report = analyzer.analyze_game(initial, moves, model_side, root_top, child_top, progress=progress)
                self.work_queue.put(("report", report, None))
            except Exception as exc:
                self.work_queue.put(("error", None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def populate_analysis(self, analysis: PositionAnalysis):
        self.write_summary(analysis.summary)
        self.tree.delete(*self.tree.get_children())
        for info in sorted(analysis.candidates, key=lambda item: item.rank):
            move = info.san + (" *" if info.is_played else "")
            self.tree.insert(
                "",
                tk.END,
                values=(
                    info.rank,
                    move,
                    info.flags,
                    f"#{info.policy_rank} {info.policy_logit:+.2f}",
                    f"{info.static_cp:+.1f}",
                    f"{info.depth2_cp:+.1f}",
                    f"{info.side_score:+.1f}",
                    f"{info.loss_cp:.1f}",
                    info.best_reply,
                    f"{info.conversion_score:+.1f}",
                ),
            )
        self.analysis_arrow = analysis.played_move or (analysis.candidates[0].move if analysis.candidates else None)
        self.refresh_board()

    def write_summary(self, text: str):
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, text)


def main():
    app = DiagnosticApp()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            app.load_pgn(path)
    app.mainloop()


if __name__ == "__main__":
    main()
