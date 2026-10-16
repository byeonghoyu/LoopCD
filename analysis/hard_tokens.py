from __future__ import annotations

import numpy as np

EPS_DEFAULT = 1.5
_TINY = 1e-12


def fixed_target_margin(logits_per_iter: np.ndarray, winner: int | None = None) -> np.ndarray:
    Z = np.asarray(logits_per_iter, dtype=np.float64)
    if winner is None:
        winner = int(Z[-1].argmax())
    z_y = Z[:, winner].copy()
    Z_other = Z.copy()
    Z_other[:, winner] = -np.inf
    return z_y - Z_other.max(axis=1)


def margin_from_probs(probs_per_iter: np.ndarray, winner_row: int) -> np.ndarray:
    P = np.asarray(probs_per_iter, dtype=np.float64)
    L = np.log(np.maximum(P, _TINY))
    L_other = L.copy()
    L_other[winner_row] = -np.inf
    return L[winner_row] - L_other.max(axis=0)


def confidence_drop(margin: np.ndarray) -> float:
    m = np.asarray(margin, dtype=np.float64)
    return float(m.max() - m[-1])


def peak_iteration(margin: np.ndarray) -> int:
    return int(np.asarray(margin).argmax())


def is_hard(drop: float, eps: float = EPS_DEFAULT) -> bool:
    return float(drop) >= eps


def classify(drops, eps: float = EPS_DEFAULT) -> list[str]:
    return ["hard" if is_hard(d, eps) else "easy" for d in drops]
