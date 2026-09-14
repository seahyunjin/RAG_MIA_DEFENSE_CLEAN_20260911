"""Linear and small-MLP Session-DRO query scorers for Experiment 39."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from .session_subset_loss import session_pair_loss


@dataclass(frozen=True)
class SessionPair:
    positive: np.ndarray
    positive_retriever: str
    negative: np.ndarray
    negative_retriever: str
    subset_size: int
    positive_session_id: str = ""
    negative_session_id: str = ""


@dataclass(frozen=True)
class SessionDROLinearModel:
    coefficients: np.ndarray
    intercept: float
    normalization: Mapping[str, tuple[np.ndarray, np.ndarray]]
    training_loss: float
    bce_loss: float
    session_loss: float
    seed: int
    epochs: int
    objective: str

    def score(self, matrix: np.ndarray, retriever: str) -> np.ndarray:
        mean, scale = self.normalization[str(retriever)]
        x = (np.asarray(matrix, dtype=float) - mean) / scale
        return x @ self.coefficients + self.intercept


@dataclass
class SessionDROMLPModel:
    state_dict: dict[str, np.ndarray]
    normalization: Mapping[str, tuple[np.ndarray, np.ndarray]]
    training_loss: float
    bce_loss: float
    session_loss: float
    seed: int
    epochs: int
    objective: str

    @staticmethod
    def network() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(11, 16), torch.nn.GELU(),
            torch.nn.Linear(16, 8), torch.nn.GELU(),
            torch.nn.Linear(8, 1),
        ).double()

    def torch_model(self) -> torch.nn.Module:
        model = self.network()
        model.load_state_dict({name: torch.tensor(value) for name, value in self.state_dict.items()})
        model.eval()
        return model

    def score(self, matrix: np.ndarray, retriever: str) -> np.ndarray:
        mean, scale = self.normalization[str(retriever)]
        x = (np.asarray(matrix, dtype=float) - mean) / scale
        with torch.no_grad():
            return self.torch_model()(torch.tensor(x, dtype=torch.float64)).numpy().reshape(-1)


def fit_normalization(features: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for retriever, matrix in features.items():
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != 11 or not np.isfinite(values).all():
            raise ValueError("normalization input must be finite N x 11")
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        output[str(retriever)] = (mean, np.where(scale == 0.0, 1.0, scale))
    return output


def _balanced_query_rows(
    member: Mapping[str, np.ndarray], normal: Mapping[str, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retrievers = sorted(set(member) & set(normal))
    if not retrievers:
        raise ValueError("member and normal retrievers do not overlap")
    target = min(min(len(member[r]), len(normal[r])) for r in retrievers)
    if target < 1:
        raise ValueError("empty retriever/class cell")
    rng = np.random.default_rng(seed)
    xs, ys, rs = [], [], []
    for retriever in retrievers:
        for values, label in ((member[retriever], 1.0), (normal[retriever], 0.0)):
            chosen = rng.choice(len(values), size=target, replace=False)
            xs.append(np.asarray(values)[chosen])
            ys.append(np.full(target, label))
            rs.extend([retriever] * target)
    return np.vstack(xs), np.concatenate(ys), np.asarray(rs, dtype=object)


def _standardize_rows(
    matrix: np.ndarray, retrievers: Sequence[str], normalization: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    return np.vstack([
        (row - normalization[str(retriever)][0]) / normalization[str(retriever)][1]
        for row, retriever in zip(np.asarray(matrix, dtype=float), retrievers)
    ])


def _pair_tensors(
    pairs: Sequence[SessionPair], normalization: Mapping[str, tuple[np.ndarray, np.ndarray]], cap_per_cell: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, list[SessionPair]]:
    rng = np.random.default_rng(seed)
    selected = []
    cells = sorted({(p.positive_retriever, int(p.subset_size)) for p in pairs})
    for retriever, size in cells:
        candidates = [p for p in pairs if p.positive_retriever == retriever and int(p.subset_size) == size]
        order = rng.permutation(len(candidates))[: min(cap_per_cell, len(candidates))]
        selected.extend(candidates[int(i)] for i in order)
    if not selected:
        empty = torch.empty((0, 11), dtype=torch.float64)
        groups = torch.empty((0,), dtype=torch.int64)
        return empty, groups, empty, groups, 0, []
    positive_rows, negative_rows, positive_groups, negative_groups = [], [], [], []
    for group, pair in enumerate(selected):
        p_mean, p_scale = normalization[pair.positive_retriever]
        n_mean, n_scale = normalization[pair.negative_retriever]
        positive_rows.append((np.asarray(pair.positive) - p_mean) / p_scale)
        negative_rows.append((np.asarray(pair.negative) - n_mean) / n_scale)
        positive_groups.extend([group] * len(pair.positive))
        negative_groups.extend([group] * len(pair.negative))
    return (
        torch.tensor(np.vstack(positive_rows), dtype=torch.float64),
        torch.tensor(positive_groups, dtype=torch.int64),
        torch.tensor(np.vstack(negative_rows), dtype=torch.float64),
        torch.tensor(negative_groups, dtype=torch.int64),
        len(selected), selected,
    )


def train_session_dro_linear(
    *,
    query_member: Mapping[str, np.ndarray],
    query_normal: Mapping[str, np.ndarray],
    normalization_source: Mapping[str, np.ndarray],
    session_pairs: Sequence[SessionPair],
    seed: int,
    objective: str = "combined",
    epochs: int = 300,
    learning_rate: float = 0.03,
    feature_mask: np.ndarray | None = None,
    cap_pairs_per_retriever_size: int = 128,
) -> SessionDROLinearModel:
    if objective not in {"combined", "query_bce_only", "session_loss_only"}:
        raise ValueError("unsupported Session-DRO objective")
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    mask = np.ones(11) if feature_mask is None else np.asarray(feature_mask, dtype=float)
    if mask.shape != (11,):
        raise ValueError("feature mask must contain 11 entries")
    normalized_source = {k: np.asarray(v) * mask for k, v in normalization_source.items()}
    normalization = fit_normalization(normalized_source)
    member = {k: np.asarray(v) * mask for k, v in query_member.items()}
    normal = {k: np.asarray(v) * mask for k, v in query_normal.items()}
    x, y, r = _balanced_query_rows(member, normal, seed)
    x = _standardize_rows(x, r, normalization)
    tensor_x = torch.tensor(x, dtype=torch.float64)
    tensor_y = torch.tensor(y, dtype=torch.float64)
    masked_pairs = [SessionPair(p.positive * mask, p.positive_retriever, p.negative * mask,
                                p.negative_retriever, p.subset_size,
                                p.positive_session_id, p.negative_session_id) for p in session_pairs]
    px, pg, nx, ng, pair_count, _ = _pair_tensors(
        masked_pairs, normalization, cap_pairs_per_retriever_size, seed + 39
    )
    if objective != "query_bce_only" and pair_count == 0:
        raise ValueError("session objective requires session pairs")
    weight = torch.zeros(11, dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weight, intercept], lr=learning_rate)
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = tensor_x @ weight + intercept
        bce = functional.binary_cross_entropy_with_logits(logits, tensor_y)
        if pair_count:
            session = session_pair_loss(px @ weight + intercept, pg, nx @ weight + intercept, ng, pair_count)
        else:
            session = torch.zeros((), dtype=torch.float64)
        loss = bce if objective == "query_bce_only" else session if objective == "session_loss_only" else .5 * bce + .5 * session
        loss.backward(); optimizer.step()
    return SessionDROLinearModel(
        weight.detach().numpy(), float(intercept.detach()), normalization,
        float(loss.detach()), float(bce.detach()), float(session.detach()),
        int(seed), int(epochs), objective,
    )


def train_session_dro_mlp(
    *,
    query_member: Mapping[str, np.ndarray],
    query_normal: Mapping[str, np.ndarray],
    normalization_source: Mapping[str, np.ndarray],
    session_pairs: Sequence[SessionPair],
    seed: int,
    epochs: int = 250,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    cap_pairs_per_retriever_size: int = 128,
) -> SessionDROMLPModel:
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    normalization = fit_normalization(normalization_source)
    x, y, r = _balanced_query_rows(query_member, query_normal, seed)
    x = _standardize_rows(x, r, normalization)
    tx = torch.tensor(x, dtype=torch.float64); ty = torch.tensor(y, dtype=torch.float64)
    px, pg, nx, ng, pair_count, _ = _pair_tensors(
        session_pairs, normalization, cap_pairs_per_retriever_size, seed + 139
    )
    if not pair_count:
        raise ValueError("MLP session objective requires session pairs")
    model = SessionDROMLPModel.network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(tx).reshape(-1)
        bce = functional.binary_cross_entropy_with_logits(logits, ty)
        session = session_pair_loss(model(px).reshape(-1), pg, model(nx).reshape(-1), ng, pair_count)
        loss = .5 * bce + .5 * session
        loss.backward(); optimizer.step()
    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    return SessionDROMLPModel(
        state, normalization, float(loss.detach()), float(bce.detach()),
        float(session.detach()), int(seed), int(epochs), "combined",
    )


def retriever_gradient(
    model: SessionDROLinearModel,
    *,
    retriever: str,
    query_member: np.ndarray,
    query_normal: np.ndarray,
    session_pairs: Sequence[SessionPair],
) -> np.ndarray:
    """Gradient of the fixed combined objective for one retriever."""

    mean, scale = model.normalization[retriever]
    target = min(len(query_member), len(query_normal))
    x = np.vstack((query_member[:target], query_normal[:target]))
    y = np.r_[np.ones(target), np.zeros(target)]
    tx = torch.tensor((x - mean) / scale, dtype=torch.float64)
    ty = torch.tensor(y, dtype=torch.float64)
    weight = torch.tensor(model.coefficients, dtype=torch.float64, requires_grad=True)
    intercept = torch.tensor([model.intercept], dtype=torch.float64, requires_grad=True)
    bce = functional.binary_cross_entropy_with_logits(tx @ weight + intercept, ty)
    chosen = [p for p in session_pairs if p.positive_retriever == retriever and p.negative_retriever == retriever][:384]
    px, pg, nx, ng, count, _ = _pair_tensors(chosen, model.normalization, 128, model.seed + 239)
    session = session_pair_loss(px @ weight + intercept, pg, nx @ weight + intercept, ng, count)
    (.5 * bce + .5 * session).backward()
    return np.r_[weight.grad.detach().numpy(), float(intercept.grad.detach())]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float); b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0
