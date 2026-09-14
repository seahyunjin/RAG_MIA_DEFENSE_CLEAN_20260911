"""Balanced linear Robust-LDF training with a fixed 1:1 BCE/ranking loss."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class RobustLinearModel:
    coefficients: np.ndarray
    intercept: float
    normalization: Mapping[str, tuple[np.ndarray, np.ndarray]]
    training_loss: float
    bce_loss: float
    ranking_loss: float
    epochs: int
    seed: int

    def score(self, matrix: np.ndarray, retriever: str) -> np.ndarray:
        mean, scale = self.normalization[str(retriever)]
        standardized = (np.asarray(matrix, dtype=float) - mean) / scale
        return standardized @ self.coefficients + self.intercept


def fit_normalization(features_by_retriever: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for retriever, values in features_by_retriever.items():
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 11 or not np.isfinite(matrix).all():
            raise ValueError("each retriever must provide a finite N x 11 training matrix")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        output[str(retriever)] = (mean, np.where(scale == 0.0, 1.0, scale))
    return output


def balanced_four_group_matrix(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray, Sequence[str]]], *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("general_member", "hard_member", "general_normal", "hard_normal")
    if any(name not in groups for name in required):
        raise ValueError(f"balanced training requires {required}")
    rng = np.random.default_rng(seed)
    retriever_names=sorted(set.intersection(*(set(map(str,groups[name][2])) for name in required)))
    if not retriever_names:raise ValueError("no retriever is represented in every balanced group")
    target=min(sum(str(value)==retriever for value in groups[name][2]) for name in required for retriever in retriever_names)
    if target<1:raise ValueError("balanced group/retriever cells cannot be empty")
    x_parts=[];y_parts=[];retrievers=[]
    for name in required:
        x, y, r = groups[name]
        r_array=np.asarray(r,dtype=object)
        for retriever in retriever_names:
            candidates=np.flatnonzero(r_array.astype(str)==retriever);selected=rng.choice(candidates,size=target,replace=False);x_parts.append(np.asarray(x)[selected]);y_parts.append(np.asarray(y)[selected]);retrievers.extend(r_array[selected].tolist())
    return np.vstack(x_parts), np.concatenate(y_parts).astype(float), np.asarray(retrievers,dtype=object)


def paired_subset_indices(positive_sizes: Sequence[int], negative_sizes: Sequence[int], *, seed: int, cap: int = 256) -> list[tuple[int,int,int]]:
    """Pair hard member/normal subsets only when their query budgets match."""
    rng=np.random.default_rng(seed);pairs=[]
    for subset_size in sorted(set(map(int,positive_sizes))&set(map(int,negative_sizes))):
        p_indices=[i for i,value in enumerate(positive_sizes) if int(value)==subset_size];n_indices=[i for i,value in enumerate(negative_sizes) if int(value)==subset_size];count=min(len(p_indices),len(n_indices));p_order=rng.permutation(p_indices)[:count];n_order=rng.permutation(n_indices)[:count];pairs.extend((int(p),int(n),subset_size) for p,n in zip(p_order,n_order))
    if len(pairs)>cap:
        mandatory=[];remaining=[]
        for size in sorted({item[2] for item in pairs}):
            group=[item for item in pairs if item[2]==size];mandatory.append(group[0]);remaining.extend(group[1:])
        chosen=rng.permutation(len(remaining))[:cap-len(mandatory)];pairs=mandatory+[remaining[int(index)] for index in chosen]
    return pairs


def train_robust_linear(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray, Sequence[str]]],
    *,
    normalization_source: Mapping[str, np.ndarray],
    positive_subsets: Sequence[tuple[np.ndarray, str] | tuple[np.ndarray, str, int]],
    negative_subsets: Sequence[tuple[np.ndarray, str] | tuple[np.ndarray, str, int]],
    include_ranking: bool,
    seed: int,
    epochs: int = 500,
    learning_rate: float = 0.03,
) -> RobustLinearModel:
    """Fit one shared scorer; retriever identity is never an input feature."""

    torch.manual_seed(int(seed));np.random.seed(int(seed))
    normalization = fit_normalization(normalization_source)
    x,y,retrievers = balanced_four_group_matrix(groups,seed=seed)
    standardized = np.vstack([
        (row-normalization[str(retriever)][0])/normalization[str(retriever)][1]
        for row,retriever in zip(x,retrievers)
    ])
    tensor_x=torch.tensor(standardized,dtype=torch.float64)
    tensor_y=torch.tensor(y,dtype=torch.float64)
    weight=torch.zeros(11,dtype=torch.float64,requires_grad=True)
    intercept=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([weight,intercept],lr=learning_rate)
    # A deterministic cap bounds full-batch research training cost; it is a
    # compute limit fixed independently of validation performance.
    def unpack(item):
        matrix,retriever,*size=item
        return np.asarray(matrix),str(retriever),int(size[0]) if size else len(matrix)
    positives=[unpack(item) for item in positive_subsets];negatives=[unpack(item) for item in negative_subsets]
    pairs=paired_subset_indices([item[2] for item in positives],[item[2] for item in negatives],seed=seed+91,cap=256)
    pair_count=len(pairs)
    if pair_count:
        pos_rows=[];pos_groups=[];neg_rows=[];neg_groups=[]
        for group,(p_index,n_index,subset_size) in enumerate(pairs):
            p_matrix,p_retriever,p_size=positives[p_index];n_matrix,n_retriever,n_size=negatives[n_index]
            if p_size!=n_size:raise AssertionError("ranking pairs must have the same subset size")
            p_mean,p_scale=normalization[str(p_retriever)];n_mean,n_scale=normalization[str(n_retriever)]
            p_standard=(np.asarray(p_matrix)-p_mean)/p_scale;n_standard=(np.asarray(n_matrix)-n_mean)/n_scale
            pos_rows.append(p_standard);neg_rows.append(n_standard);pos_groups.extend([group]*len(p_standard));neg_groups.extend([group]*len(n_standard))
        pair_pos_x=torch.tensor(np.vstack(pos_rows),dtype=torch.float64);pair_neg_x=torch.tensor(np.vstack(neg_rows),dtype=torch.float64)
        pair_pos_group=torch.tensor(pos_groups,dtype=torch.int64);pair_neg_group=torch.tensor(neg_groups,dtype=torch.int64)
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits=tensor_x@weight+intercept
        bce=functional.binary_cross_entropy_with_logits(logits,tensor_y)
        if include_ranking and pair_count:
            p_logits=pair_pos_x@weight+intercept;n_logits=pair_neg_x@weight+intercept
            p_min=torch.full((pair_count,),float("inf"),dtype=torch.float64).scatter_reduce(0,pair_pos_group,p_logits,reduce="amin",include_self=True)
            n_max=torch.full((pair_count,),-float("inf"),dtype=torch.float64).scatter_reduce(0,pair_neg_group,n_logits,reduce="amax",include_self=True)
            ranking=functional.softplus(n_max-p_min).mean()
            loss=.5*bce+.5*ranking
        else:
            ranking=torch.zeros((),dtype=torch.float64)
            loss=bce
        loss.backward();optimizer.step()
    return RobustLinearModel(
        coefficients=weight.detach().numpy(),intercept=float(intercept.detach().item()),
        normalization=normalization,training_loss=float(loss.detach()),
        bce_loss=float(bce.detach()),ranking_loss=float(ranking.detach()),
        epochs=int(epochs),seed=int(seed),
    )


def weight_cosine(left: RobustLinearModel, right: RobustLinearModel) -> float:
    a=np.asarray(left.coefficients);b=np.asarray(right.coefficients)
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denominator) if denominator else 0.0
