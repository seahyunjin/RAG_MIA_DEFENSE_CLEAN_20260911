"""Vectorized exact ordered-subset white-box attack against frozen Legacy-V16."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
import math

import numpy as np

from .adaptive_union_policy import component_scores
from .persistent_branch_policy import BRANCHES


@dataclass(frozen=True)
class V16OrderedSubsetResult:
    subset_size: int
    ordered_indices: tuple[int, ...]
    terminal_risk: float
    maximum_prefix_risk: float
    terminal_blocked: bool
    ever_blocked: bool
    permutations_evaluated: int
    utility_proxy: float
    actual_proxy_retention: float


def ordered_subset_count(query_count: int = 15, max_size: int = 5) -> int:
    return int(sum(math.factorial(query_count)//math.factorial(query_count-k) for k in range(1,max_size+1)))


def _risk_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.searchsorted(reference,values,side="right")/(len(reference)+1.0)


def _upper_p(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (len(reference)-np.searchsorted(reference,values,side="left")+1.0)/(len(reference)+1.0)


def _cct(p_values: np.ndarray) -> np.ndarray:
    clipped=np.clip(np.asarray(p_values,dtype=float),1e-12,1-1e-12)
    statistic=np.mean(np.tan((.5-clipped)*np.pi),axis=-1)
    combined=.5-np.arctan(statistic)/np.pi
    return -np.log10(np.clip(combined,1e-15,1.0))


class VectorizedLegacyV16:
    """Numerically equivalent V16 evaluator for all ordered subsets of Q15."""

    def __init__(self, frozen_model: object, *, threshold: float) -> None:
        self.model=frozen_model;self.calibration=frozen_model.calibration
        self.reference=frozen_model.reference;self.threshold=float(threshold)
        self.component_reference=component_scores(self.reference,self.reference)

    def _query_arrays(self, rows: Sequence[Mapping[str,object]]) -> dict[str,np.ndarray]:
        top1=[];gaps=[];concentration=[];margin=[];documents=[];mirabel=[]
        for row in rows:
            scores=np.asarray(row["legacy_scores"],dtype=float);one=float(row["legacy_top1"])
            background=scores[1:];spread=float(background.std(ddof=0))
            top1.append(one);gaps.append(one-float(scores[1]));concentration.append((one-float(background.mean()))/max(spread,np.finfo(float).eps));margin.append(float(row["legacy_mirabel_margin"]));documents.append(str(row["legacy_doc_ids"][0]));mirabel.append(float(_risk_percentile(np.asarray([margin[-1]]),self.calibration.mirabel_margin_reference)[0]))
        return {"top1":np.asarray(top1),"gap":np.asarray(gaps),"concentration":np.asarray(concentration),"margin":np.asarray(margin),"document":np.asarray(documents,dtype=object),"mirabel":np.asarray(mirabel)}

    def _early_risk(self, arrays: Mapping[str,np.ndarray], prefixes: np.ndarray, turn: int) -> np.ndarray:
        idx=prefixes[:,:turn];top1=arrays["top1"][idx];docs=arrays["document"][idx]
        same=np.ones(len(idx),dtype=float)
        for position in range(turn):
            same=np.maximum(same,np.sum(docs==docs[:,position,None],axis=1))
        aggregates={
            "mean_top1_similarity":top1.mean(axis=1),
            "mean_top1_top2_gap":arrays["gap"][idx].mean(axis=1),
            "mean_standardized_retrieval_concentration":arrays["concentration"][idx].mean(axis=1),
            "mean_gumbel_margin":arrays["margin"][idx].mean(axis=1),
            "max_top1_similarity":top1.max(axis=1),
            "same_document_concentration":same/turn,
        }
        p=np.column_stack([_upper_p(aggregates[name],self.calibration.feature_references[turn][name]) for name in self.calibration.features])
        raw=-2*np.sum(self.calibration.weights[None,:]*np.log(np.clip(p,1e-12,1.0)),axis=1)
        return _risk_percentile(raw,self.calibration.early_score_references[turn])

    def _rolling_risk(self, arrays: Mapping[str,np.ndarray], prefixes: np.ndarray, turn: int) -> np.ndarray:
        idx=prefixes[:,:turn];order=np.sort(arrays["top1"][idx],axis=1)
        delta=order-self.calibration.set_locations[turn]
        distance=np.einsum("ni,ij,nj->n",delta,self.calibration.set_precisions[turn],delta)
        p_order=_upper_p(distance,self.calibration.set_distance_references[turn])
        # Early raw for the full set (prefix >=5) is calibrated but not exposed
        # as a separate Early branch after Q4.
        top1=arrays["top1"][idx];docs=arrays["document"][idx];same=np.ones(len(idx))
        for position in range(turn):same=np.maximum(same,np.sum(docs==docs[:,position,None],axis=1))
        ag={"mean_top1_similarity":top1.mean(1),"mean_top1_top2_gap":arrays["gap"][idx].mean(1),"mean_standardized_retrieval_concentration":arrays["concentration"][idx].mean(1),"mean_gumbel_margin":arrays["margin"][idx].mean(1),"max_top1_similarity":top1.max(1),"same_document_concentration":same/turn}
        p=np.column_stack([_upper_p(ag[name],self.calibration.feature_references[turn][name]) for name in self.calibration.features])
        early_raw=-2*np.sum(self.calibration.weights[None,:]*np.log(np.clip(p,1e-12,1.0)),axis=1)
        p_early=_upper_p(early_raw,self.calibration.early_score_references[turn])
        raw=_cct(np.column_stack([p_early,p_order]))
        return _risk_percentile(raw,self.calibration.set_cct_references[turn])

    def score_orders(self, rows: Sequence[Mapping[str,object]], orders: np.ndarray) -> np.ndarray:
        orders=np.asarray(orders,dtype=np.int16)
        if orders.ndim!=2 or orders.shape[1]<1 or orders.shape[1]>5:raise ValueError("orders must be N x Q1..Q5")
        arrays=self._query_arrays(rows);n,k=orders.shape
        mirabel=np.zeros(n);early=np.zeros(n);rolling=np.zeros(n);sticky=np.zeros(n)
        for turn in range(1,k+1):
            current=orders[:,turn-1]
            mirabel=np.maximum(mirabel,arrays["mirabel"][current])
            if turn<=4:early=np.maximum(early,self._early_risk(arrays,orders,turn))
            if turn>=5:rolling=np.maximum(rolling,self._rolling_risk(arrays,orders,turn))
            branch_values={"Mirabel":mirabel,"Early":early,"Rolling CCT":rolling}
            branch_p=np.column_stack([_upper_p(branch_values[name],np.sort(self.reference[name][:,turn-1])) for name in BRANCHES])
            clipped=np.clip(branch_p,1e-12,1-1e-12)
            sparse=-np.log10(np.clip(np.minimum(1.0,3*np.min(clipped,axis=1)),1e-15,1.0))
            dense=_cct(clipped)
            p_sparse=_upper_p(sparse,np.sort(self.component_reference["sparse_equal_minp"][:,turn-1]))
            p_dense=_upper_p(dense,np.sort(self.component_reference["dense_V15"][:,turn-1]))
            instantaneous=-np.log10(np.clip(np.minimum(1.0,2*np.minimum(p_sparse,p_dense)),1e-15,1.0))
            sticky=np.maximum(sticky,instantaneous)
        return sticky

    def exact_attack(self,rows:Sequence[Mapping[str,object]],query_utility_proxy:Sequence[float],*,subset_size:int,retention:float=.90,batch_size:int=200000)->V16OrderedSubsetResult:
        return self.exact_attack_many_retention(
            rows, query_utility_proxy, subset_size=subset_size,
            retentions=(retention,), batch_size=batch_size,
        )[float(retention)]

    def exact_attack_many_retention(self,rows:Sequence[Mapping[str,object]],query_utility_proxy:Sequence[float],*,subset_size:int,retentions:Sequence[float]=(.70,.90,1.0),batch_size:int=200000)->dict[float,V16OrderedSubsetResult]:
        utility=np.asarray(query_utility_proxy,dtype=float);full=float(utility.max())
        retention_values=tuple(float(value) for value in retentions)
        if any(not 0<=value<=1 for value in retention_values):raise ValueError("retentions must lie in [0,1]")
        best={value:None for value in retention_values};evaluated={value:0 for value in retention_values}
        iterator=itertools.permutations(range(len(rows)),int(subset_size))
        while True:
            chunk=list(itertools.islice(iterator,int(batch_size)))
            if not chunk:break
            orders=np.asarray(chunk,dtype=np.int16)
            scores=self.score_orders(rows,orders);subset_utility=np.max(utility[orders],axis=1)
            for retention in retention_values:
                valid=subset_utility+1e-15>=retention*full
                if not np.any(valid):continue
                valid_indices=np.flatnonzero(valid);local=int(valid_indices[np.argmin(scores[valid])])
                evaluated[retention]+=int(valid.sum())
                candidate=(float(scores[local]),tuple(map(int,orders[local])))
                if best[retention] is None or candidate<best[retention]:best[retention]=candidate
        output={}
        for retention in retention_values:
            if best[retention] is None:raise RuntimeError("no utility-valid ordered subset")
            score,order=best[retention];selected_utility=float(utility[list(order)].max())
            output[retention]=V16OrderedSubsetResult(int(subset_size),order,score,score,score>self.threshold,score>self.threshold,evaluated[retention],selected_utility,1.0 if full==0 else selected_utility/full)
        return output
