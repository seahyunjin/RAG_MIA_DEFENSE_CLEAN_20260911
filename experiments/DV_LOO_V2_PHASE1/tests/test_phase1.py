import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import empirical_cdf, mirabel_margin, strict_threshold


def test_empirical_cdf_branch_reference():
    got=empirical_cdf([1,2,3,4],[0,1,2.5,4,5])
    assert np.allclose(got,[0,.25,.5,1,1])


def test_final_max_requires_recalibration():
    left=empirical_cdf([1,2,3,4],[1,2,3,4])
    right=empirical_cdf([4,3,2,1],[4,3,2,1])
    combined=np.maximum(left,right)
    threshold,fp,_=strict_threshold(combined,.25)
    assert fp <= 1
    assert np.sum(combined>threshold)==fp


def test_strict_threshold():
    threshold,fp,fpr=strict_threshold(np.arange(100,dtype=float),.03)
    assert fp==3 and math.isclose(fpr,.03)


def test_mirabel_boundary_equivalence():
    *_,tau,margin=mirabel_margin([0.1,0.2,0.3,0.9],rho=.05)
    assert (margin>0)==(0.9>tau)


def test_mirabel_uses_population_std():
    top,smax,mu,sigma,tau,margin=mirabel_margin([0.1,0.2,0.3,0.4],rho=.05)
    assert top==3 and smax==.4
    assert math.isclose(mu,.2)
    assert math.isclose(sigma,np.std([.1,.2,.3],ddof=0))
