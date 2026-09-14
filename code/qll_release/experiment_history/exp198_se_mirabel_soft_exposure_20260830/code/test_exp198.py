#!/usr/bin/env python3
import numpy as np
from run_exp198 import FLOOR, TOP_K, TOTAL, largest_remainder_caps, mirabel_probabilities

def main():
    p,tau=mirabel_probabilities(np.linspace(.8,.2,10),0.1,.1,1000)
    caps,raw=largest_remainder_caps(p)
    assert np.isclose(p.sum(),1) and len(caps)==TOP_K
    assert caps.sum()==TOTAL and caps.min()>=FLOOR and caps.max()<=221
    uniform=np.ones(10)/10;u,_=largest_remainder_caps(uniform)
    assert sorted(u.tolist())==[204,204]+[205]*8
    assert caps[0]<caps[-1]
    print("EXP198_UNIT_TESTS_PASS")

if __name__=="__main__":main()
