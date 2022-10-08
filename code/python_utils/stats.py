import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from collections import Counter

def run_mcnemar(test1,test2,exact=False,correction=True):
    res=mcnemar(pd.crosstab(test1,test2).values,exact,correction)
    return res, res.pvalue<=.05

def transitions(orig_test_labels, orig_lbls, pred_lbls):
    trans = []
    for l, o, p in zip(orig_test_labels, orig_lbls, pred_lbls):
        o_tru = "N" if o == 0 else "P"
        p_tru = "N" if p == 0 else "P"
        o_pos = "T" if o == l else "F"
        p_pos = "T" if p == l else "F"
        ol = o_pos + o_tru
        pl = p_pos + p_tru
        if (ol != pl):
            trans.append(f"{ol}->{pl}")
    trans = sorted(Counter(trans).items())

    return ", ".join([f"{k}:{v}" for k, v in trans])

def sig_test(pred_lbls,test_labels, orig_lbl, rep_lbl=[]):
    osig, otest = run_mcnemar(orig_lbl, np.array(pred_lbls))
    res=f" Sig_Orig_pvalue={osig.pvalue} (Significant={otest}), "
    if(len(rep_lbl)>0):
        rsig, rtest = run_mcnemar(rep_lbl, np.array(pred_lbls))
        res+=f"Sig_Repl_pvalue={rsig.pvalue} (Significant={rtest}), "
    return res+transitions(test_labels, orig_lbl, pred_lbls)
