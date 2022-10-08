import argparse
import sys
from python_utils.data import *
from python_utils.time import *
import json
from tqdm import tqdm
import scipy
import os
from collections import Counter
import numpy as np
from scipy.special import softmax
import pandas as pd
import logging
import datetime
import pickle
import gzip
from utils.data_utils import load_config
from nltk.corpus import stopwords

stp = stopwords.words('english')

from multiprocessing import Pool, cpu_count, Manager
from multiprocessing.managers import BaseManager

PROCESSES = cpu_count()

log = logging.getLogger(__name__)


threads, thread_similarity, all_coh, all_div=None,None,None,None
pbar,res_dict=None,None

def grid_search(args):
    l_range, s_range, d_range = args

    new_all_coh = []
    new_all_div = []
    for i, s, c, e in zip(threads, thread_similarity, all_coh, all_div):
        if l_range[0] <= len(i) <= l_range[1] and \
                datetime.timedelta(d_range[0]) <= (i[-1][1] - i[0][1]) <= datetime.timedelta(d_range[1]) and \
                s_range[0] <= s <= s_range[1]:
            new_all_coh.append(c)
            new_all_div.append(e)

    new_all_coh = np.asarray(new_all_coh)
    new_all_div = np.asarray(new_all_div)

    cond = np.isnan(new_all_coh)
    c = np.mean(new_all_coh[np.logical_not(cond)])
    n = np.sum(cond)

    cond = np.logical_or(np.isnan(new_all_div), np.isinf(new_all_div))
    e = np.mean(new_all_div[np.logical_not(cond)])
    en = np.sum(cond)

    res_dict[f"l:{l_range}, s:{s_range}, d:{d_range}"] = [c, n, e, en, len(new_all_coh)]

    res = np.asarray(list(res_dict.values()))
    mxid = np.lexsort((res[:, 0], res[:, 2]))[-1]
    pbar.set_description(f"Max=> {list(res_dict.items())[mxid]}")
    pbar.update()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action='store', dest='name', type=str, help='run name')
    parser.add_argument('-c', action='store', dest='nprocesses', type=int, default=4, help='cpus')
    parser.add_argument('-p', action='store', dest='parallel_batches', type=int, default=1024, help='parallel_batches')
    parser.add_argument('-i', action='store', dest='idx', type=int,default=None, help='input file part idx')
    parser.add_argument('--quick', dest='quick', action='store_true', help='quick search')
    parser.set_defaults(quick=False)
    args = parser.parse_args()
    log.warning(args)

    config = load_config()

    all_res = {}

    name = args.name
    log.warning(name.upper())


    pattern=args.name if args.idx is None else f"{args.name}_{args.idx}"

    ofile = f"{config.validation_dir}/sample_threads_{pattern}.p"
    with open(ofile, "rb") as f:
        threads = pickle.load(f)

    ofile = f"{config.validation_dir}/sample_thread_sim_{pattern}.npz"
    with np.load(ofile) as data:
        thread_similarity = data["arr_0"]

    with open(f"{config.validation_dir}/sample_divergence_{pattern}.p", "rb") as f:
        all_coh=pickle.load(f)

    with open(f"{config.validation_dir}/sample_coherence_{pattern}.p", "rb") as f:
        all_div=pickle.load(f)


    if args.quick:
        grid = {"l_range":[[3,5]]+[[3,i] for i in range(10,110,10)],
                "s_range":[(round(0+i,1),round(1-i,1)) for i in np.arange(0,0.5,0.1)],
                "d_range":[[i,j] for i in [0]    for j in [30,60,90,120,180,365]]}
    else:
        grid = {"l_range":[[3,5]]+[[3,i] for i in range(10,110,10)],
                "s_range": [[round(i, 1), round(j, 1)] for i in np.arange(0, 0.5, 0.1) for j in np.arange(0.6, 1.01, 0.1)],
                "d_range":[[i,j] for i in [0]    for j in [30,60,90,120,180,365]]}

    pool_iter = [(l_range, s_range, d_range) for l_range in grid["l_range"] for s_range in grid["s_range"] for d_range in grid["d_range"]]

    batch_size = args.parallel_batches
    st = 0
    en = batch_size

    manager = Manager()
    count = manager.list()

    BaseManager.register("pbar", tqdm)
    bmanager = BaseManager()
    bmanager.start()
    pbar = bmanager.pbar(total=len(pool_iter))

    main_dict = {}
    while st < len(pool_iter):
        res_dict = manager.dict()
        with Pool(processes=args.nprocesses) as pool:
            pool.map(func=grid_search, iterable=pool_iter[st:en])
        st = en
        en += batch_size
        main_dict.update(dict(res_dict))

        log.warning("")

    pbar.close()

    q="_q" if args.quick else ""

    with open(f"{config.validation_dir}/grid_search/coh_div_grid_search_dict_{pattern}{q}.p", "wb") as f:
        pickle.dump(main_dict, f)