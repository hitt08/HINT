from sklearn.feature_extraction.text import CountVectorizer
import logging
import sys
from scipy.stats import entropy


from python_utils.data import *
from python_utils.time import *
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from multiprocessing.managers import BaseManager
from utils.thread_utils import tokenize_split

log = logging.getLogger()

th_div = None

def get_heldout_entropy(thread,tk_passage_collection,ngram=3,sep = "~|~"):
    div = []
    for d, _ in thread:
        vect = CountVectorizer(tokenizer=tokenize_split, strip_accents='ascii', max_features=100000, token_pattern=None,ngram_range=(1, ngram))
        held_out_data = [sep.join(tk_passage_collection[td]) for td, _ in thread if td != d]
        thread_dist = vect.fit_transform(held_out_data)
        prob_dist = vect.transform([sep.join(tk_passage_collection[d])])
        h = entropy(prob_dist.toarray(), np.mean(thread_dist.toarray(), axis=0), axis=1)
        div.append(h)

    return np.mean(div)


def get_thread_diversity(idx):
    thread = th_div.threads[idx]

    if th_div.skip_threshold is not None and len(thread) > th_div.skip_threshold:
        th_div.res[idx]= np.nan
    else:
        th_div.res[idx]= np.nan if len(thread) <= 1 else get_heldout_entropy(thread, th_div.tk_passage_collection, ngram=th_div.ngram, sep="~|~")

    th_div.pbar.update()
    if idx%1000==0:
        log.warning("")


class Thread_Diversity:
    def __init__(self, threads, tk_passage_collection, ngram=3, nprocesses=4, skip_threshold=None):
        self.threads = threads
        self.tk_passage_collection = tk_passage_collection
        self.ngram = ngram

        self.nprocesses = nprocesses

        self.res = {}
        self.pbar = None
        self.skip_threshold=skip_threshold

    def get_diversity(self):
        global th_div
        th_div = self

        pool_iter = list(range(len(self.threads)))

        if self.nprocesses>1:
            manager = Manager()
            BaseManager.register("pbar", tqdm)
            bmanager = BaseManager()
            bmanager.start()
            self.pbar = bmanager.pbar(total=len(self.threads))
            self.res = manager.dict()

            with Pool(processes=self.nprocesses) as pool:
                pool.map(func=get_thread_diversity, iterable=pool_iter)  # [st:en])

        else:
            self.pbar=tqdm(total=len(self.threads))
            self.res={}
            for idx in pool_iter:
                get_thread_diversity(idx)

        self.pbar.close()
        div = np.asarray([self.res[i] for i in pool_iter])

        return div