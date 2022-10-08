import logging
import sys
sys.path.insert(1, f'/nfs/jup/sensitivity_classifier/')
from python_utils.data import *
from python_utils.time import *
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
from scipy.special import softmax
from multiprocessing import Pool, cpu_count, Manager
import os
from utils.data_utils import write_json_dump,read_json_dump

log = logging.getLogger()


def get_thread_df(threads, thread_keywords, outFile):
    thread_dict = []
    for idx, thread in enumerate(threads):
        temp = {}
        temp["length"] = len(thread)
        temp["period"] = thread[-1][1] - thread[0][1]

        temp["keywords"] = []
        temp["keyword_weight"] = 0

        for w, s in thread_keywords[idx]:
            temp["keywords"].append(w)
            temp["keyword_weight"] += s
        temp["keyword_weight"] /= 3
        thread_dict.append(temp)
    thread_df = pd.DataFrame(thread_dict).sort_values("keyword_weight", ascending=False)

    # outFile=f"{outPath}/{ofile}_threads.json.gz"
    thread_df["period"] = thread_df["period"].apply(lambda x: str(x))
    write_json_dump(outFile, thread_df.reset_index().to_dict(orient="records"), compress=True)
    thread_df["period"] = pd.to_timedelta(thread_df["period"])
    log.warning(f"Thread weights saved to : {outFile}")



key_ext = None
def keyword_parallel(idx):
    cond = key_ext.pred_labels == idx
    if not np.any(cond):
        return

    cluster = [key_ext.keyword_corpus[i] for i in np.argwhere(key_ext.labels == idx).squeeze(-1)]

    vocab = []
    for p in cluster:
        vocab.extend([t for t in set(p)])

    # t0=time.time()
    df = list(Counter(vocab).items())
    dfs = np.zeros(len(key_ext.feature_names))
    for w, s in df:
        if w in key_ext.features_set:
            widx = key_ext.word2id[w]
            dfs[widx] = s / len(cluster)
            if dfs[widx] >= key_ext.max_df:
                dfs[widx] = 0

    # t0=time.time()
    weight = np.asarray(np.abs(np.mean(key_ext.train_matrix[key_ext.pred_labels == idx], axis=0)))[0]

    # t0=time.time()
    tmpWeight = np.log(softmax(dfs, axis=-1)) + np.log(softmax(weight, axis=-1))
    tempIdx = np.argsort(tmpWeight)[::-1][:key_ext.top_n]
    wl = np.asarray(key_ext.feature_names)[tempIdx].tolist()

    key_ext.res[idx] = list(zip(wl, tmpWeight[tempIdx]))


class Keyword_Extractor:
    def __init__(self, data, features, top_n_keywords=10, max_doc_freq=0.9, filter_ids=None, print_log=False,
                 nprocesses=8):
        self.res = []
        self.keyword_corpus = data["words"]
        self.labels = data["labels"]
        self.feature_names = features
        self.max_df = max_doc_freq
        self.print_text = print_log
        self.top_n = top_n_keywords
        self.nprocesses = nprocesses

        self.features_set = set(self.feature_names)
        self.word2id = dict([(w, i) for i, w in enumerate(self.feature_names)])

        if filter_ids is not None:
            self.pred_labels = self.labels[filter_ids]
            self.train_matrix = data["matrix"][filter_ids]
        else:
            self.pred_labels = self.labels
            self.train_matrix = data["matrix"]

    def thread_cluster_keywords(self, batch_size=1000):
        global key_ext

        pool_iter = list(set(self.pred_labels.tolist()))
        key_ext = self

        if self.nprocesses>1:
            manager = Manager()
            self.res = manager.dict()
            st = 0
            en = batch_size
            with tqdm(total=len(pool_iter)) as pbar:
                while st < len(pool_iter):
                    with Pool(processes=self.nprocesses) as pool:
                        pool.map(func=keyword_parallel, iterable=pool_iter[st:en])
                    pbar.update(len(pool_iter[st:en]))
                    st = en
                    en = st + batch_size
        else:
            self.res={}
            for idx in tqdm(pool_iter):
                keyword_parallel(idx)

        return self.res


def get_threads_keywords(threads, tk_passage_collection, passage_doc_ids, passage_features, passage_tfidf_matrix,outPath, top_n=3,nprocesses=8, force=False):
    # tk_passage_collection,passage_doc_ids,passage_features,passage_tfidf_matrix : For all the passages in a collection
    # passage_doc_ids,passage_tfidf_matrix: should have the same index

    threadFile = f"{outPath}threads_keywords.json.gz"

    if not force and os.path.exists(threadFile):
        log.warning(f"Threads Keywords Founds at:\n\t{threadFile}")
    else:
        doc2thread_dict = {}
        for i, t in enumerate(threads):
            for d, _ in t:
                doc2thread_dict[d] = i

        thread_pred_labels = []
        passage_filter_ids = []
        keyword_collection = []
        for i, x in enumerate(passage_doc_ids):
            if x in doc2thread_dict:
                thread_pred_labels.append(doc2thread_dict[x])
                passage_filter_ids.append(i)
            else:
                thread_pred_labels.append(-1)
            keyword_collection.append(tk_passage_collection[x])
        thread_pred_labels = np.asarray(thread_pred_labels)

        data = dict(words=keyword_collection, matrix=passage_tfidf_matrix, labels=thread_pred_labels)

        key_ext = Keyword_Extractor(data,passage_features,top_n_keywords=top_n,filter_ids=passage_filter_ids, print_log=False,nprocesses=nprocesses)

        thread_keywords = key_ext.thread_cluster_keywords()

        get_thread_df(threads, thread_keywords, threadFile)

    thread_df = pd.DataFrame(read_json_dump(threadFile, compress=True)).set_index("index")
    thread_df["period"] = pd.to_timedelta(thread_df["period"])

    # return keyword_df,thread_df,keyword2thread
    return thread_df