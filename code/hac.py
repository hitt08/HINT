from sklearn.cluster import AgglomerativeClustering
import pickle
import numpy as np
import logging
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import sys

from python_utils.data import *
from utils.data_utils import load_config,read_nh_part,read_dict_dump
from utils.thread_utils import ev_thread_score,eval_ev_threads
import networkx as nx
from weight_functions import shortest_path_length,get_all_direct_paths,get_shortest_path_lengths
from entity_graph import create_entity_graph
from collections import Counter

alpha = 0
gamma = 0
mCE = 0  #Max common entities between any passage pair
mEB = 0  #Max entites between in the shortest path between any passage pair
w = 0 #Weight Threshold
G=None
doc_ids = None


def time_decay(x):  # Similarity
    global alpha
    T = np.max(x) - np.min(x)
    # y=np.full((x.shape[0],x.shape[0]),x)
    return np.exp(-alpha * np.abs(x[:, None] - x) / T)


def cos_time(a):  # distance
    tmp = cosine_similarity(a[:, :-1]) * time_decay(a[:, -1])
    cond = (1 - tmp) > w #weight threshold
    tmp[cond] = 0
    return 1 - tmp#cosine_similarity(a[:, :-1]) * time_decay(a[:, -1])


def cos_ent(a):  # distance
    tmp=cosine_similarity(a[:, :-1])
    cond = (1 - tmp) > w  # weight threshold
    tmp[cond] = 0
    cond = np.logical_not(cond)
    tmp[cond] = tmp[cond] * nxe_weight(a[:, -1], cond)
    return 1 -  tmp


def cos_ent_time(a):  # distance
    log = logging.getLogger(__name__)
    tmp = cosine_similarity(a[:, :-2]) * time_decay(a[:, -2])
    cond = (1 - tmp) > w #weight threshold
    tmp[cond] = 0
    cond = np.logical_not(cond)
    log.warning(f"Cond: {np.sum(cond)},{cond.shape},{tmp[cond].shape}")
    tmp[cond] = tmp[cond] * nxe_weight(a[:, -1], cond)

    return 1 - tmp

    # return 1 - cosine_similarity(a[:, :-2]) * time_decay(a[:, -2]) * nxe_weight(a[:, -1])


def nxe_weight(x,cond,verbose=True):  # Similarity
    global gamma,mCE,mEB,G,doc_ids
    log = logging.getLogger(__name__)

    items = doc_ids[x.astype(int)]
    # print(items)
    # temp=dict([(t, 0) for t in targets])
    ent_similarity = np.zeros((len(items), len(items)))
    log.warning(f"ES: {ent_similarity.shape},{ent_similarity[cond].shape}")

    if verbose:
        pbar=tqdm(total=len(items))

    for idx, s in enumerate(items):
        s_cond = cond[idx].copy()
        s_cond[:idx] = False #Compute only once for a pair of documents
        targets = items[s_cond]

        if verbose:
            pbar.set_description(f"Targets: {len(targets)}")

        temp = get_all_direct_paths(G, s, targets)


        # print(targets)
        # temp = dict([(t, len(list(nx.all_simple_paths(G, s, t, cutoff=2)))) for t in targets])
        # temp[t]==len(list(nx.all_simple_paths(G, s, t, cutoff=2)))

        targets_k, targets_p = [], []
        for t in targets:
            targets_k.append(t)
            targets_p.append(temp[t])
        targets_k = np.asarray(targets_k)
        targets_p = np.asarray(targets_p)

        is_path_weight = targets_p == 0
        ent_weight_args = np.argwhere(is_path_weight == False).squeeze(-1)
        path_weight_args = np.argwhere(is_path_weight).squeeze(-1)
        # print(targets_p,is_path_weight,path_weight_args,path_weight_args.shape)#,len(path_weight_args))
        res = np.zeros(len(targets))

        ent_weights = targets_p[ent_weight_args]

        # path_weights = np.asarray(get_shortest_path_lengths(G, s, targets_k[path_weight_args], cutoff=mEB * 2))

        path_weights = np.zeros_like(path_weight_args)
        for i, t in enumerate(targets_k[path_weight_args]):
            try:
                path_weights[i] = shortest_path_length(G, s, t, cutoff=mEB * 2)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                path_weights[i] = mEB * 2  # G.number_of_nodes()

        ent_weights = 0.5 * (1 + (1 - np.exp(-gamma * (ent_weights / mCE))))
        # path_weights = 1 - (1 - item_weights[path_weight_args]) * 0.5 * (1 - np.log(path_weights / 2) / np.log(max_ent_between))
        path_weights = 0.5 * np.exp(-gamma * ((path_weights / 2) / mEB))

        res[ent_weight_args] = ent_weights
        res[path_weight_args] = path_weights

        ent_similarity[idx][s_cond] = res

        if verbose:
            pbar.update()

    if verbose:
        pbar.close()

    #Copy values a->b to b->a
    ent_similarity[np.tril_indices(ent_similarity.shape[0])] = ent_similarity[np.tril_indices(ent_similarity.shape[0])[::-1]]

    log.warning(f"ES2: {ent_similarity.shape},{ent_similarity[cond].shape}")

    return ent_similarity[cond]


def get_threads(hac_model, model_doc_ids, date_data, passage_matrix, passage_doc2id):

    threads = []
    thread_similarity = []

    with tqdm(total=hac_model.n_clusters_) as pbar:
        for l in range(hac_model.n_clusters_):
            c_lbl_idx = np.argwhere(hac_model.labels_ == l).squeeze()
            thread = []
            c_lbls = np.array(model_doc_ids)[c_lbl_idx].tolist()
            if type(c_lbls) == str:
                c_lbls = [c_lbls]
            for i in c_lbls:
                thread.append((i, date_data[i]))
            thread = sorted(thread, key=lambda x: x[1])
            # thread_period = thread[-1][1] - thread[0][1]

            thread_sim = np.mean(ev_thread_score(thread, passage_doc2id, passage_matrix)) if len(thread) > 1 else np.nan

            threads.append(thread)
            thread_similarity.append(thread_sim)
            pbar.update()

    thread_similarity = np.asarray(thread_similarity)
    return threads, np.array(thread_similarity)


def get_hac_threads(ofile,k,train_doc_ids,train_emb,passage_doc_ids,passage_emb,date_data,is_td=False,a=0,dt_feat=None, is_ent=False,g=0,ent_graph=None,max_common_ent=5,max_ent_bet=5,weight_threshold=0.8):
    global alpha,gamma,mCE,mEB,G,doc_ids,w
    log = logging.getLogger(__name__)

    doc2id = dict([(doc_id, idx) for idx, doc_id in enumerate(passage_doc_ids)])

    alpha=a
    gamma=g
    mCE=max_common_ent
    mEB=max_ent_bet
    w=weight_threshold
    G=ent_graph
    doc_ids=np.asarray(passage_doc_ids)

    ent_doc_ids_feat = np.asarray([doc2id[d] for d in train_doc_ids]) if is_ent else None

    linkage = "complete"
    if is_td and is_ent:
        log.warning("Time Decay with Entity Similarity")
        train_emb = np.hstack((train_emb, dt_feat[:, None],ent_doc_ids_feat[:, None]))  # term features, dt_feat, doc_ids
        affinity = cos_ent_time
    elif is_td:
        log.warning("Time Decay")
        train_emb = np.hstack((train_emb, dt_feat[:, None]))  # term features, dt_feat
        affinity = cos_time
    elif is_ent:
        log.warning("Entity Similarity")
        train_emb = np.hstack((train_emb, ent_doc_ids_feat[:, None]))  # term features, doc_ids
        affinity = cos_ent
    else:
        log.warning("Ward")
        linkage = "ward"
        affinity = "euclidean"

    log.warning(f"Clustering. Train Data: {train_emb.shape}, Labels: {k}")

    nh_clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=affinity).fit(train_emb)

    with open(ofile, "wb") as f:
        pickle.dump(nh_clustering, f)
    log.warning(f"\tModel Dumped at: {ofile}")

    log.warning("\tGenerating Threads")
    threads, thread_similarity = get_threads(nh_clustering, train_doc_ids,date_data, passage_emb, doc2id)

    return threads, thread_similarity


if __name__ == "__main__":
    # global pnct_psg_tfidf_train_matrix,outdir
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str, default="minilm",help='embedding model: minilm; distilroberta;tfidf')
    parser.add_argument('-a', action='store',dest='alpha', type=float,default=0,help='time decay factor')
    parser.add_argument('-g', action='store', dest='gamma', type=float, default=0, help='entity factor')
    parser.add_argument('--mce', action='store', dest='mCE', type=float, default=5, help='max common entities')
    parser.add_argument('--meb', action='store', dest='mEB', type=float, default=5, help='max entity between')
    parser.add_argument('-w', action='store', dest='weight_threshold', type=float, default=0.7,help='edge weight threshold')

    parser.add_argument('--td', dest='td', action='store_true', help='time decay')
    parser.set_defaults(td=False)
    parser.add_argument('--ent', dest='ent', action='store_true', help='entity similarity')
    parser.set_defaults(ent=False)

    args = parser.parse_args()
    log = logging.getLogger(__name__)
    log.warning(args)

    config = load_config()


    with open(f"{config.graph_root_dir}/{config.doc_ids_file}", "rb") as f:
        pnct_nh_5w1h_passage_doc_ids = pickle.load(f)

    with np.load(f"{config.emb_root_dir}/nh_5w1h_{args.emb}_emb.npz") as data:
        nh_5w1h_emb = data["arr_0"]

    with open(f"{config.root_dir}/5w1h/nh_date_data.p", "rb") as f:
        nh_date_data = pickle.load(f)

    with np.load(f"{config.graph_root_dir}/{config.date_features_file}") as data:
        dt_feat = data["arr_0"]

    with np.load(f"{config.emb_root_dir}/nh_{args.emb}_emb.npz") as data:
        nh_passage_emb = data["arr_0"]

    nh_part = read_nh_part(config.root_dir)

    doc2storyid = read_dict(f"{config.root_dir}/newshead/processed/doc2storyid_filtered.txt")
    thread_true_labels = [int(doc2storyid[i]) for i in pnct_nh_5w1h_passage_doc_ids]


    nh_entity_dict=read_dict_dump(f"{config.graph_root_dir}/{config.entity_dict_file}")


    threads,thread_similarity=[],[]

    pattern = f"{args.emb}_w{args.weight_threshold}"
    if args.td:
        pattern += f"_td{args.alpha}"
    if args.ent:
        pattern += f"_ent{args.gamma}"
    if not args.td and not args.ent:
        pattern += f"_ward"

    for p_id, (p_st, p_en) in tqdm(nh_part.items()):
        log.warning(f"Part: {p_id + 1}")
        train_doc_ids = pnct_nh_5w1h_passage_doc_ids[p_st:p_en]
        train_5w1h_emb=nh_5w1h_emb[p_st:p_en]
        train_dt_emb = dt_feat[p_st:p_en]
        k= len(set(thread_true_labels[p_st:p_en]))

        if args.ent:
            G=create_entity_graph(nh_entity_dict,train_doc_ids)
            log.warning(f"Entity Graph: Number of Nodes= {G.number_of_nodes()}. Number of Edges= {G.number_of_edges()}")
        else:
            G=None

        ofile = f"{config.graph_root_dir}/hac/hac_{pattern}_{p_id}.p"
        part_threads,part_thread_similarity=get_hac_threads(ofile, k, train_doc_ids, train_5w1h_emb, pnct_nh_5w1h_passage_doc_ids, nh_passage_emb, nh_date_data, is_td=args.td, a=args.alpha,dt_feat=train_dt_emb, is_ent=args.ent, g=args.gamma, ent_graph=G, max_common_ent=args.mCE, max_ent_bet=args.mEB,weight_threshold=args.weight_threshold)

        threads.extend(part_threads)
        thread_similarity.append(part_thread_similarity)


    thread_similarity=np.hstack(thread_similarity)

    ofile = f"{config.threads_dir}/threads_hac_{pattern}.p"
    with open(ofile, "wb") as f:
        pickle.dump(threads, f)
    log.warning(f"\tThreads stored at: {ofile}")
    np.savez_compressed(f"{config.threads_dir}/thread_sim_hac_{pattern}.npz", thread_similarity)


    res = f"\nThread Count:{len(threads)}"
    res += f"\nAvg Length:{np.mean([len(i) for i in threads])}"
    res += f"\nCosine Score: {np.mean(thread_similarity[np.logical_not(np.isnan(thread_similarity))])}"
    log.warning(res)

    eval_ev_threads(threads, doc2storyid, pnct_nh_5w1h_passage_doc_ids, thread_true_labels, min_len=0, return_dict=False, print_res=True)


# runjob -c "hac.py -e minilm" -i "hitt08/ubuntu-py:gpu_v2" -n "mini-ward" -m 32 -r
# runjob -c "hac.py -e minilm --td -a 10" -i "hitt08/ubuntu-py:gpu_v2" -n "mini-td" -m 64 -r
# runjob -c "hac.py -e minilm --td -a 10 --ent -g 0.1" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-e01" -m 64 -r
# runjob -c "hac.py -e minilm --td --ent -a 1 -g 0.001" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t1-e01" -m 64 -r
# runjob -c "hac.py -e minilm --td --ent -a 10 -g 1.5" -i "hitt08/ubuntu-py:gpu_v2" -n "mini-td-ent" -m 64 -r
# runjob -c "hac.py -e minilm --ent -g 1.5" -i "hitt08/ubuntu-py:gpu_v2" -n "mini-ent" -m 64 -r


# runjob -c "hac.py -e minilm --td -a 10 -w 1" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-w1" -m 64 -r
# runjob -c "hac.py -e minilm --td -a 10 -w 0.7" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-w7" -m 64 -r
# runjob -c "hac.py -e minilm --td -a 10 -w 0.8" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-w8" -m 64 -r

# runjob -c "hac.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-e01-w7" -m 64 -r
# runjob -c "hac.py -e minilm --td -a 10 --ent -g 0.1 -w 0.8" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-e01-w8" -m 64 -r
# runjob -c "hac.py -e minilm --td -a 10 --ent -g 0.1 -w 1" -i "hitt08/ubuntu-py:gpu_v2" -n "m-t10-e01-w1" -m 64 -r


# runjob -c "hac.py -e distilroberta --td -a 10 -w 0.7" -i "hitt08/ubuntu-py:gpu_v2" -n "r-t10-w7" -m 64 -r
# runjob -c "hac.py -e distilroberta --td -a 10 -w 0.8" -i "hitt08/ubuntu-py:gpu_v2" -n "r-t10-w8" -m 64 -r

# runjob -c "hac.py -e distilroberta --td -a 10 --ent -g 0.1 -w 0.7" -i "hitt08/ubuntu-py:gpu_v2" -n "r-t10-e01-w7" -m 64 -r
# runjob -c "hac.py -e distilroberta --td -a 10 --ent -g 0.1 -w 0.8" -i "hitt08/ubuntu-py:gpu_v2" -n "r-t10-e01-w8" -m 64 -r


