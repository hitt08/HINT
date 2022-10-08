import cupy as cp
import numpy as np
import sys
import pickle
import logging
import argparse
from tqdm import tqdm
import networkx as nx
sys.path.insert(1, f'/nfs/jup/sensitivity_classifier/')
from python_utils.data import *
from python_utils.time import *
from utils.data_utils import read_nh_part
from weight_functions import cos_time, get_nxe_weight_edges

log=logging.getLogger()

def construct_passage_graph(alpha=10,gamma=1,max_e=5,weight_threshold=0.7,root_dir="/nfs/threading",emb="minilm",notification_iter=-1):
    graph_root_dir = f"{root_dir}/graph_threading"
    nh_part = read_nh_part(root_dir)
    with np.load(f"{graph_root_dir}/nh_5w1h_{emb}_dt_emb.npz") as data:
        nh_5w1h_dt_emb = data["arr_0"]

    with np.load(f"{graph_root_dir}/dt_feat.npz") as data:
        dt_feat = data["arr_0"]

    with open(f"{graph_root_dir}/pnct_nh_5w1h_passage_data.p","rb") as f:
        pnct_nh_5w1h_passage_data=pickle.load(f)

    nh_5w1h_dt_emb_cp = cp.asarray(nh_5w1h_dt_emb)
    G=nx.read_gpickle(f"{graph_root_dir}/entity_graph.gp")
    # G = nx.read_gpickle(f"{graph_root_dir}/entity_graph_no_date.gp")

    log.warning("Creating Batches")
    pool_iter = []
    for p_id, (p_st, p_en) in tqdm(nh_part.items()):
        batch_size = 128
        st = 0
        en = batch_size
        temp_pool_iter = []
        b = 0
        while st < nh_5w1h_dt_emb[p_st:p_en].shape[0]:
            temp_pool_iter.append((b, nh_5w1h_dt_emb_cp[p_st:p_en][st:en]))
            st = en
            en = st + batch_size
            b += 1
        pool_iter.append(temp_pool_iter)

    eT=max_e

    log.warning("Constructing Passage Graph")
    notifications=notification_iter>=0
    notification_counter=0
    for p_id, (p_st, p_en) in nh_part.items():
        log.warning(f"Part: {p_id + 1}/{len(nh_part)}")
        T = np.max(dt_feat[p_st:p_en]) - np.min(dt_feat[p_st:p_en])
        node_id = 0
        edges = []
        pt_passage_data = pnct_nh_5w1h_passage_data["train"]["ids"][p_st:p_en]
        with tqdm(total=nh_5w1h_dt_emb[p_st:p_en].shape[0]) as pbar:
            for idx, x in pool_iter[p_id]:
                item_weights = cp.asnumpy(
                    cos_time(x, nh_5w1h_dt_emb_cp[p_st:p_en], T=T, alpha=alpha, use_gpu=True))
                temp = []
                candidates = np.argwhere(item_weights < weight_threshold)
                for s in range(item_weights.shape[0]):
                    source_node_id = idx * batch_size + s
                    neighbours = candidates[candidates[:, 0] == s, 1]
                    neighbours = neighbours[neighbours > source_node_id]

                    if len(neighbours)>0:
                        source = pt_passage_data[source_node_id]
                        targets=[pt_passage_data[n] for n in neighbours]
                        new_edges=get_nxe_weight_edges(G, source, targets,item_weights[s, neighbours], T=eT, gamma=gamma)
                        temp.extend(new_edges)
                        #
                        # target = pt_passage_data[n]
                        # nw = 1 - (1 - item_weights[s, n]) * get_nxe_weight(G, source, target, T=eT, gamma=gamma)
                        # # ews.append(nw)
                        # # if nw < 0.9:#1 - (1 - weight_threshold) * 0.5:
                        #     # print(nw)
                        # temp.append((source, target, nw))
                    pbar.update()
                    if notifications:
                        notification_counter+=1
                        if notification_counter%notification_iter==0:
                            log.warning("")
                            notification_counter=0

                edges.extend(temp)
                pbar.set_description(f"Number of Edges: {len(edges)}")

            DG = nx.DiGraph()
            DG.add_weighted_edges_from(edges)
            pbar.set_description(f"Number of Nodes: {DG.number_of_nodes()}. Number of Edges: {DG.number_of_edges()}")
        nx.write_gpickle(DG,f"{graph_root_dir}/passage_graphs/{emb}_nx_graph_el{eT}_w{weight_threshold}_a{alpha}_g{gamma}_{p_id}.gp")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str,default="minilm", help='embedding model minilm or distilroberta')

    parser.add_argument('-w', action='store', dest='weight_threshold', type=float, default=0.7, help='passage weight threshold')
    parser.add_argument('-a', action='store',dest='alpha', type=float,default=10,help='time decay factor')
    parser.add_argument('-g', action='store',dest='gamma', type=float,default=1,help='entity factor')
    parser.add_argument('-t', action='store', dest='max_e', type=int, default=5, help='max between entities')
    parser.add_argument('-n', action='store', dest='notification', type=int,default=-1, help='break tqdm notifications')
    args = parser.parse_args()

    log.warning(args)

    construct_passage_graph(alpha=args.alpha, gamma=args.gamma, max_e=args.max_e, weight_threshold=args.weight_threshold, emb=args.emb,notification_iter=args.notification)
