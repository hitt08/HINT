from sklearn.cluster import AgglomerativeClustering
import torch
import pickle
import numpy as np
import logging
from tqdm import tqdm
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import sys

from npc import sp_split_nw_comm

from python_utils.data import *
from utils.data_utils import load_config,read_nh_part,read_dict_dump
from utils.thread_utils import eval_ev_threads, get_graph_threads
import networkx as nx
from weight_functions import cos_time, cosine_similarity_custom,    get_nxe_weight_edges
from entity_graph import create_entity_graph
from collections import Counter
import cupy as cp
import os

config=load_config()

def get_connected_nodes(DG):
    return [DG.subgraph(cc).nodes for cc in nx.connected_components(DG.to_undirected())]

class GraphThreadGenerator:
    def __init__(self, is_td=False, alpha=0, is_ent=False, gamma=0, max_common_ent=5, max_ent_bet=5,
                 weight_threshold=0.8,method="npc",fwd_only=True, use_gpu=True):


        self.is_td=is_td
        self.alpha=alpha
        self.is_ent=is_ent
        self.gamma=gamma
        self.max_common_ent=max_common_ent
        self.max_ent_bet=max_ent_bet
        self.weight_threshold=weight_threshold
        self.method=method
        self.fwd_only=fwd_only  #If True: Edges are always defined foward in time (Set to FALSE for continous runs to extend already identified threads)
        self.use_gpu=use_gpu
        self.log=logging.getLogger()

    def find_inner_communities(self,DG, communities, th="auto", outdegree_limit=-1):
        res_graph = nx.DiGraph()
        comms = communities

        for c in comms:
            gres = sp_split_nw_comm(c, DG, th=th, outdegree_limit=outdegree_limit, logging=False, return_graph=True)
            res_graph = nx.compose(res_graph, gres)

        return res_graph

    def create_doc_graph(self,train_doc_ids,train_emb,train_emb_y=None,td_T=None,ent_graph=None,batch_size=128,DG=None,batch_start=0,verbose=True):

        pool_iter = []
        st = 0
        en = batch_size
        b = 0
        while st < train_emb.shape[0]:
            pool_iter.append((b, train_emb[st:en]))
            st = en
            en = st + batch_size
            b += 1

        edges = []

        if verbose:
            pbar=tqdm(total=train_emb.shape[0])
            print_newline_batch=1000
            print_newline_cnt=0

        if  self.fwd_only:
            train_emb_y=train_emb

        for idx, x in pool_iter:
            if self.is_td:
                item_weights=cos_time(x, train_emb_y, T=td_T, alpha=self.alpha, use_gpu=self.use_gpu)
            else:
                item_weights=1-cosine_similarity_custom(x,train_emb_y,use_gpu=self.use_gpu)


            if self.use_gpu:
                item_weights = cp.asnumpy(item_weights)
            temp = []
            candidates = np.argwhere(item_weights < self.weight_threshold)
            for s in range(item_weights.shape[0]):
                source_node_id = idx * batch_size + s + batch_start
                neighbours = candidates[candidates[:, 0] == s, 1]

                if self.fwd_only:
                    nxt_neighbours = neighbours[neighbours > source_node_id]
                    prev_neighbours=[]
                else:
                    nxt_neighbours = neighbours[neighbours > source_node_id]
                    prev_neighbours = neighbours[neighbours < source_node_id]

                source = train_doc_ids[source_node_id]

                if len(nxt_neighbours) > 0:
                    targets = [train_doc_ids[n] for n in nxt_neighbours]
                    if self.is_ent:
                        new_edges = get_nxe_weight_edges(ent_graph, source, targets, item_weights[s, nxt_neighbours], gamma=self.gamma, max_common_ent=self.max_common_ent, max_ent_between=self.max_ent_bet)
                    else:
                        new_edges = list(zip([source] * len(targets),targets, item_weights[s, nxt_neighbours]))
                    temp.extend(new_edges)

                if not self.fwd_only and len(prev_neighbours) > 0:
                    targets = [train_doc_ids[n] for n in prev_neighbours]
                    if self.is_ent:
                        new_edges = get_nxe_weight_edges(ent_graph, source, targets, item_weights[s, prev_neighbours], gamma=self.gamma, max_common_ent=self.max_common_ent, max_ent_between=self.max_ent_bet,reverse=True)
                    else:
                        new_edges = list(zip(targets,[source] * len(targets), item_weights[s, prev_neighbours]))
                    temp.extend(new_edges)


                if verbose:
                    pbar.update()
                    print_newline_cnt+=1

            edges.extend(temp)
            if verbose:
                pbar.set_description(f"Number of Edges: {len(edges)}")
                if print_newline_cnt>print_newline_batch:
                    self.log.warning("")
                    print_newline_cnt=0

        if DG is None:
            DG = nx.DiGraph()
        DG.add_weighted_edges_from(edges)
        return DG
    
    def get_threads(self,ofile,train_doc_ids, train_emb, passage_doc_ids, passage_emb, date_data,dt_feat=None,ent_graph=None,force_graph_create=False,verbose=True):
        doc2id = dict([(doc_id, idx) for idx, doc_id in enumerate(passage_doc_ids)])

        if self.is_td:
            train_emb = np.hstack((train_emb, dt_feat[:, None]))  # term features, dt_feat
            td_T = np.max(dt_feat) - np.min(dt_feat)
        else:
            td_T=None

        if self.use_gpu:
            train_emb = cp.asarray(train_emb)

        self.log.warning(f"Train Data: {train_emb.shape}. TD: {self.is_td}, ENT: {self.is_ent}, Continous: {not self.fwd_only}")

        if os.path.exists(ofile) and not force_graph_create:
            self.log.warning(f"DG found at: {ofile}")
            DG=nx.read_gpickle(ofile)
        else:
            self.log.warning(f"Creating Graph")
            DG=self.create_doc_graph(train_doc_ids,train_emb,td_T=td_T,ent_graph=ent_graph,verbose=verbose)
            nx.write_gpickle(DG,ofile)

        self.log.warning(f"Running {self.method.upper()}")
        if self.method=="npc":
            res_graph = self.find_inner_communities(DG, [DG.nodes], th="auto", outdegree_limit=1)
            communities = get_connected_nodes(res_graph)
        elif self.method in ["louvain","leiden"]:
            res_graph=None
            from cdlib import algorithms

            for k,v in tqdm(nx.get_edge_attributes(DG,"weight").items()):
                if v<0:
                    DG.add_edge(k[0],k[1],weight=0)
            UDG=DG.to_undirected()
            
            communities = algorithms.louvain(UDG,weight="weight") if self.method == "louvain" else algorithms.leiden(UDG,weights="weight")
            communities = communities.communities
    

        self.log.warning(f"Generating Threads")
        threads, thread_similarity= get_graph_threads(communities, date_data, passage_emb, doc2id)

        return res_graph, threads, thread_similarity


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str, default="minilm",help='embedding model: minilm; distilroberta;tfidf')
    parser.add_argument('-a', action='store',dest='alpha', type=float,default=0,help='time decay factor')
    parser.add_argument('-g', action='store', dest='gamma', type=float, default=0, help='entity factor')
    parser.add_argument('--mce', action='store', dest='mCE', type=float, default=5, help='max common entities')
    parser.add_argument('--meb', action='store', dest='mEB', type=float, default=5, help='max entity between')
    parser.add_argument('-w', action='store', dest='weight_threshold', type=float, default=0.7,help='edge weight threshold')

    parser.add_argument('-m', action="store", dest="method", type=str,default="npc", help='method name: npc, samp, louvain, leiden')

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

    # #######################################################################################################
    # #######################################################################################################

    threads,thread_similarity=[],[]
    res_graph = nx.DiGraph()
    pattern = f"{args.emb}_w{args.weight_threshold}"
    if args.td:
        pattern += f"_td{args.alpha}"
    if args.ent:
        pattern += f"_ent{args.gamma}"

    for p_id, (p_st, p_en) in nh_part.items():
        log.warning(f"Part: {p_id + 1}")
        train_doc_ids = pnct_nh_5w1h_passage_doc_ids[p_st:p_en]
        train_5w1h_emb=nh_5w1h_emb[p_st:p_en]
        train_dt_emb = dt_feat[p_st:p_en]
        k= len(set(thread_true_labels[p_st:p_en]))

        if args.ent:
            G=create_entity_graph(nh_entity_dict,train_doc_ids)
        else:
            G=None

        graph_ofile = f"{config.graph_root_dir}/passage_graphs/{pattern}_graph_{p_id}.gp"
        use_gpu = torch.cuda.is_available()

        gt = GraphThreadGenerator(is_td=args.td, alpha=args.alpha, is_ent=args.ent, gamma=args.gamma,
                                  max_common_ent=args.mCE, max_ent_bet=args.mEB, weight_threshold=args.weight_threshold,
                                  method=args.method,use_gpu=use_gpu)

        DG_comm,part_threads, part_thread_similarity = gt.get_threads(ofile=graph_ofile, train_doc_ids=train_doc_ids, train_emb=train_5w1h_emb,
                                                               passage_doc_ids=pnct_nh_5w1h_passage_doc_ids, passage_emb=nh_passage_emb,
                                                               date_data=nh_date_data, dt_feat=train_dt_emb,ent_graph=G,force_graph_create=False)

        threads.extend(part_threads)
        thread_similarity.append(part_thread_similarity)
        if args.method=="npc":
            res_graph=nx.compose(res_graph,DG_comm)



    thread_similarity = np.hstack(thread_similarity)

    pattern=f"{args.method}_{pattern}"

    ofile = f"{config.threads_dir}/threads_graph_{pattern}.p"
    with open(ofile, "wb") as f:
        pickle.dump(threads, f)
    log.warning(f"\tThreads stored at: {ofile}")
    np.savez_compressed(f"{config.threads_dir}/thread_sim_graph_{pattern}.npz", thread_similarity)

    nx.write_gpickle(res_graph, f"{config.graph_root_dir}/passage_graphs/gpc_{pattern}_graph.gp")

    res = f"\nThread Count:{len(threads)}"
    res += f"\nAvg Length:{np.mean([len(i) for i in threads])}"
    res += f"\nCosine Score: {np.mean(thread_similarity[np.logical_not(np.isnan(thread_similarity))])}"
    log.warning(res)

    eval_ev_threads(threads, doc2storyid, pnct_nh_5w1h_passage_doc_ids, thread_true_labels, min_len=0,return_dict=False, print_res=True)

