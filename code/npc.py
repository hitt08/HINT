import pickle
import argparse
import time
from tqdm import tqdm
from utils.thread_utils import eval_ev_threads
import networkx as nx
import numpy as np
import logging
import sys

from python_utils.data import *
from python_utils.time import *
from utils.data_utils import read_nh_part
from passage_graph import construct_passage_graph
log=logging.getLogger()

def split_graph(DG, th=70, outdegree=-1):   ##Remove siginifcantly weaker links in a community
    log=logging.getLogger()
    rem_edges = []
    for c in nx.connected_components(DG.to_undirected()):
        if len(c) > 1:
            res = DG.subgraph(c)
            A = list(res.edges.data('weight'))
            W = np.array(A)[:, 2].astype(float)

            if type(th) == str and th == "auto":
                p25, p75 = np.percentile(W, [25, 75])
                th = 1.5 * (p75 - p25) * p75
                # log.warning(f"NPC TH= {th}")
                for i in np.argwhere(W > np.percentile(W, th))[:, 0]:
                    if DG.out_degree[A[i][1]] > outdegree:
                        rem_edges.extend([A[i][:2]])
            else:
                rem_edges.extend([A[i][:2] for i in np.argwhere(W > np.percentile(W, th))[:, 0] if
                                  DG.out_degree[A[i][1]] > outdegree])


    DG.remove_edges_from(rem_edges)

    return DG

def sp_split_nw_comm(comm, DG, th=70, outdegree_limit=-1, return_graph=False,logging=False):  ##NPC main function
    graph = DG.subgraph(comm)
    id2node = dict([(i, n) for i, n in enumerate(graph.nodes)])


    A = nx.adjacency_matrix(graph).tocsc()
    trim_nodes = np.nonzero((A.indptr[1:] - A.indptr[:-1]) > 1)[0]

    if logging:
        log.warning("trimming edges")
        pbar = tqdm(total=len(trim_nodes))

    for y in trim_nodes:
        a = A[:, y].nonzero()[0]
        srt_ids = np.argsort(A[a, y].toarray()[:, 0])
        A[a[srt_ids[1:]], y] = 0
        if logging:
            pbar.update()

    if logging:
        pbar.close()
        log.warning("reconstructing graph")

    A.eliminate_zeros()
    A = A.tocsr()

    res = nx.relabel_nodes(nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph), id2node)
    res = split_graph(res, th=th, outdegree=outdegree_limit)


    if return_graph:
        return res
    else:
        temp = [res.subgraph(c).nodes for c in nx.connected_components(res.to_undirected())]
        return temp


def split_nw_comm(comm, DG, th=70, outdegree_limit=-1, return_graph=False,
                  logging=False):
    graph = DG.subgraph(comm)
    id2node = dict([(i, n) for i, n in enumerate(graph.nodes)])


    A = nx.to_numpy_array(graph)
    trim_nodes = np.nonzero(np.count_nonzero(A, axis=0) > 1)[0]
    if logging:
        log.warning("trimming edges")
        pbar = tqdm(total=len(trim_nodes))

    for y in trim_nodes:
        a = np.nonzero(A[:, y])[0]
        srt_ids = np.argsort(A[a, y])
        A[a[srt_ids[1:]], y] = 0
        if logging:
            pbar.update()

    if logging:
        pbar.close()
        log.warning("reconstructing graph")
    res = nx.relabel_nodes(nx.from_numpy_array(A, create_using=nx.DiGraph), id2node)
    res = split_graph(res, th=th, outdegree=outdegree_limit)

    if return_graph:
        return res
    else:
        temp = [res.subgraph(c).nodes for c in nx.connected_components(res.to_undirected())]
        return temp



def find_inner_communities(file_ptrn, communities, th=70, outdegree_limit=-1, sparse=False, logging=False, save=False):
    res = []
    r_len = 0
    res_graph = nx.DiGraph()
    for p_id, comms in enumerate(communities):
        DG = nx.read_gpickle(f"{file_ptrn}_{p_id}.gp")
        temp = []
        if not logging:
            pbar = tqdm(total=len(comms))
        else:
            log.warning(f"Part:{p_id + 1}")
        for c in comms:
            if sparse:
                if save:
                    gres = sp_split_nw_comm(c, DG, th=th, outdegree_limit=outdegree_limit, logging=logging,
                                            return_graph=True)
                    res_graph = nx.compose(res_graph, gres)
                    temp.extend([gres.subgraph(cc).nodes for cc in nx.connected_components(gres.to_undirected())])
                else:
                    temp.extend(sp_split_nw_comm(c, DG, th=th, outdegree_limit=outdegree_limit, logging=logging))

            else:
                temp.extend(split_nw_comm(c, DG, th=th, outdegree_limit=outdegree_limit, logging=logging))

            if not logging:
                pbar.update()

        if not logging:
            pbar.close()

        res.append(temp)
        r_len += len(temp)
        log.warning(f"{len(temp)}, {r_len}")
    if save:
        nx.write_gpickle(res_graph, f"{file_ptrn}_th{th}_o{outdegree_limit}.gp")

    return res