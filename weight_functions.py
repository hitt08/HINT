import cupy as cp
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
import numpy as np
import networkx as nx
from collections import Counter

def row_norms(X):
    norms = cp.einsum("ij,ij->i", X, X)
    cp.sqrt(norms, norms)
    X /= norms[:, cp.newaxis]
    return X

def cosine_similarity_custom(a,b,use_gpu=True):
    if use_gpu:
        a=row_norms(a)
        b=row_norms(b).T
        return cp.dot(a, b)
    else:
        return cosine_similarity(a,b)

def time_decay(x,y,T=None,alpha=0.75,use_gpu=True): #Similarity #High alpha, exponentially penalises large time gaps (very low similarity)
    np_mod=cp if use_gpu else np
    if T is None:
        temp=np_mod.hstack((x,y))
        T=np_mod.max(temp)-np_mod.min(temp)
    return np_mod.exp(-alpha*np_mod.abs(x[:,None]-y)/T)

def cos_time(x,y,T=None,alpha=10,use_gpu=True): #Distance
    return 1-cosine_similarity_custom(x[:,:-1],y[:,:-1],use_gpu)*time_decay(x[:,-1],y[:,-1],T,alpha,use_gpu)


# def get_nxe_weight(G, s, t, T, gamma=1):
#     is_ent_weight = True
#     try:
#         p = len(list(nx.all_simple_paths(G, s, t, cutoff=2)))
#         if p == 0:
#             p = nx.shortest_path_length(G, s, t)
#             is_ent_weight = False
#             if p == 0:
#                 p = G.number_of_nodes()
#
#     except (nx.NodeNotFound, nx.NetworkXNoPath):
#         p = G.number_of_nodes()
#         is_ent_weight = False
#
#
#
#     if is_ent_weight:
#         # return gamma * 0.5 * (1 + p / T)
#         return 0.5 * (1+(1-np.exp(-gamma*(p/T))))#1 + np.log(p) / np.log(T))
#         # return gamma*0.5 * (1 + np.log(p) / np.log(T))
#     else:
#         return 0.5 * (1 - np.log(p / 2) / np.log(G.number_of_nodes()))


def get_nxe_weight_edges(G, s, targets,item_weights, gamma=1,max_common_ent=5,max_ent_between=5,reverse=False):

    temp = get_all_direct_paths(G, s, targets)

    # temp=dict([(t, 0) for t in targets])
    # for i in nx.all_simple_paths(G, s, targets, cutoff=2):
    #     temp[i[-1]] += 1

    targets_k,targets_p=[],[]
    for t in targets:
        targets_k.append(t)
        targets_p.append(temp[t])
    targets_k = np.asarray(targets_k)
    targets_p = np.asarray(targets_p)

    is_path_weight = targets_p == 0
    ent_weight_args = np.argwhere(is_path_weight == False).squeeze(-1)
    path_weight_args = np.argwhere(is_path_weight).squeeze(-1)

    ent_weights = targets_p[ent_weight_args]
    # path_weights = np.asarray(get_shortest_path_lengths(G, s, targets_k[path_weight_args], cutoff=max_ent_between * 2))

    path_weights = np.zeros_like(path_weight_args)
    for i, t in enumerate(targets_k[path_weight_args]):
        try:
            path_weights[i] = shortest_path_length(G, s, t,cutoff=max_ent_between*2)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            path_weights[i] = max_ent_between*2#G.number_of_nodes()

    ent_weights  = 1 - (1 - item_weights[ent_weight_args]) * 0.5 * (1 + (1 - np.exp(-gamma * (ent_weights / max_common_ent))))
    path_weights = 1 - (1 - item_weights[path_weight_args]) * 0.5 * np.exp(-gamma * ((path_weights/2) / max_ent_between))

    if reverse: #target->source
        res = list(zip(targets_k[ent_weight_args],[s] * len(ent_weight_args), ent_weights))
        res += list(zip(targets_k[path_weight_args],[s] * len(path_weight_args), path_weights))
    else: #source->target
        res = list(zip([s] * len(ent_weight_args), targets_k[ent_weight_args], ent_weights))
        res += list(zip([s] * len(path_weight_args), targets_k[path_weight_args], path_weights))

    return res


def get_all_direct_paths(G,source,targets=None):
    if targets is not None:
        target_sets=set(targets)
    direct_paths=[]
    if targets is not None:
        for i in G.neighbors(source):
            direct_paths.extend([d for d in G.neighbors(i) if d in target_sets and d !=source])
    else:
        for i in G.neighbors(source):
            direct_paths.extend([d for d in G.neighbors(i) if d != source])

    return Counter(direct_paths)


def get_shortest_path_lengths(G,source,targets,cutoff=10):
    c=-1
    node2level={}
    res=dict([(i,cutoff) for i in targets])
    tmp_trgs=set(list(res.keys()))

    for s,neighbours in nx.breadth_first_search.bfs_successors(G,source,cutoff):
        if s not in node2level:
            c+=1
            node2level[s]=c

        for n in neighbours:
            if n not in node2level:
                node2level[n]=node2level[s]+1

        n_set=set(neighbours).intersection(tmp_trgs)
        tmp_trgs=tmp_trgs.difference(n_set)
        for x in n_set:
            res[x]=node2level[x]

        if len(tmp_trgs)==0:
            break
    return [res[i] for i in targets]

def shortest_path_length(G, s, targets, cutoff=6):
    pred, succ, w = bidirectional_pred_succ(G, s, targets, cutoff=cutoff)
    c = 0
    prev = w
    # from source to w
    while w is not None:
        c += 1
        w = pred[w]
    # from w to target
    w = succ[prev]
    while w is not None:
        c += 1
        w = succ[w]

    return c - 1

def bidirectional_pred_succ(G, source, target, cutoff=None):
    if target == source:
        return ({target: None}, {source: None}, source)

    # handle either directed or undirected
    if G.is_directed():
        Gpred = G.pred
        Gsucc = G.succ
    else:
        Gpred = G.adj
        Gsucc = G.adj

    # predecesssor and successors in search
    pred = {source: None}
    succ = {target: None}

    # initialize fringes, start with forward
    forward_fringe = [source]
    reverse_fringe = [target]

    c = 0
    while forward_fringe and reverse_fringe:
        if cutoff and c >= cutoff + 1:
            break
        c += 1
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc[v]:
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:  # path found
                        return pred, succ, w
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred[v]:
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:  # found path
                        return pred, succ, w

    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
