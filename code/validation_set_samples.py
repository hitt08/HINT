import argparse
import sys
import torch
from graph_threads import GraphThreadGenerator
from python_utils.data import *
from python_utils.time import *
import numpy as np
import logging
from scipy.sparse import load_npz as sparse_load_npz
import pickle
import time
from utils.data_utils import load_config,read_dict_dump,read_nh_part
from entity_graph import create_entity_graph
from hac import get_hac_threads
from utils.keywords import get_threads_keywords
from utils.diversity import Thread_Diversity
from utils.coherence import get_coherence
from utils.thread_utils import eval_ev_threads
import networkx as nx
log = logging.getLogger(__name__)


def perform_sampling(points, docs_ids,doc2storyid,story2docs, size=100, percent=True, seed=0):
    stories = sorted(list(set([doc2storyid[d] for d in docs_ids if len(story2docs[doc2storyid[d]]) > 2])))

    if percent and size <= 100:
        size = round(size * len(stories) / 100)

    np.random.seed(seed)
    sample_story_ids = np.random.choice(range(len(stories)), size, replace=False)

    sample_stories = np.asarray(stories, dtype=object)[sample_story_ids]

    doc_set = set(docs_ids)
    sample_docs = [d for s in sample_stories for d in story2docs[s] if d in doc_set]

    np.random.shuffle(sample_docs)

    sample_doc_arg = [docs_ids.index(d) for d in sample_docs]

    sample_points = points[sample_doc_arg]

    return sample_points, sample_docs, sample_stories

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str, default="minilm", help='embedding model tfidflsa, minilm or distilroberta')
    parser.add_argument('-p', action='store', dest='pct', type=int,default=1000,help='data percentage')
    parser.add_argument('-s', action='store', dest='seed', type=int,default=1, help='seed')
    parser.add_argument('-i', action='store', dest='part_idx', type=int,required=True, help='part idx')

    parser.add_argument('-c', action="store", dest="cpus", type=int, default=1,help='num processes')

    parser.add_argument('-a', action='store',dest='alpha', type=float,default=10,help='time decay factor')
    parser.add_argument('-g', action='store',dest='gamma', type=float,default=0.5,help='entity factor')
    parser.add_argument('--mce', action='store', dest='mCE', type=float, default=5, help='max common entities')
    parser.add_argument('--meb', action='store', dest='mEB', type=float, default=5, help='max entity between')
    parser.add_argument('-w', action='store', dest='weight_threshold', type=float, default=0.7,help='edge weight threshold')

    parser.add_argument('-m', action="store", dest="method", type=str,default="hac", help='method name: hac, samp, louvain, leiden, npc')

    parser.add_argument('--td', dest='td', action='store_true', help='time decay')
    parser.set_defaults(td=False)
    parser.add_argument('--ent', dest='ent', action='store_true', help='entity similarity')
    parser.set_defaults(ent=False)

    parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='use gpu')
    parser.set_defaults(use_gpu=False)


    args = parser.parse_args()

    config=load_config()
    log.warning(args)

    ##### Read Data ######
    with open(f"{config.graph_root_dir}/{config.doc_ids_file}", "rb") as f:
        pnct_nh_5w1h_passage_doc_ids = pickle.load(f)

    with np.load(f"{config.emb_root_dir}/nh_5w1h_{args.emb}_emb.npz") as data:
        nh_5w1h_emb = data["arr_0"]

    with open(f"{config.root_dir}/5w1h/nh_date_data.p", "rb") as f:
        nh_date_data = pickle.load(f)

    with np.load(f"{config.emb_root_dir}/nh_{args.emb}_emb.npz") as data:
        nh_passage_emb = data["arr_0"]

    nh_part = read_nh_part(config.root_dir)

    doc2storyid = read_dict(f"{config.root_dir}/newshead/processed/doc2storyid_filtered.txt")
    story2docs = {}
    for k, v in doc2storyid.items():
        if v not in story2docs:
            story2docs[v] = []
        story2docs[v].append(k)
    #############

    p_st,p_en=nh_part[args.part_idx]
    train_doc_ids = pnct_nh_5w1h_passage_doc_ids[p_st:p_en]
    train_5w1h_emb = nh_5w1h_emb[p_st:p_en]


    log.warning(f"PCT:{args.pct} SEED: {args.seed}")
    sample_train_5w1h_emb, sample_docs, sample_stories = perform_sampling(train_5w1h_emb, train_doc_ids,doc2storyid,story2docs, size=args.pct,seed=args.seed)
    log.warning(f"\tDocs:{len(sample_docs)},Stories:{len(sample_stories)}")
    st = time.time()

    if args.ent:
        nh_entity_dict = read_dict_dump(f"{config.graph_root_dir}/{config.entity_dict_file}")
        G = create_entity_graph(nh_entity_dict, sample_docs)
        log.warning(f"Entity Graph: Number of Nodes= {G.number_of_nodes()}. Number of Edges= {G.number_of_edges()}")
    else:
        G = None

    if args.td:
        with np.load(f"{config.graph_root_dir}/{config.date_features_file}") as data:
            dt_feat = data["arr_0"]
        tmp_args=[pnct_nh_5w1h_passage_doc_ids.index(d) for d in sample_docs]
        sample_dt_emb=dt_feat[tmp_args]
    else:
        sample_dt_emb=None

    pattern = f"{args.emb}_m{args.method}_w{args.weight_threshold}"
    if args.td:
        pattern += f"_td{args.alpha}"
    if args.ent:
        pattern += f"_ent{args.gamma}"
    if not args.td and not args.ent and args.method=="hac":
        pattern += f"_ward"

    pattern+=f"_p{args.pct}_s{args.seed}_{args.part_idx}"

    if args.method=="hac":
        ofile = f"{config.validation_dir}/sample_hac_{pattern}.p"
        sample_threads, sample_thread_sim = get_hac_threads(ofile, len(sample_stories), sample_docs, sample_train_5w1h_emb, pnct_nh_5w1h_passage_doc_ids,
                                                               nh_passage_emb, nh_date_data, is_td=args.td,
                                                               a=args.alpha, dt_feat=sample_dt_emb, is_ent=args.ent,
                                                               g=args.gamma, ent_graph=G, max_common_ent=args.mCE,
                                                               max_ent_bet=args.mEB,weight_threshold=args.weight_threshold)

    else:
        ofile = f"{config.validation_dir}/sample_graph_{pattern.replace(args.method,'npc')}.gp"
        use_gpu = torch.cuda.is_available() and args.use_gpu
        gt = GraphThreadGenerator(is_td=args.td, alpha=args.alpha, is_ent=args.ent, gamma=args.gamma,
                                  max_common_ent=args.mCE, max_ent_bet=args.mEB, weight_threshold=args.weight_threshold,
                                  method=args.method,use_gpu=use_gpu)

        DG_comm,sample_threads, sample_thread_sim = gt.get_threads(ofile=ofile, train_doc_ids=sample_docs, train_emb=sample_train_5w1h_emb,
                                                               passage_doc_ids=pnct_nh_5w1h_passage_doc_ids, passage_emb=nh_passage_emb,
                                                               date_data=nh_date_data, dt_feat=sample_dt_emb,ent_graph=G,force_graph_create=False)

        if args.method=="npc":
            nx.write_gpickle(DG_comm, f"{config.validation_dir}/sample_gpc_graph_{pattern}.gp")

    log.warning(f"\tTime Take: {fmt_time(time.time() - st)}")

    ofile = f"{config.validation_dir}/sample_threads_{pattern}.p"
    with open(ofile, "wb") as f:
        pickle.dump(sample_threads, f)

    ofile = f"{config.validation_dir}/sample_thread_sim_{pattern}.npz"
    np.savez_compressed(ofile, sample_thread_sim)

    sample_doc_labels = [doc2storyid[d] for d in sample_docs]
    eval_ev_threads(sample_threads, doc2storyid, sample_docs, sample_doc_labels, min_len=0, return_dict=False,print_res=True)
    log.warning(f"")

    log.warning(f"Computing Divergence")
    nh_tk_pnct_collection = read_dict_dump(f"{config.root_dir}/newshead/processed/pnct.json.gz")
    st = time.time()
    thread_div = Thread_Diversity(sample_threads, nh_tk_pnct_collection, nprocesses=args.cpus)
    div = thread_div.get_diversity()
    log.warning(f"\n\tDiv: {np.mean(div[np.logical_not(np.isnan(div))])}")

    with open(f"{config.validation_dir}/sample_divergence_{pattern}.p", "wb") as f:
        pickle.dump(div, f)

    log.warning(f"\tTime Take: {fmt_time(time.time() - st)}")

    pnct_nh_tfidf_train_matrix=sparse_load_npz(f"{config.emb_root_dir}/{config.passage_tfidf_emb_file}")

    with open(f"{config.emb_root_dir}/{config.passage_tfidf_features_file}", "rb") as f:
        pnct_nh_features=pickle.load(f)

    thread_df = get_threads_keywords(sample_threads, nh_tk_pnct_collection, pnct_nh_5w1h_passage_doc_ids, pnct_nh_features, pnct_nh_tfidf_train_matrix,f"{config.validation_dir}/{pattern}_", top_n=5, force=False, nprocesses=args.cpus)

    tmp = np.asarray(get_coherence(sample_threads, thread_df, nh_tk_pnct_collection))
    log.warning(f"\n\tCoh: {np.mean(tmp[np.logical_not(np.isnan(tmp))])}")

    with open(f"{config.validation_dir}/sample_coherence_{pattern}.p", "wb") as f:
        pickle.dump(tmp, f)