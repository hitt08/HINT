import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cupy as cp
import numpy as np
import sys
import pickle
import logging
import argparse
from graph_threads import GraphThreadGenerator,get_connected_nodes
from entity_graph import create_entity_graph
from utils.data_utils import load_config,read_nh_part,read_dict_dump
from tqdm import tqdm
import networkx as nx
from python_utils.data import *
from python_utils.time import *
from weight_functions import cos_time, get_nxe_weight_edges
from utils.thread_utils import eval_ev_threads,eval_filter_threads, get_graph_threads
import datetime
from collections import Counter
import time
import torch
log=logging.getLogger()

config=load_config()



class ContinousGraphThreadGenerator:

    def __init__(self, opath,name, fivew1h_emb, passage_doc_ids,passage_emb, date_data,doc2storyid, thread_true_labels, filter_params,
                 history_days,incremental_days,
                 is_td=False, dt_feat=None, alpha=0, 
                 is_ent=False,entity_dict=None, gamma=0, max_common_ent=5, max_ent_bet=5,
                 weight_threshold=0.8, use_gpu=True):
        
        self.is_td=is_td
        self.alpha=alpha
        self.is_ent=is_ent
        self.gamma=gamma
        self.max_common_ent=max_common_ent
        self.max_ent_bet=max_ent_bet
        self.weight_threshold=weight_threshold
        self.use_gpu=use_gpu
        self.log=logging.getLogger()

        self.gt = GraphThreadGenerator(is_td=args.td, alpha=args.alpha, is_ent=args.ent, gamma=args.gamma,
                                  max_common_ent=args.mCE, max_ent_bet=args.mEB, weight_threshold=args.weight_threshold,
                                  method="npc",use_gpu=use_gpu)

        
        self.train_emb = np.hstack((fivew1h_emb, dt_feat[:, None]))  if self.is_td else fivew1h_emb
        if use_gpu:
            self.train_emb = cp.asarray(self.train_emb)

        self.entity_dict = entity_dict
        
        
        self.history_days=history_days
        self.incremental_days=incremental_days
        
        self.passage_doc_ids = passage_doc_ids
        self.passage_emb = passage_emb
        self.date_data=date_data
        self.doc2storyid=doc2storyid
        self.thread_true_labels=thread_true_labels

        self.doc2id = dict([(doc_id, idx) for idx, doc_id in enumerate(passage_doc_ids)])

        self.date_data=date_data
        self.date2doc_count = Counter([date_data[i].date() for i in passage_doc_ids])
        self.min_date = min(self.date2doc_count.keys())
        self.max_date = max(self.date2doc_count.keys())

        log.warning(f"Total Days: {self.max_date-self.min_date}")

        self.filter_params=filter_params #Candidate Selection Params

        self.opath=opath
        self.pattern = f"{name}_w{weight_threshold}"
        if is_td:
            self.pattern += f"_td{alpha}"
        if is_ent:
            self.pattern += f"_ent{gamma}"
        
        self.pattern += f"_h{self.history_days}"

        self.dt_eval_dict=[]
    
    
    def filter_dt_doc_ids(self,st_date, days):
        _, dt_en = self.get_st_en_passage_ids(st_date, days)

        pool_pt_passage_data = set(self.passage_doc_ids[:dt_en])

        return pool_pt_passage_data

    def get_st_en_passage_ids(self,st_date, days):
        new_date = self.min_date
        st_idx = 0
        while new_date < st_date:
            if new_date in self.date2doc_count:
                st_idx += self.date2doc_count[new_date]
            new_date += datetime.timedelta(days=1)

        en_idx = st_idx
        for i in range(days):
            new_date = st_date + datetime.timedelta(days=i)
            if new_date in self.date2doc_count:
                en_idx += self.date2doc_count[new_date]

        return st_idx, en_idx

    def get_filtered_passage_data(self,start_date,days):
        passage_data = self.filter_dt_doc_ids(start_date, days=days)
        passage_doc_ids_filtered, thread_true_labels_filtered = [], []
        for i in range(len(self.passage_doc_ids)):
            if self.passage_doc_ids[i] in passage_data:
                passage_doc_ids_filtered.append(self.passage_doc_ids[i])
                thread_true_labels_filtered.append(self.thread_true_labels[i])
        
        return passage_doc_ids_filtered, thread_true_labels_filtered

    def history_run(self,verbose=True):

        st_time=time.time()
        DG_HIST,HIST_next_date,g_time,c_time=self.update_communities(self.min_date,days=self.history_days,fwd_only=True,verbose=verbose)

        t_st=time.time()
        communities = get_connected_nodes(DG_HIST)
        if verbose:
            self.log.warning(f"Generating Threads")
        threads, thread_similarity= get_graph_threads(communities, self.date_data, self.passage_emb, self.doc2id)    



        passage_doc_ids_filtered, thread_true_labels_filtered=self.get_filtered_passage_data(self.min_date,self.history_days)

        eval_dict=eval_filter_threads([],self.filter_params,None,threads,thread_similarity,self.doc2storyid,passage_doc_ids_filtered,thread_true_labels_filtered,print_res=verbose)[0]
        t_time=time.time()-t_st

        en_time=time.time()-st_time
        eval_dict["G_Time"] = g_time
        eval_dict["C_Time"]=c_time
        eval_dict["T_Time"]=t_time
        eval_dict["Time"]=en_time
        self.dt_eval_dict=[{str(self.min_date):eval_dict}]

        ofile = f"{self.opath}/threads_hist_{self.pattern}.p"
        with open(ofile, "wb") as f:
            pickle.dump(threads, f)

        if verbose:
            log.warning(f"\tThreads stored at: {ofile}")

        np.savez_compressed(f"{self.opath}/thread_sim_hist_{self.pattern}.npz", thread_similarity)

        nx.write_gpickle(DG_HIST, f"{self.opath}/hist_{self.pattern}_graph.gp")

        write_json_dump(f"{self.opath}/eval_dict_{self.pattern}_hist.jsonl", self.dt_eval_dict,mode="w")
        
        return DG_HIST,HIST_next_date,threads, thread_similarity


    def incremental_run(self,start_date,DG_INCR,verbose=False):
        total_days = (self.max_date - start_date).days

        eval_batch = 7 if self.incremental_days<7 else self.incremental_days
        c = 0

        save_batch = 100
        sv=0

        prev_date = str(self.min_date)
        st_time=time.time()
        g_time,c_time=0,0
        with tqdm(total=total_days) as pbar:
            while total_days > 0:
                pdesc = str(start_date)
                pdesc += f'| DG:{DG_INCR.number_of_nodes()}, '
                pdesc += f'| S_H:{round(self.dt_eval_dict[-1][prev_date]["sel_h"], 3)}, '
                pdesc += f'S_N:{round(self.dt_eval_dict[-1][prev_date]["sel_nmi"], 3)}, '
                pdesc += f'A_H:{round(self.dt_eval_dict[-1][prev_date]["all_h"], 3)}, '
                pdesc += f'A_N:{round(self.dt_eval_dict[-1][prev_date]["all_nmi"], 3)}, '
                pdesc += f'GT:{fmt_time(self.dt_eval_dict[-1][prev_date]["G_Time"])}, '
                pdesc += f'T:{fmt_time(self.dt_eval_dict[-1][prev_date]["Time"])}'
                pbar.set_description(pdesc)

                st_time=time.time()
                DG_INCR,next_date,g_time,c_time=self.update_communities(start_date,days=self.incremental_days,DG=DG_INCR,fwd_only=False,verbose=verbose)

                c += self.incremental_days
                sv += self.incremental_days
                if c>=eval_batch or sv>=save_batch:
                    c = 0

                    t_st=time.time()
                    communities = get_connected_nodes(DG_INCR)

                    if verbose:
                        self.log.warning(f"Generating Threads")
                    threads, thread_similarity= get_graph_threads(communities, self.date_data, self.passage_emb, self.doc2id,verbose=verbose)    
                    passage_doc_ids_filtered, thread_true_labels_filtered=self.get_filtered_passage_data(start_date,self.incremental_days)
                    eval_dict=eval_filter_threads([],self.filter_params,None,threads,thread_similarity,self.doc2storyid,passage_doc_ids_filtered,thread_true_labels_filtered,print_res=verbose)[0]

                    t_time=time.time()-t_st
                    en_time=time.time()-st_time

                    eval_dict["G_Time"] = g_time
                    eval_dict["C_Time"]=c_time
                    eval_dict["T_Time"]=t_time
                    eval_dict["Time"]=en_time

                    self.dt_eval_dict.append({str(start_date): eval_dict})
                    write_json_dump(f"{self.opath}/eval_dict_{self.pattern}_d{self.incremental_days}_incr.jsonl", self.dt_eval_dict[-1:],mode="a")

                    prev_date = str(start_date)
                    log.warning("")


                if sv>=save_batch:
                    sv=0
                    nx.write_gpickle(DG_INCR, f"{self.opath}/incr_{self.pattern}_d{self.incremental_days}_graph.gp")



                total_days -= self.incremental_days
                start_date = next_date
                pbar.update(self.incremental_days)




        t_st=time.time()
        communities = get_connected_nodes(DG_INCR)
        threads, thread_similarity= get_graph_threads(communities, self.date_data, self.passage_emb, self.doc2id)    
        eval_dict=eval_filter_threads([],self.filter_params,None,threads,thread_similarity,self.doc2storyid,self.passage_doc_ids,self.thread_true_labels,print_res=True)[0]

        t_time=time.time()-t_st
        en_time=time.time()-st_time

        eval_dict["G_Time"] = g_time
        eval_dict["C_Time"]=c_time
        eval_dict["T_Time"]=t_time
        eval_dict["Time"]=en_time

        self.dt_eval_dict.append({str(start_date): eval_dict})
        write_json_dump(f"{self.opath}/eval_dict_{self.pattern}_d{self.incremental_days}_incr.jsonl", self.dt_eval_dict[-1:],mode="a")

        nx.write_gpickle(DG_INCR, f"{self.opath}/incr_{self.pattern}_d{self.incremental_days}_graph.gp")

        ofile = f"{self.opath}/threads_incr_{self.pattern}_d{self.incremental_days}.p"
        with open(ofile, "wb") as f:
            pickle.dump(threads, f)
        log.warning(f"\tThreads stored at: {ofile}")

        np.savez_compressed(f"{self.opath}/thread_sim_incr_{self.pattern}_d{self.incremental_days}.npz", thread_similarity)
       
        return DG_INCR,threads, thread_similarity


    def update_communities(self,st_date,days=1,DG=None,fwd_only=False,verbose=True):
        dt_st, dt_en = self.get_st_en_passage_ids(st_date, days)

        train_emb = self.train_emb[dt_st:dt_en]
        train_ids = self.passage_doc_ids[:dt_en]
        train_emb_y=None if fwd_only else self.train_emb[:dt_en]

        if self.is_td:
            td_T = self.filter_params["d"][-1]*24*60*60#np.max(self.dt_feat) - np.min(self.dt_feat)
        else:
            td_T=None


        if verbose:
            self.log.warning(f"Train Data: {train_emb.shape}. TD: {self.is_td}, ENT: {self.is_ent}, Continous: {not fwd_only}")

        ent_graph=create_entity_graph(self.entity_dict,self.passage_doc_ids[:dt_en],verbose=verbose)
        
        st=time.time()
        
        self.gt.fwd_only=fwd_only
        
        DG=self.gt.create_doc_graph(train_ids,train_emb,train_emb_y,td_T,ent_graph,DG=DG,batch_start=dt_st,verbose=verbose)
        g_time = time.time()-st
        st=time.time()
        if verbose:
            self.log.warning(f"Running NPC")
        res_graph = self.gt.find_inner_communities(DG, [DG.nodes], th="auto", outdegree_limit=1)
        c_time = time.time()-st
   
        return res_graph,  self.date_data[self.passage_doc_ids[:dt_en+1][-1]].date(), g_time, c_time






if __name__=="__main__":
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

    parser.add_argument('-t', action='store', dest='hist_days', type=int, default=30, help='number of days for history run')
    parser.add_argument('-d', action='store', dest='incr_days', type=int, default=1, help='number of days for incremental run')

    parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='use gpu')
    parser.set_defaults(use_gpu=False)

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
    use_gpu = torch.cuda.is_available() and args.use_gpu
    run_name=args.emb if use_gpu else f"cpu_{args.emb}"
    filter_params={'l': [3, 100],  's': [0.2, 0.8],  'd': [0, 120]}
    cont_gt=ContinousGraphThreadGenerator(config.continous_root_dir,run_name, nh_5w1h_emb, pnct_nh_5w1h_passage_doc_ids,nh_passage_emb, nh_date_data,doc2storyid, thread_true_labels, filter_params,
                history_days=args.hist_days,incremental_days=args.incr_days,
                is_td=args.td, dt_feat=dt_feat, alpha=args.alpha, 
                is_ent=args.ent,entity_dict=nh_entity_dict, gamma=args.gamma, max_common_ent=args.mCE, max_ent_bet=args.mEB,
                weight_threshold=args.weight_threshold, use_gpu=use_gpu)


    # 30 Days
    total_hist_st = time.time()
    log.warning(f"Historical Run: {args.hist_days}")
    DG_HIST,HIST_next_date,_, _ = cont_gt.history_run(verbose=True)
    log.warning(f"\nTotal History Run Time: {fmt_time(time.time()-total_hist_st)}")

    dt_eval_dict = read_json_dump(f"{config.continous_root_dir}/eval_dict_{cont_gt.pattern}_hist.jsonl")

    write_json_dump(f"{config.continous_root_dir}/eval_dict_{cont_gt.pattern}_d{args.incr_days}_incr.jsonl", dt_eval_dict, mode="w")

    log.warning(f"Incremental Start Date: {HIST_next_date}")
    total_incr_st = time.time()
    start_date = HIST_next_date
    DG_INCR, threads, thread_similarity = cont_gt.incremental_run(start_date,DG_HIST,verbose=False)

    log.warning(f"\nTotal Incr Run Time: {fmt_time(time.time()-total_incr_st)}")
    log.warning(f"\nTotal E2E Run Time: {fmt_time(time.time()-total_hist_st)}")

