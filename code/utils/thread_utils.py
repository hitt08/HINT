from sklearn.metrics import homogeneity_score
from sklearn.metrics import  normalized_mutual_info_score as nmi
import time
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
import datetime
from collections import Counter
import logging

def delta_to_sec(x):
    return x.days+((x.seconds/60)/60)/24

def tokenize_split(string,sep="~|~"):
    return string.split(sep)

def ev_thread_score(thread,passage_doc2id,points):
    prev=None
    thread_sim=[]
    for i,_ in thread:
        idx=passage_doc2id[i]
        x=points[idx][None,:] if len(points[idx].shape)<2 else points[idx]

        if prev is None:
            prev=x
            continue
        s=1-cosine(prev,x)
        thread_sim.append(s)
        prev=x
    return thread_sim


def eval_ev_threads(threads,doc2storyid, passage_doc_ids, passage_labels, min_len=0, return_dict=True, print_res=True):
        ext_true = []
        ext_pred = []
        pred_label_dict = {}
        docs = {}
        ext_spans=[]
        for i, t in enumerate(threads):
            temp_true, temp_pred = [], []
            temp_docs, temp_label = {}, {}

            for d, _ in t:
                if d not in docs:
                    temp_true.append(int(doc2storyid[d]))
                    temp_pred.append(i)
                    temp_docs[d] = int(doc2storyid[d])
                    temp_label[d] = i

            if len(temp_true) >= min_len:
                ext_true.extend(temp_true)
                ext_pred.extend(temp_pred)
                for d in temp_docs:
                    docs[d] = temp_docs[d]
                    pred_label_dict[d] = temp_label[d]

                ext_spans.append(delta_to_sec(t[-1][1] - t[0][1]))

        all_true = list(map(int, passage_labels))
        all_pred = []
        for i, x in enumerate(passage_doc_ids):
            if x in pred_label_dict:
                all_pred.append(pred_label_dict[x])
            else:
                all_pred.append(-1)

        res_dict = {
            "sel_docs": len(docs),
            "sel_pred": len(set(ext_pred)),
            "sel_true": len(set(ext_true)),
            "sel_pred_len": np.mean(list(Counter(ext_pred).values())),
            "sel_true_len": np.mean(list(Counter(docs.values()).values())),
            "sel_h": homogeneity_score(ext_true, ext_pred),
            "sel_nmi": nmi(ext_true, ext_pred),
            "all_docs": len(all_true),
            "all_true": len(set(all_true)),
            "all_h": homogeneity_score(all_true, all_pred),
            "all_nmi": nmi(all_true, all_pred),
            "sel_span": np.mean(ext_spans)
        }

        if print_res:
            log=logging.getLogger(__name__)
            res = "Selected:"
            res += f'\nDocs: {res_dict["sel_docs"]}, Pred Stories: {res_dict["sel_pred"]}, True Stories: {res_dict["sel_true"]}'  # ev_thread_score
            res += f'\nMean Pred Len: {res_dict["sel_pred_len"]}, Mean True Len: {res_dict["sel_true_len"]}, Mean Span: {res_dict["sel_span"]}'
            res += f'\n\tH:{res_dict["sel_h"]}'
            res += f'\n\tNMI:{res_dict["sel_nmi"]}'

            res += f"\nAll:"
            res += f'\nDocs: all_docs True Stories: {res_dict["all_true"]}'
            res += f'\n\tH:{res_dict["all_h"]}'
            res += f'\n\tNMI:{res_dict["all_nmi"]}'
            log.warning(res)

        if return_dict:
            return res_dict


def eval_filter_threads(eval_dict, params, pkey, threads, thread_similarity, doc2storyid, passage_doc_ids, passage_labels,print_res=True):
    
    param=params[pkey] if pkey is not None else params

    if print_res:
        print(param)
    l_range = param["l"]
    s_range = param["s"]
    d_range = param["d"]

    new_threads, new_sim, new_span = [], [], []
    for i, s in zip(threads, thread_similarity):


        if l_range[0] <= len(i) <= l_range[1] \
                and datetime.timedelta(d_range[0]) <= (i[-1][1] - i[0][1]) <= datetime.timedelta(d_range[1]) \
                and s_range[0] <= s <= s_range[1]:

            new_threads.append(i)
            new_sim.append(s)
            # new_span.append(delta_to_sec(i[-1][1] - i[0][1]))


    res = eval_ev_threads(new_threads, doc2storyid, passage_doc_ids, passage_labels,print_res=print_res)
    if pkey is not None:
        res["name"] = pkey
    res["cos"] = np.mean(new_sim)
    # res["span"] = np.mean(new_span)

    eval_dict.append(res)

    if print_res:
        print(f"Average Cosine Similarity: {res['cos']}", end=", ")
        print(f"Average Time Span (Days): {res['sel_span']}", end=", ")
        print()
    return eval_dict



def get_graph_threads(communities, date_data, passage_matrix, passage_doc2id,verbose=True):

    threads = []
    thread_similarity = []

    if verbose:
        pbar=tqdm(total=np.count_nonzero(communities))
    for com in communities:
        res = com
        thread = []
        for x in res:
            thread.append((x, date_data[x]))
        thread = np.asarray(thread)
        srt_ids = np.lexsort([thread[:, 0], thread[:, 1]])
        thread = thread[srt_ids].tolist()  # sorted(thread, key=lambda x: x[1])
        # thread_period = thread[-1][1] - thread[0][1]

        thread_sim = np.mean(ev_thread_score(thread, passage_doc2id, passage_matrix))  if len(thread) > 1 else np.nan

        threads.append(thread)
        thread_similarity.append(thread_sim)
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()

    thread_similarity = np.asarray(thread_similarity)
    return threads, np.array(thread_similarity)

# def filter_com_threads(community, passage_doc_ids, passage_matrix,nh_date_data, thread_len=5, min_sim=0.2, max_sim=0.8, days=2):
#     c = 0
#     threads = []
#     thread_similarity = []
#     with tqdm(total=sum([np.count_nonzero(i) for i in coms])) as pbar:
#         pc = 0
#         for pt_coms in coms:
#             for com in pt_coms:
#                 # pbar.set_description(f"Found: {c}")
#                 res = com
#                 thread = []
#                 for x in res:
#                     thread.append((x, nh_date_data[x]))
#                 thread = np.asarray(thread)
#                 srt_ids=np.lexsort([thread[:, 0], thread[:, 1]])
#                 thread = thread[srt_ids].tolist()#sorted(thread, key=lambda x: x[1])
#                 thread_period = thread[-1][1] - thread[0][1]
#                 if len(thread) >= thread_len and thread_period >= datetime.timedelta(days):
#                     thread_sim = np.mean(ev_thread_score(thread, passage_doc_ids, passage_matrix))
#                     if thread_sim >= min_sim and thread_sim <= max_sim:
#                         threads.append(thread)
#                         thread_similarity.append(thread_sim)
#                         c += 1
#
#                 pc += 1
#                 if pc % 100 == 0:
#                     pbar.set_description(f"Found: {c}")
#                     pbar.update(pc)
#                     pc = 0
#         pbar.update(pc)
#         pbar.set_description(f"Found: {c}")
#         time.sleep(1)
#         pbar.refresh()
#
#     res = f"Thread Count:{len(threads)}"
#     res += f"\nAvg Length:{np.mean([len(i) for i in threads])}"
#     res += f"\nCosine Score: {np.mean(thread_similarity)}"
#     # print(res)
#
#     return res,threads, thread_similarity

# def eval_ev_threads(threads,passage_doc_ids, passage_labels,doc2storyid):
#     ext_true=[]
#     ext_pred=[]
#     pred_label_dict={}
#     docs={}
#     for i,t in enumerate(threads):
#         for d,_ in t:
#             ext_true.append(int(doc2storyid[d]))
#             ext_pred.append(i)
#             docs[d]=int(doc2storyid[d])
#             pred_label_dict[d]=i
#
#     all_true=list(map(int,passage_labels))
#     all_pred=[]
#     for i,x in enumerate(passage_doc_ids):
#         if x in pred_label_dict:
#             all_pred.append(pred_label_dict[x])
#         else:
#             all_pred.append(-1)
#
#     res="Selected:"
#     res+=f"\nDocs: {len(docs)}, Pred Stories: {len(set(ext_pred))}, True Stories: {len(set(ext_true))}"
#     res+=f"\nMean Pred Len: {np.mean(list(Counter(ext_pred).values()))}, Mean True Len: {np.mean(list(Counter(docs.values()).values()))}"
#     res+=f"\n\tH:{homogeneity_score(ext_true,ext_pred)}"
#     res+=f"\n\tNMI:{nmi(ext_true,ext_pred)}"
#
#
#     res+=f"\nAll:"
#     res+=f"\nDocs: {len(all_true)}, True Stories: {len(set(all_true))}"
#     res+=f"\n\tH:{homogeneity_score(all_true,all_pred)}"
#     res+=f"\n\tNMI:{nmi(all_true,all_pred)}"
#
#     return res