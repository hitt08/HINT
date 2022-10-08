import numpy as np
import sys
import pickle
import logging
from tqdm import tqdm
from python_utils.data import *
from python_utils.time import *
from utils.data_utils import read_dict_dump,read_nh_part,load_config

log=logging.getLogger()

def get_passage_single_split(tr_docnos,te_docnos,collection,labels):
    data_split = {}
    passages={}
    passages["train"],passages["test"]=[],[]
    for k in tqdm(collection.keys()):
        d=k.split("_")[0]
        if d in tr_docnos:
            passages["train"].append(k)
        elif d in te_docnos:
            passages["test"].append(k)

    data_split_keys = list(passages.keys())
    for key in data_split_keys:
        data_split[key] = get_data_split(passages[key],labels,collection)
    return data_split


# To Execute this, first the NewsHead data is needed to crawled

if __name__ =="__main__":
    config=load_config()

    log.warning("Reading data")

    nh_part=read_nh_part(config.root_dir)
    temp_ids = []
    for i, (s, e) in tqdm(nh_part.items()):
        part_ids = read(f"{config.root_dir}/dpp/nh_parts_ids_{i}.txt") ##Sorted by timestamps
        temp_ids.extend(part_ids)

    with open(f"{config.root_dir}/5w1h/nh_date_data.p", "rb") as f:
        nh_date_data = pickle.load(f)

    id2story = read_dict(f"{config.root_dir}/newshead/processed/id2story_filtered.txt")
    doc2storyid = read_dict(f"{config.root_dir}/newshead/processed/doc2storyid_filtered.txt")
    tk_nh_5w1h_pnct_collection = read_dict_dump(f"{config.root_dir}/5w1h/nh_pnct.json.gz")
    ids = set([k.split("_")[0] for k in tk_nh_5w1h_pnct_collection])

    nh_tk_pnct_collection = read_dict_dump(f"{config.root_dir}/newshead/processed/pnct.json.gz")
    nh_tk_pnct_collection=dict([(i,nh_tk_pnct_collection[i]) for i in tk_nh_5w1h_pnct_collection.keys()])  #Re-order

    log.warning("Processing Doc Ids")
    pnct_nh_5w1h_passage_data = get_passage_single_split(ids, [], tk_nh_5w1h_pnct_collection, doc2storyid)
    pnct_nh_passage_data = get_passage_single_split(ids, [], nh_tk_pnct_collection, doc2storyid)
    sorter = np.argsort(pnct_nh_5w1h_passage_data["train"]["ids"])
    sort_ids = sorter[np.searchsorted(pnct_nh_5w1h_passage_data["train"]["ids"], temp_ids, sorter=sorter)]  #Docs Sorted by Timestamp and DocIds

    for k, v in pnct_nh_5w1h_passage_data["train"].items():
        pnct_nh_5w1h_passage_data["train"][k] = [v[i] for i in sort_ids]
    for k, v in pnct_nh_passage_data["train"].items():
        pnct_nh_passage_data["train"][k] = [v[i] for i in sort_ids]

    ofile=f"{config.graph_root_dir}/pnct_nh_5w1h_passage_data.p"
    with open(ofile,"wb") as f:
        pickle.dump(pnct_nh_5w1h_passage_data,f)
    log.warning(f"5W1H Doc Data saved at: {ofile}")

    ofile = f"{config.graph_root_dir}/pnct_nh_passage_data.p"
    with open(ofile, "wb") as f:
        pickle.dump(pnct_nh_passage_data, f)
    log.warning(f"Doc Data saved at: {ofile}")


    assert pnct_nh_5w1h_passage_data["train"]["ids"]==pnct_nh_passage_data["train"]["ids"]

    ofile = f"{config.graph_root_dir}/{config.doc_ids_file}"
    with open(ofile, "wb") as f:
        pickle.dump(pnct_nh_5w1h_passage_data["train"]["ids"], f)

    doc2id = {}
    for i, idx in enumerate(pnct_nh_5w1h_passage_data["train"]["ids"]):
        doc2id[idx] = i
    ofile = f"{config.graph_root_dir}/{config.doc2id_file}"
    write_dict(ofile)

    dt_feat = np.array([nh_date_data[i].timestamp() for i in pnct_nh_5w1h_passage_data["train"]["ids"]])
    ofile = f"{config.graph_root_dir}/{config.date_features_file}"
    np.savez_compressed(ofile, dt_feat)
    log.warning(f"Date Features saved at: {ofile}")