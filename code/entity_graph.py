import networkx as nx
import pickle
from tqdm import tqdm
import sys

from python_utils.data import *
from python_utils.time import *
from Giveme5W1H.extractor.tools.timex import Timex
import logging
from utils.data_utils import load_config,write_dict_dump

log=logging.getLogger()

ent_types ={"who" :set(["LOCATION" ,"ORGANIZATION" ,"MISC" ,"PERSON"]),
           "where" :set(["LOCATION"]),
           "when" :set(["DATE"])}

def get_entities(doc_id ,entities_dict ,ent_type="who"):
    if ent_type not in entities_dict[doc_id] or entities_dict[doc_id][ent_type] is None:
        return []

    res =[]
    cur_ent =""
    cur_ent_type =""
    cur_ent_idx =-1
    for parts in entities_dict[doc_id][ent_type]["parts"]:
        for part in parts:
            if type(part )!=dict:
                continue
            tk =part["nlpToken"]
            if tk["ner"] in ent_types[ent_type]:
                if tk["ner" ]==cur_ent_type and tk["index" ]==cur_ent_idx +1:  # Same Entity
                    if ent_type=="when":
                        continue
                    cur_ent+=tk["word" ] +tk["after"]
                else:
                    if cur_ent!="":
                        if type(cur_ent)==str:
                            cur_ent =cur_ent.strip()
                        res.append(cur_ent)
                    if ent_type=="when":
                        cur_ent =Timex.from_timex_text(tk["timex"]["value"]).get_start_date()
                    else:
                        cur_ent =tk["word" ] +tk["after"]
                    cur_ent_type =tk["ner"]
                cur_ent_idx =tk["index"]

    if cur_ent:  # Last Entity
        if type(cur_ent )==str:
            cur_ent =cur_ent.strip()
        res.append(cur_ent)

    return res

def resolve_entities(ent_5w1h_dict,ques=["who", "where"]):
    config=load_config()
    log=logging.getLogger(__name__)

    log.warning(f"Resolving {'/'.join([q.title() for q in ques])} Entities")
    nh_resolved_entities = {}

    ent_keys = ent_5w1h_dict.keys()

    for k in tqdm(ent_keys):
        nh_resolved_entities[k] = {}
        for q in ques:  # ,"when"]:
            nh_resolved_entities[k][q] = get_entities(k, ent_5w1h_dict, ent_type=q)

            if q=="when": #convert to timestamp
                nh_resolved_entities[k][q]=[e.timestamp() for e in nh_resolved_entities[k][q]]



    pattern='_'.join([q for q in ques])
    ofile=f"{config.graph_root_dir}/nh_{pattern}_entities.json"
    write_dict_dump(ofile,nh_resolved_entities)
    log.warning(f"Resolved entites stored at: {ofile}")

    return nh_resolved_entities


def create_entity_graph(entities_dict,doc_ids=None,verbose=False):
    log = logging.getLogger(__name__)
    if verbose:
        log.warning("Constructing Graph")

    G = nx.Graph()
    edges = []
    node_attr = []

    ent_docs=entities_dict.keys() if doc_ids is None else doc_ids

    if verbose:
        pbar=tqdm(total=len(ent_docs))
    for k in ent_docs:
        v=entities_dict[k]
        G.add_node(k, type="passage")
        for _, ents in v.items():
            for e in ents:
                edges.append((k, e, 1))
                node_attr.append((e, "entity"))

        if verbose:
            pbar.update()

    if verbose:
        pbar.close()

    G.add_weighted_edges_from(edges)
    nx.set_node_attributes(G, dict(node_attr), name="type")

    if verbose:
        log.warning(f"Entity Graph: Number of Nodes= {G.number_of_nodes()}. Number of Edges= {G.number_of_edges()}")

    return G

if __name__ =="__main__":
    config = load_config()

    log.warning("Reading NH Entities")
    with open(f"{config.root_dir}/5w1h/nh_entities.p","rb") as f:
        nh_entities=pickle.load(f)

    entities_dict=resolve_entities(nh_entities, ques=["who", "where"])
    G = create_entity_graph(entities_dict)
    log.warning(f"Number of Nodes: {G.number_of_nodes()}. Number of Edges: {G.number_of_edges()}")
    nx.write_gpickle(G, f"{config.graph_root_dir}/entity_graph_no_date.gp")

    entities_dict = resolve_entities(nh_entities, ques=["who", "where","when"])
    G = create_entity_graph(entities_dict)
    log.warning(f"Number of Nodes: {G.number_of_nodes()}. Number of Edges: {G.number_of_edges()}")
    nx.write_gpickle(G, f"{config.graph_root_dir}/entity_graph.gp")
