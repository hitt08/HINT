import logging
import argparse
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import numpy as np
from scipy.sparse import save_npz as sparse_save_npz
from sbert_encode import encode_sbert
from utils.thread_utils import tokenize_split

# from python_utils.data import *
from python_utils.time import *
import torch
from sentence_transformers import SentenceTransformer
from utils.data_utils import load_config,read_dict_dump

log=logging.getLogger()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reduce_dim(matrix, dim=300, lsa_model=None, random_seed=0):
    st = time.time()
    if (lsa_model is None):
        svd = TruncatedSVD(dim, random_state=random_seed)
        normalizer = Normalizer(copy=False)
        lsa_model = make_pipeline(svd, normalizer)

        X = lsa_model.fit_transform(matrix)
        log.warning(f"LSA Time Taken: {fmt_time(time.time() - st)}")

        explained_variance = svd.explained_variance_ratio_.sum()
        log.warning("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
        return X, lsa_model
    else:
        return lsa_model.transform(matrix)

def merge_doc_tk(docs,sep="~|~"):
    res=dict([(k,dict([(sk,[]) for sk in docs[k].keys()])) for k in docs.keys()])
    for k in res.keys():
        res[k]["data"]=[sep.join(doc) for doc in docs[k]["data"]]
        res[k]["ids"]=docs[k]["ids"]
        res[k]["labels"]=docs[k]["labels"]
    return res

def tfidf_vectorize(doc_data, lsa=False, dim=None, max_df=0.95):
    st = time.time()
    docs = merge_doc_tk(doc_data)
    vect = TfidfVectorizer(tokenizer=tokenize_split, strip_accents='ascii', max_df=max_df, max_features=100000,token_pattern=None)
    features = vect.fit(docs["train"]["data"])
    train_matrix = features.transform(docs["train"]["data"])
    # test_matrix = features.transform(docs["test"]["data"])

    log.warning(f"Time Taken: {fmt_time(time.time() - st)}")

    if lsa:
        train_lsa, lsa_model = reduce_dim(train_matrix, dim=dim)
        return features, train_matrix, train_lsa, lsa_model
    else:
        return features, train_matrix


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str, default="minilm",help='embedding model: minilm; distilroberta;tfidf')
    parser.add_argument('-b', action='store', dest='batch', type=int, default=8, help='batch size')
    args = parser.parse_args()

    log.warning(args)

    config=load_config()

    if args.emb=="tfidf":
        log.warning("Vectorising 5W1H")
        with open(f"{config.graph_root_dir}/pnct_nh_5w1h_passage_data.p", "rb") as f:
            pnct_nh_data = pickle.load(f)
        _, pnct_nh_tfidf_train_matrix, pnct_nh_lsa_train_matrix, pnct_nh_lsa_model = tfidf_vectorize(pnct_nh_data, lsa=True, dim=200, max_df=0.9)

        log.warning(f"TFIDF size: {pnct_nh_tfidf_train_matrix.shape}")
        log.warning(f"LSA size: {pnct_nh_lsa_train_matrix.shape}")
        sparse_save_npz(f"{config.emb_root_dir}/nh_5w1h_tfidf_emb.npz",pnct_nh_tfidf_train_matrix)
        np.savez_compressed(f"{config.emb_root_dir}/nh_5w1h_tfidflsa_emb.npz",pnct_nh_lsa_train_matrix)

        with open(f"{config.emb_root_dir}/pnct_nh_5w1h_lsa_model.p", "wb") as f:
            pickle.dump(pnct_nh_lsa_model,f)

        log.warning("Vectorising Passages")
        with open(f"{config.graph_root_dir}/pnct_nh_passage_data.p", "rb") as f:
            pnct_nh_data = pickle.load(f)
        pnct_nh_features, pnct_nh_tfidf_train_matrix, pnct_nh_lsa_train_matrix, pnct_nh_lsa_model = tfidf_vectorize(pnct_nh_data, lsa=True, dim=200, max_df=0.9)

        log.warning(f"TFIDF size: {pnct_nh_tfidf_train_matrix.shape}")
        log.warning(f"LSA size: {pnct_nh_lsa_train_matrix.shape}")
        sparse_save_npz(f"{config.emb_root_dir}/{config.passage_tfidf_emb_file}",pnct_nh_tfidf_train_matrix)
        np.savez_compressed(f"{config.emb_root_dir}/nh_tfidflsa_emb.npz",pnct_nh_lsa_train_matrix)

        with open(f"{config.emb_root_dir}/{config.passage_tfidf_features_file}", "wb") as f:
            pickle.dump(pnct_nh_features.get_feature_names(),f)
        with open(f"{config.emb_root_dir}/pnct_nh_5w1h_lsa_model.p", "wb") as f:
            pickle.dump(pnct_nh_lsa_model,f)

        log.warning(f"Files Written at: {config.emb_root_dir}")

    else:

        with open(f"{config.graph_root_dir}/{config.doc_ids_file}", "rb") as f:
            pnct_nh_5w1h_passage_doc_ids=pickle.load(f)

        log.warning(f"Running on {device}")
        model="all-MiniLM-L6-v2" if args.emb=="minilm" else "all-distilroberta-v1"
        sbert_model = SentenceTransformer(model)
        sbert_model.to(device)

        log.warning("Vectorising 5W1H")
        collection = read_dict_dump(f"{config.root_dir}/5w1h/nh_5w1h_collection.json.gz")
        sentences=[collection[d] for d in pnct_nh_5w1h_passage_doc_ids] #Reorder
        res=encode_sbert(sbert_model,sentences,batch=args.batch)
        log.warning(f"emb size: {res.shape}")
        np.savez_compressed(f"{config.emb_root_dir}/nh_5w1h_{args.emb}_emb.npz", res)

        log.warning("Vectorising Passages")
        collection = read_dict_dump(f"{config.root_dir}/newshead/processed/nh_collection.json.gz")
        sentences = [collection[d] for d in pnct_nh_5w1h_passage_doc_ids] #Reorder
        res = encode_sbert(sbert_model, sentences, batch=args.batch)
        log.warning(f"emb size: {res.shape}")
        np.savez_compressed(f"{config.emb_root_dir}/nh_{args.emb}_emb.npz", res)