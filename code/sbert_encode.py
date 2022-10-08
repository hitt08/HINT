import pickle
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import logging
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def encode_sbert(sbert_model,sentences,batch=10000):
    sbert_model.eval()
    res=[]
    st=0
    en=batch
    c=0

    with torch.no_grad():
        with tqdm(total=len(sentences)) as pbar:
            while st<len(sentences):
                d=sentences[st:en]
                res.extend(sbert_model.encode(d))
                st=en
                en=st+batch
                pbar.update(batch)
                if c%10==0:
                    log.warning("")
                c+=1
    res=np.stack(res)
    print(res.shape)
    return res

if __name__ == "__main__":
    # global pnct_psg_tfidf_train_matrix,outdir
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='ifile', type=str,required=True, help='input pickle data file')
    parser.add_argument('-o', action='store', dest='ofile', type=str,required=True, help='ouput data file npy')
    parser.add_argument('-m', action='store', dest='model', type=str,default="all-MiniLM-L6-v2", help='sbert model')
    parser.add_argument('-b', action='store', dest='batch', type=int,default=8, help='batch size')

    args = parser.parse_args()
    log = logging.getLogger(__name__)
    log.warning(f"Running on {device}")
    log.warning(args)

    with open(args.ifile,"rb") as f:
        sentences=pickle.load(f)

    sbert_model = SentenceTransformer(args.model)
    sbert_model.to(device)

    res=encode_sbert(sbert_model,sentences,batch=args.batch)

    np.save(f"{args.ofile}.npy",res)
