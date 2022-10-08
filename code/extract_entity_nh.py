import pickle
from tqdm import tqdm
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.candidate import Candidate
import logging
import sys

from python_utils.data import *
from multiprocessing import Pool,Manager,cpu_count
from multiprocessing.managers import BaseManager



ques=["who","when","where"]

def get_nh_5w1h(d_id):
    with gzip.open(f"newshead_extract/output/{d_id}.json.gz","rb") as f:
        data=json.loads(f.read().decode())
    text=data.setdefault('text', '')
    if text is None or text=="":
        text=data.setdefault('maintext', '')
    title=data.setdefault('title', '')
    desc=data.setdefault('description', '')

    if title is None:
        title=''
    if desc is None:
        desc=''
    if text is None:
        text=''

    document = Document(title, desc, text, raw_data=data)

    for question in data['fiveWoneH']:
        # check if there is a annotatedLiteral
        candidates=[]
        if 'extracted' in data['fiveWoneH'][question]:
            extracted = data['fiveWoneH'][question]['extracted']
            # check if the literal holds data
            if extracted is not None:
                for extract in extracted:
                    c=Candidate()
                    c.set_parts(extract["parts"])
                    candidates.append(c)

        document.set_answer(question,candidates)
    return document

def nh_extract_entities(k):
    doc=get_nh_5w1h(k)

    res_entity_parts={}
    for q in ques:
        try:
            candidate = doc.get_top_answer(q)
            valid_entity=False
            for part_list in candidate.get_parts():
                for part in part_list:
                    if type(part)!=dict:
                        continue
                    if part["nlpToken"]["ner"]!="O":
                        valid_entity=True
                        break

            if valid_entity:
                res_entity_parts[q]=candidate.get_json()
            else:
                res_entity_parts[q]=None
        except IndexError:
            pass

    nh_entities[k]=res_entity_parts
    pbar.update()

if __name__ == "__main__":
    log = logging.getLogger(__name__)

    doc2storyid=read_dict(f"data/newshead/processed/doc2storyid_filtered.txt" )

    pool_iter=list(doc2storyid.keys())

    BaseManager.register("pbar", tqdm)
    bmanager = BaseManager()
    bmanager.start()
    pbar = bmanager.pbar(total=len(pool_iter))

    manager=Manager()
    nh_entities=manager.dict()
    nh_data_answer=manager.dict([(q,manager.dict()) for q in ques])

    batch_size=20000
    st=0
    en=batch_size
    b=0

    PROCESSES=16 if 16 < cpu_count() else cpu_count()
    log.warning(f"Running {PROCESSES}-way parallel")
    while st < len(pool_iter):
        with Pool(processes=PROCESSES) as pool:
            pool.map(func=nh_extract_entities, iterable=pool_iter[st:en])
        st = en
        en = st + batch_size
        log.warning(f"Batch:{b}. Dumping Data")
        with open("data/nh_entities.p","wb") as f:
            pickle.dump(dict(nh_entities),f)

        b+=1

    pbar.close()




