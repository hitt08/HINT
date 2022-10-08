from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import sys

from python_utils.time import *
import logging

log = logging.getLogger()


def get_coherence(threads, thread_df, passage_collecion):
    texts = []
    for t in threads:
        for d, _ in t:
            texts.append(passage_collecion[d])

    topics = thread_df["keywords"].values.tolist()

    log.warning("Preparing corpus")
    st = time.time()

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    log.warning(f"Time Take: {fmt_time(time.time() - st)}")

    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='c_v', texts=texts, processes=8)

    log.warning("Computing coherence")

    coherence = cm.get_coherence_per_topic()

    log.warning(f"Time Take: {fmt_time(time.time() - st)}")

    return coherence