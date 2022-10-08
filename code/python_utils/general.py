def print_continous(text):
    print(" "*1000+"\r"+text,end="\r")

def get_word_dicts(vocab):
    word_to_id = {}
    for i in range(len(vocab)):
        word_to_id[vocab[i][0]] = i
    word_to_id["UNK"] = len(word_to_id)
    id_to_word = ["" for i in range(len(word_to_id))]
    for k, v in word_to_id.items():
        id_to_word[v] = k
    return word_to_id, id_to_word

def lower_list(l):
    return list(map(str.lower,l))

def vec_concat(x):
    return [w for a in x for w in a]

