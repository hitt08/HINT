import json
import gzip


def read_json(url):
    data = []
    with open(url) as fl:
        data.append(json.loads(fl.read()))
    return data


def read(file_path):
    if file_path[-3:] == ".gz":
        with gzip.open(file_path, 'rb') as f:
            lines = [line.strip() for line in f]
    else:
        with open(file_path) as f:
            lines = [line.strip() for line in f]
    return lines


def write(file_path,data,mode="w"):
    with open(file_path,mode) as f:
        if type(data)==list:
            for line in data:
                f.write(str(line))
                f.write('\n')
        else:
            f.write(str(data))
            f.write('\n')

def read_json_dump(url):
    data = []
    with open(url) as fl:
        for line in fl.readlines():
            data.append(json.loads(line))
    return data


def write_json_dump(url, data,mode="w"):
    f = open(url, mode) #Write in write/append mode
    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        f.write(d)
        f.write('\n')
    f.close()


def write_dict(url, data, sep="~|~", mode="w"):
    f = open(url, mode)
    for k, v in data.items():
        f.write(f"{k}{sep}{v}")
        f.write('\n')
    f.close()


def read_dict(url, sep="~|~"):
    res = {}
    with open(url) as fl:
        for line in fl.read().splitlines():
            k, v = line.split(sep)
            res[k.strip()] = v.strip()
        fl.close()
    return res


def get_data_split(doc_ids,doc_labels,collection):
    data,labels = [],[]
    for i in doc_ids:
        data.append(collection[i])
        labels.append(doc_labels[i])
    return {"ids": doc_ids, "data": data, "labels": labels}