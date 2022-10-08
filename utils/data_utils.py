import json
import gzip
import sys
sys.path.insert(1, f'/nfs/jup/sensitivity_classifier/')
from python_utils.data import read_dict
from types import SimpleNamespace as Namespace

def load_config(url="config.json"):
    with open(url,"r") as f:
        res = json.load(f, object_hook=lambda d: Namespace(**d))
    return res


def write_dict_dump(url, data,mode="wb",compress=True):
    if compress:
        f = gzip.open(url+".gz", mode) #Write in write/append mode
    else:
        f = open(url, mode) #Write in write/append mode
    out_data = json.dumps(data)
    if compress:
        out_data=out_data.encode()
    f.write(out_data)
    f.close()

def read_dict_dump(url,compress=True):
    if compress:
        f = gzip.open(url, "rb")
    else:
        f = open(url, "r")
    out_data = json.loads(f.read())
    f.close()
    return out_data


def read_json_dump(url, compress=True):
    data = []
    if compress:
        f = gzip.open(url, "rt")
    else:
        f = open(url, "r")
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data


def write_json_dump(url, data, mode="wt", compress=True):
    if compress:
        if not url.endswith(".gz"):
            url = url + ".gz"
        f = gzip.open(url, mode)  # Write in write/append mode
    else:
        f = open(url, mode)  # Write in write/append mode

    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        # if compress:
        #     d = d.encode()
        f.write(d)
        f.write('\n')
    f.close()

def read_nh_part(root_dir):
    nh_part = read_dict(f"{root_dir}/dpp/nh_parts.txt")
    for k, v in nh_part.items():
        nh_part.pop(k)
        nh_part[int(k)] = tuple(map(int, v.strip("()").split(",")))

    return nh_part