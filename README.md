# HINT
### Narvala, H., McDonald, G., Ounis, I. (2023). Effective **H**ierarchical **In**formation **T**hreading Using Network Community Detection. In Proceedings of ECIR 2023. 
https://doi.org/10.1007/978-3-031-28244-7_44


1. Preprocess
```
python preprocess.py
python vectorise.py -e minilm
```

2. Parameter Tuning
```
python validation_set_samples.py -e minilm  -m npc --td -a 10 --ent -g 0.1 -w 0.7 -i 0
python grid_search.py -n minilm_mnpc_td10.0_ent0.1_p1000_s1 -i 0 --quick
```


3. SeqINT - https://doi.org/10.1016/j.ipm.2023.103274
```
python hac.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7
```


4. HINT
```
python graph_threads.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7 
```


5. Incremental HINT
```
python daily_run.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7 -t 1 --gpu
```



## Citation

### HINT
```
@InProceedings{narvala2023_hint,
author = {Hitarth Narvala and Graham McDonald and Iadh Ounis},
title={Effective Hierarchical Information Threading Using Network Community Detection],
booktitle={European Conference on Information Retrieval},
year={2023},
pages={701--716},
doi={10.1007/978-3-031-28244-7_44}
}
```

### SeqINT
```
@article{narvala2023_seqint,
author = {Hitarth Narvala and Graham McDonald and Iadh Ounis},
title = {Identifying chronological and coherent information threads using 5W1H questions and temporal relationships},
journal = {Information Processing & Management},
year = {2023},
volume = {60},
number = {3},
pages = {103274},
doi = {10.1016/j.ipm.2023.103274},
}
```
