1) preprocess
python vectorise.py -e minilm
python preprocess.py

2) parameter tuning
validation_set_samples.py -e minilm  -m npc --td -a 10 --ent -g 0.1 -w 0.7 -i 0
grid_search.py -n minilm_mnpc_td10.0_ent0.1_p1000_s1 -i 0 --quick


3) SeqINT
hac.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7

4) HINT
python graph_threads.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7 

5) Incremental HINT
python daily_run.py -e minilm --td -a 10 --ent -g 0.1 -w 0.7 -t 1 --gpu