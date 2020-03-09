import sys
import os

num_walks = [10,30,100,300]
rep_size = [32,64,128,256]
walk_len = [5,15,50,100]
window = [5,10,20]

input_file = "ppi-edge-list.txt"

for n_walk in num_walks:
    r_size = 64
    w_len = 40
    win = 5
    output_file = "deepwalk_" + str(n_walk) + "_" + str(r_size) + "_" + str(w_len) + "_" + str(win) + ".embeddings"
    command = "time deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(r_size) + " --walk-length " + str(w_len) + " --window-size " + str(win) + " --undirected true --input " + input_file + " --output " + output_file + " --workers 8"
    os.system(command)
    print(output_file + " ---- DONE")

for r_size in rep_size:
    n_walk = 10
    w_len = 40
    win = 5
    output_file = "deepwalk_" + str(n_walk) + "_" + str(r_size) + "_" + str(w_len) + "_" + str(win) + ".embeddings"
    command = "time deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(r_size) + " --walk-length " + str(w_len) + " --window-size " + str(win) + " --undirected true --input " + input_file + " --output " + output_file + " --workers 8"
    os.system(command)
    print(output_file + " ---- DONE")

for w_len in walk_len:
    n_walk = 10
    r_size = 64
    win = 5
    output_file = "deepwalk_" + str(n_walk) + "_" + str(r_size) + "_" + str(w_len) + "_" + str(win) + ".embeddings"
    command = "time deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(r_size) + " --walk-length " + str(w_len) + " --window-size " + str(win) + " --undirected true --input " + input_file + " --output " + output_file + " --workers 8"
    os.system(command)
    print(output_file + " ---- DONE")

for win in window:
    n_walk = 10
    r_size = 64
    w_len = 40
    output_file = "deepwalk_" + str(n_walk) + "_" + str(r_size) + "_" + str(w_len) + "_" + str(win) + ".embeddings"
    command = "time deepwalk --format edgelist --number-walks " + str(n_walk) + " --representation-size " + str(r_size) + " --walk-length " + str(w_len) + " --window-size " + str(win) + " --undirected true --input " + input_file + " --output " + output_file + " --workers 8"
    os.system(command)
    print(output_file + " ---- DONE")
