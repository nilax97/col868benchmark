import sys
import os

num_walks = [10,30,100,300]
dimension = [32,64,128,256]
walk_len = [5,15,50,100]
window = [5,10,20]
iteration = [1,5,10,50,100]
p_list = [0.1,0.5,1,2,10]
q_list = [0.1,0.5,1,2,10]

input_file = "PROTEINS_full_A.txt"

for n_walk in num_walks:
    dim = 128
    w_len = 80
    win = 10
    ite = 1
    p = 1
    q = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")

for dim in dimension:
    n_walk = 10
    w_len = 80
    win = 10
    ite = 1
    p = 1
    q = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")

for w_len in walk_len:
    n_walk = 10
    dim = 128
    win = 10
    ite = 1
    p = 1
    q = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")

for win in window:
    n_walk = 10
    dim = 128
    w_len = 80
    ite = 1
    p = 1
    q = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")

for ite in iteration:
    n_walk = 10
    dim = 128
    w_len = 80
    win = 10
    p = 1
    q = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")
    
for p in p_list:
    n_walk = 10
    dim = 128
    w_len = 80
    win = 10
    ite = 1
    q = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")

for q in q_list:
    n_walk = 10
    dim = 128
    w_len = 80
    win = 10
    ite = 1
    p = 1
    output_file = "node2vec_" + str(n_walk) + "_" + str(dim) + "_" + str(w_len) + "_" + str(win)+ "_" + str(ite)+ "_" + str(p)+ "_" + str(q) + ".embeddings"
    command = "time python2 -W ignore ../../node2vec/src/main.py --input " + input_file + " --output "+ output_file + " --dimensions " + str(dim) + " --walk-length " + str(w_len) + " --num-walks " + str(n_walk) + " --window-size " + str(win) + " --iter " + str(ite) + " --workers 8 --p " + str(p) + " --q " + str(q)
    os.system(command)
    print(output_file + " ---- DONE")

