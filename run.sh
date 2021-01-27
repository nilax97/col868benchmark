pip3 install -r requirements.txt

unzip P-GNN/data/ppi.zip
cp ppi/ppi-class_map.json Dataset/PPI/.
cp ppi/ppi-G.json Dataset/PPI/.
rm -r ppi

cp P-GNN/data/PROTEINS_full/PROTEINS_full_A.txt Dataset/Protein/.
cp P-GNN/data/PROTEINS_full/PROTEINS_full_node_labels.txt Dataset/Protein/.

wget https://snap.stanford.edu/data/loc-brightkite_edges.txt.gz
gunzip loc-brightkite_edges.txt.gz
mv loc-brightkite_edges.txt Dataset/Brightkite/.

echo "In case you get an error about gsed or sed - ignore"
gsed -i 's/,//g' Dataset/Protein/PROTEINS_full_A.txt
sed -i 's/,//g' Dataset/Protein/PROTEINS_full_A.txt

cd Dataset/PPI
python3 make_edge.py
rm ppi-G.json
cd ../..
cd .
echo "Datasets copied and cleaned"

cd deepwalk
pip3 install -r requirements.txt
pip3 install deepwalk
cd ..
easy_install deepwalk

echo "Deepwalk installed"

cd node2vec
pip2 install -r requirements.txt
cd ..

echo "Node2vec installed"

cd Dataset/Protein
python3 node2vec_test.py > node2vec_out.txt
python3 deepwalk_test.py > deepwalk_out.txt

echo "Protein embeddings generated"

mkdir node2vec_embeddings
mkdir deepwalk_embeddings
mv node2vec_*.embeddings node2vec_embeddings/.
mv deepwalk_*.embeddings deepwalk_embeddings/.

python3 node2vec_pair.py > node2vec_pair_out.txt
python3 deepwalk_pair.py > deepwalk_pair_out.txt

echo "Protein pairwise node classification done"

cd ../Brightkite

python3 node2vec_test.py > node2vec_out.txt
python3 deepwalk_test.py > deepwalk_out.txt

echo "Brightkite embeddings generated"

mkdir node2vec_embeddings
mkdir deepwalk_embeddings
mv node2vec_*.embeddings node2vec_embeddings/.
mv deepwalk_*.embeddings deepwalk_embeddings/.

python3 node2vec_link_pred.py > node2vec_pair_out.txt
python3 deepwalk_link_pred.py > deepwalk_pair_out.txt

echo "Brightkite link prediction done"

cd ../PPI

python3 node2vec_test.py > node2vec_out.txt
python3 deepwalk_test.py > deepwalk_out.txt

echo "PPI embeddings generated"

mkdir node2vec_embeddings
mkdir deepwalk_embeddings
mv node2vec_*.embeddings node2vec_embeddings/.
mv deepwalk_*.embeddings deepwalk_embeddings/.

python3 node2vec_link_pred.py > node2vec_pair_out.txt
python3 deepwalk_link_pred.py > deepwalk_pair_out.txt

echo "PPI link prediction done"

python3 node2vec_class.py > node2vec_class_out.txt
python3 deepwalk_class.py > deepwalk_class_out.txt

echo "PPI multi-label classification done"

cd ../..

