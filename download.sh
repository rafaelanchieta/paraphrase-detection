#!/bin/bash
# wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip
mkdir -p "models"
wget liara.picos.ifpi.edu.br/embeddings
mv embeddings models/
wget liara.picos.ifpi.edu.br/embeddings.vectors.npy
mv embeddings.vectors.npy models/
# unzip skip_s300.zip -d /models
# rm skip_s300.zip

echo "Done!!"
