#!/bin/bash
wget http://143.107.183.175:22980/download.php?file=embeddings/word2vec/skip_s300.zip
mkdir -p "models"
unzip skip_s300.zip -d /models
rm skip_s300.zip

echo "Done!!"
