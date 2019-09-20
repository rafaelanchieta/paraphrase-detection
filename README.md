# Paraphrase Detection for the Portuguese Language

# Dependencies
Python 3

Install the requirements

`pip install -r requirements.txt`

Run the following commant to download the pre-trained embeddings

`./download.sh` 

# Usage
Results on the test set

`python3 classification.py`

# ToDo
- [x] Executar o parser AMR nos pares de sentenças
- [x] Avaliar a similaridade usando a métrica SEMA
- [x] Analisar se as senteças mais similares são paráfrase
- [x] Incluir outras features (classificação)