# Paraphrase Detection for the Portuguese Language

```
@inproceedings{anchietaEpardo2020,
    title = {Exploring the Potentiality of Semantic Features for Paraphrase Detection},
    author = {Anchi\^{e}ta, Rafael Torres and Pardo, Thiago Alexandre Salgueiro Pardo},
    booktitle = {Proceedings of the 14th International Conference on the Computational Processing of Portuguese},
    year = {2020},
}
```

# Requirements
- Python 3

- Install the requirements

    `pip install -r requirements.txt`

- Run the following commant to download the pre-trained embeddings

    `./download.sh` 

# Usage

`python3 detect.py -s1 hypo.txt -s2 test.txt`