import argparse

from extractfeatures import ExtractFeatures
from prediction import Prediction


def main(data):
    features = ExtractFeatures('models/embeddings', data.hypo, data.test).extract_features()
    print(Prediction('trained-model/svm.pkl').predict(features))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Paraphrase Detection for Portuguese', epilog='Usage: python detect.py '
                                                                                             '-h sentence_h.txt -t '
                                                                                             'sentence_t.txt')
    args.add_argument('-s1', '--hypo', help='Hypothesis', required=True)
    args.add_argument('-s2', '--test', help='Test', required=True)
    main(args.parse_args())