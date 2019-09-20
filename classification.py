import codecs

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


class Classification:

    def __init__(self, model, test):
        self.model = joblib.load(model)
        self.x = self.get_features(test)
        self.y = self.get_labels(test)

    @staticmethod
    def get_features(file):
        """
        Gets features from file
        :param file: feature file
        :return: array of features
        """
        x = []
        with codecs.open(file, 'r', 'utf-8') as f_input:
            for line in f_input.readlines():
                features = []
                value = line.split('\t')
                features.append(float(value[0].strip()))
                features.append(float(value[1].strip()))
                features.append(float(value[2].strip()))
                features.append(float(value[3].strip()))
                x.append(features)
        return x

    @staticmethod
    def get_labels(file):
        """
        Gets labels from features file
        :param file: features file
        :return: array of labels
        """
        y = []
        data = codecs.open(file, 'r')
        for line in data.readlines():
            y.append(float(line.split('\t')[4].strip()))
        return y

    def classifier(self):
        x_resampled, y_resampled = SMOTE().fit_resample(self.x, self.y)
        y_pred = self.model.predict(x_resampled)
        print(classification_report(y_resampled, y_pred, target_names=['para', 'none']))


if __name__ == '__main__':
    model = 'trained-models/mod_skip300bayes.pkl'
    test = 'features/skip300/test/features-test-all.txt'
    Classification(model, test).classifier()

    # best skip300bays ptpt
    # best glove50bayes ptbr
