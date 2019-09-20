import codecs

import joblib
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


class Training:

    def __init__(self, algorithm, train, dev):
        """
        Constructor
        :param algorithm:
        :param train: train file
        :param dev: dev file
        """
        self.algorithm = algorithm
        self.x_train = self.get_features(train)
        self.y_train = self.get_label(train)
        self.x_test = self.get_features(dev)
        self.y_test = self.get_label(dev)

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
    def get_label(file):
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

    def trainer(self, model):
        """
        Runs several classifiers (svm, naive bayes, decision tree, and nn)
        :return: confusion matrix and cross validation k=10
        """
        # x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_percentage)

        if self.algorithm == 'svm':
            clf = svm.SVC(kernel='linear', gamma='auto', verbose=True)
            clf.fit(self.x_train, self.y_train)
            print('Training score: ', clf.score(self.x_train, self.y_train))
            print('Dev score: ', clf.score(self.x_test, self.y_test))
            joblib.dump(clf, 'trained-models/mod_'+model+'svm.pkl')
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'naive_bayes':
            x_train_resampled, y_train_resampled = SMOTE().fit_resample(self.x_train, self.y_train)
            x_dev_resampled, y_dev_resampled = SMOTE().fit_resample(self.x_test, self.y_test)
            clf = GaussianNB()
            clf.fit(x_train_resampled, y_train_resampled)
            print('Training score: ', clf.score(x_train_resampled, y_train_resampled))
            print('Dev score: ', clf.score(x_dev_resampled, y_dev_resampled))
            joblib.dump(clf, 'trained-models/mod_'+model+'bayes.pkl')
            y_pred = clf.predict(x_dev_resampled)
        elif self.algorithm == 'tree':
            clf = DecisionTreeClassifier()
            clf.fit(self.x_train, self.y_train)
            print('Training score: ', clf.score(self.x_train, self.y_train))
            print('Dev score: ', clf.score(self.x_test, self.y_test))
            joblib.dump(clf, 'trained-models/mod_'+model+'tree.pkl')
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'nn':
            clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,10), random_state=1, verbose=10)
            clf.fit(self.x_train, self.y_train)
            print('Training score: ', clf.score(self.x_train, self.y_train))
            print('Dev score: ', clf.score(self.x_test, self.y_test))
            joblib.dump(clf, 'trained-models/mod_'+model+'nn.pkl')
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'lre':
            clf = LogisticRegression(random_state=0, solver='lbfgs', verbose=10)
            clf.fit(self.x_train, self.y_train)
            print('Training score: ', clf.score(self.x_train, self.y_train))
            print('Dev score: ', clf.score(self.x_test, self.y_test))
            joblib.dump(clf, 'trained-models/mod_'+model+'lre.pkl')
            y_pred = clf.predict(self.x_test)
        # report = self.eval_cross_validation(clf)
        report = self.evaluation(y_pred, y_dev_resampled)
        return report

    def eval_cross_validation(self, clf):
        """
        Runs cross validation k = 10
        :param clf: classifier
        :return: accuracy of classifier
        """
        pass
        # scores = cross_val_score(clf, self.x_train, self.y_train, cv=10)
        # return 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)
        # print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

    @staticmethod
    def evaluation(y_pred, y_test):
        """
        Results of classifiers
        :param y_pred: predicted labels
        :param y_test: test set
        :return: confusion matrix
        """
        # print(confusion_matrix(y_test, y_pred).ravel())
        return classification_report(y_test, y_pred, target_names=['para', 'none'])


if __name__ == '__main__':
    # best glove100
    # algorithms = ['svm', 'naive_bayes', 'tree', 'nn', 'lre']
    algorithms = ['naive_bayes']
    # models = ['fasttext50', 'fasttext100', 'glove50', 'glove100', 'glove300', 'skip50', 'skip100', 'skip300']
    models = ['skip300']
    # files = ['features-train.txt', 'sema-train.txt']
    for model in models:
        for algorithm in algorithms:
            t = Training(algorithm, 'features/'+model+'/train/features-train-all.txt', 'features/'+model+'/dev/features-dev-all.txt')
            report = t.trainer(model)
            with codecs.open('trained-models/mod_'+model+'alg_'+algorithm, 'w', 'utf-8') as out:
                out.write(report)