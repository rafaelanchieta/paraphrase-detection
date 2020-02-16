import joblib


class Prediction:

    def __init__(self, model):
        self.model = joblib.load(model)

    def predict(self, features):
        y_pred = self.model.predict(features)
        result = []
        for i in y_pred:
            if i == 1:
                result.append('paráfrase')
            else:
                result.append('não paráfrase')
        return result
