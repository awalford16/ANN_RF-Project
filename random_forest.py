from sklearn import ensemble
import sklearn.metrics as metrics

class RandomForest:
    def __init__(self, trees, min_leafs):
        self.classifier = ensemble.RandomForestClassifier(n_estimators=trees, min_samples_leaf=min_leafs)

    def train_forest(self, x, y):
        self.forest = self.classifier.fit(x, y)

    def forest_predict(self, test_x):
        return self.forest.predict(test_x)

    def get_forest_error(self, train_x, train_y, test_x, test_y):
        train_accuracy = metrics.accuracy_score(train_y, self.forest_predict(train_x))
        test_accuracy = metrics.accuracy_score(test_y, self.forest_predict(test_x))

        return train_accuracy, test_accuracy

