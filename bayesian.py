import pandas as pd
import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm


def load_data(url):
    return pd.read_csv(url, encoding="latin-1")
   

def feature_engineering(data: pd.DataFrame):
    f = feature_extraction.text.CountVectorizer(stop_words='english')
    X = f.fit_transform(data["v2"])
    data["v1"]=data["v1"].map({'spam':1,'ham':0})
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33, random_state=42)
    return X_train.toarray(), X_test.toarray(), y_train.to_numpy(), y_test.to_numpy()
    

class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
        return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


def accuracy_score(actual, predicted):
    correct = 0
    for i in range(len(actual)):
      if actual[i] == predicted[i]:
        correct += 1
    return correct / float(len(actual)) * 100.0


def main():
    # Define variables
    learning_rate = 0.11001
    
    data = load_data("spam.csv")
    X_train, X_test, y_train, y_test = feature_engineering(data)
    bayes = MultinomialNB(alpha=learning_rate)
    bayes.fit(X_train, y_train)
    score_train = bayes.score(X_train, y_train)
    score_test = bayes.score(X_test, y_test)
    recall_test = metrics.recall_score(y_test, bayes.predict(X_test))
    precision = metrics.precision_score(y_test, bayes.predict(X_test))
    print(f"""
          Bayesian Classifier with learning rate is {learning_rate} \n
          1. Train Accuracy: {score_train} \n
          2. Test Accuracy: {score_test} \n
          3. Test Recal: {recall_test} \n
          4. Test Precision: {precision} \n
          """)


if __name__ == "__main__":
    main()