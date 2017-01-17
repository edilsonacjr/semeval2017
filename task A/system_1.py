
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

PROC_DIR = '../data/processed/'


def main():

    df_train = pd.read_csv(PROC_DIR + 'train.csv')
    df_test = pd.read_csv(PROC_DIR + 'test.csv')
    df_dev = pd.read_csv(PROC_DIR + 'dev.csv')
    df_devtest = pd.read_csv(PROC_DIR + 'devtest.csv')

    X_train = df_train['text'].values
    y_train = df_train['label'].values

    X_dev = df_dev['text'].values
    y_dev = df_dev['label'].values

    X_test = df_test['text'].values
    y_test = df_test['label'].values


    # pipelines to be used in the ensemble

    pipelines = []

    # tfidf with svm linear
    pipe1 = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
    pipelines.append(('tfidf-SVM', pipe1))

    # tfidf with MultinomialNB
    pipe2 = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pipelines.append(('tfidf-MNB', pipe2))



    # learning weights for classifiers - Stacking
    """
    eclf = VotingClassifier(estimators=pipelines, voting='hard', n_jobs=1)
    eclf.fit(X_train, y_train)
    individual_results = {}
    for estimator, (name, _) in zip(eclf.estimators_, eclf.estimators):
        individual_results[name] = estimator.predict(X_train)

    df_estimators = pd.DataFrame(individual_results)

    stacker = LogisticRegression()
    stacker.fit(df_estimators.values, y_train)

    for estimator, (name, _) in zip(eclf.estimators_, eclf.estimators):
        individual_results[name] = estimator.predict(X_test)

    df_estimators = pd.DataFrame(individual_results)

    y_pred = stacker.predict(df_estimators.values)

    print('Stacking ====')
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    """

    # ensemble, hard for majority of votes, soft uses argmax of probabilities
    eclf = VotingClassifier(estimators=pipelines, voting='hard', n_jobs=4)

    eclf.fit(X_train, y_train)

    print('Ensemble ====')
    print('Accuracy (train): %.5f' % accuracy_score(y_train, eclf.predict(X_train)))
    print('Accuracy (dev): %.5f' % accuracy_score(y_dev, eclf.predict(X_dev)))
    print('Accuracy (test): %.5f' % accuracy_score(y_test, eclf.predict(X_test)))
    print('F1 score: %.5f' % f1_score(y_test, eclf.predict(X_test), average='macro'))


    # Final evaluation

    X_new = np.concatenate((X_train, X_dev, X_test))
    y_new = np.concatenate((y_train, y_dev, y_test))

    eclf = VotingClassifier(estimators=pipelines, voting='hard', n_jobs=4)

    eclf.fit(X_new, y_new)

    df_evaldev = pd.read_csv(PROC_DIR + 'eval-dev.csv')
    df_evalfinal = pd.read_csv(PROC_DIR + 'eval-final.csv')

    X_evaldev = df_evaldev['text'].values

    X_evalfinal = df_evalfinal['text'].values


    df_saida = pd.DataFrame({'id': df_evaldev['id'].values, 'class': eclf.predict(X_evaldev)})
    df_saida.to_csv('../data/results/eval-dev.txt', header=False, index=False, sep='\t', columns=['id', 'class'])

    df_saida = pd.DataFrame({'id': df_evalfinal['id'].values, 'class': eclf.predict(X_evalfinal)})
    df_saida.to_csv('../data/results/eval-final.txt', header=False, index=False, sep='\t', columns=['id', 'class'])


if __name__ == '__main__':
    main()