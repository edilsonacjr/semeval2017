
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

from gensim.models import Word2Vec

PROC_DIR = '../data/processed/'


class W2VTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def transform(self, X, y=None,**fit_params):
        out_m = []

        for text in X:
            parag_M = []
            for token in text.split():
                if token in self.model:
                    parag_M.append(self.model[token])
            if parag_M:
                out_m.append(np.average(parag_M, axis=0))
            else:
                out_m.append(np.random.rand(1, 300)[0])
        return np.array(out_m)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


def main():


    model = Word2Vec.load_word2vec_format('~/GoogleNews-vectors-negative300.bin', binary=True)

    # Lexicon load
    lex_file = open('../opinion-lexicon/negative-words.txt', 'r')
    neg_lexicon = lex_file.read().splitlines()
    lex_file.close()

    lex_file = open('../opinion-lexicon/positive-words.txt', 'r')
    pos_lexicon = lex_file.read().splitlines()
    lex_file.close()


    df_train = pd.read_csv(PROC_DIR + 'train.csv')
    df_test = pd.read_csv(PROC_DIR + 'test.csv')
    df_dev = pd.read_csv(PROC_DIR + 'dev.csv')
    df_devtest = pd.read_csv(PROC_DIR + 'devtest.csv')

    le = LabelEncoder()
    le.fit(df_train['label'].values)

    X_train = df_train['text'].values
    y_train = le.transform(df_train['label'].values)

    X_dev = df_dev['text'].values
    y_dev = le.transform(df_dev['label'].values)

    X_test = df_test['text'].values
    y_test = le.transform(df_test['label'].values)


    # pipelines to be used in the ensemble

    pipelines = []

    pipe1 = make_pipeline(TfidfVectorizer(sublinear_tf=True), SVC(kernel='linear', probability=True))
    pipelines.append(('tfidf-SVM', pipe1))

    pipe3 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
                          SVC(kernel='linear', probability=True))
    pipelines.append(('char-SVM', pipe3))

    pipe5 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon+neg_lexicon)),
                          SVC(kernel='linear', probability=True))
    pipelines.append(('lexicon-SVM', pipe5))

    pipe6 = make_pipeline(W2VTransformer(model=model), KNeighborsClassifier(1))
    pipelines.append(('w2v-KNN', pipe6))

    """
    # tfidf
    pipe1 = make_pipeline(TfidfVectorizer(sublinear_tf=True), SVC(kernel='linear', probability=True))
    pipelines.append(('tfidf-SVM', pipe1))

    pipe12 = make_pipeline(TfidfVectorizer(sublinear_tf=True), KNeighborsClassifier(3))
    pipelines.append(('tfidf-KNN', pipe12))

    pipe13 = make_pipeline(TfidfVectorizer(sublinear_tf=True), DecisionTreeClassifier())
    pipelines.append(('tfidf-CART', pipe13))

    pipe14 = make_pipeline(TfidfVectorizer(sublinear_tf=True), RandomForestClassifier())
    pipelines.append(('tfidf-RF', pipe14))

    #pipe15 = make_pipeline(TfidfVectorizer(sublinear_tf=True), AdaBoostClassifier())
    #pipelines.append(('tfidf-ADA', pipe15))

    #pipe16 = make_pipeline(TfidfVectorizer(sublinear_tf=True), MultinomialNB())
    #pipelines.append(('tfidf-MNB', pipe16))


    # ngram
    #pipe2 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4)),
    #                      SVC(kernel='linear', probability=True))
    #pipelines.append(('ngram-SVM', pipe2))

    pipe22 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4)),
                           KNeighborsClassifier(3))
    pipelines.append(('ngram-KNN', pipe22))

    #pipe23 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4)),
    #                       DecisionTreeClassifier())
    #pipelines.append(('ngram-CART', pipe23))

    #pipe24 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4)),
    #                       RandomForestClassifier())
    #pipelines.append(('ngram-RF', pipe24))

    #pipe25 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4)),
    #                       AdaBoostClassifier())
    #pipelines.append(('ngram-ADA', pipe25))

    #pipe26 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 4)),
    #                       MultinomialNB())
    #pipelines.append(('ngram-MNB', pipe26))


    # tfidf char-ngram
    pipe3 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
                          SVC(kernel='linear', probability=True))
    pipelines.append(('char-SVM', pipe3))

    pipe32 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
                           KNeighborsClassifier(3))
    pipelines.append(('char-KNN', pipe32))

    pipe33 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
                           DecisionTreeClassifier())
    pipelines.append(('char-CART', pipe33))

    pipe34 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
                           RandomForestClassifier())
    pipelines.append(('char-RF', pipe34))

    pipe35 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
                           AdaBoostClassifier())
    pipelines.append(('char-ADA', pipe35))

    #pipe36 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(2, 3), analyzer='char'),
    #                       MultinomialNB())
    #pipelines.append(('char-MNB', pipe36))

    # count using lexicon negative
    pipe4 = make_pipeline(CountVectorizer(vocabulary=set(neg_lexicon)),
                          SVC(kernel='linear', probability=True))
    pipelines.append(('lexicon-SVM', pipe4))

    #pipe42 = make_pipeline(CountVectorizer(vocabulary=set(neg_lexicon)),
    #                       KNeighborsClassifier(3))
    #pipelines.append(('lexicon-KNN', pipe42))

    pipe43 = make_pipeline(CountVectorizer(vocabulary=set(neg_lexicon)),
                           DecisionTreeClassifier())
    pipelines.append(('lexicon-CART', pipe43))

    pipe44 = make_pipeline(CountVectorizer(vocabulary=set(neg_lexicon)),
                           RandomForestClassifier())
    pipelines.append(('lexicon-RF', pipe44))

    pipe45 = make_pipeline(CountVectorizer(vocabulary=set(neg_lexicon)),
                           AdaBoostClassifier())
    pipelines.append(('lexicon-ADA', pipe45))

    pipe46 = make_pipeline(CountVectorizer(vocabulary=set(neg_lexicon)),
                           MultinomialNB())
    pipelines.append(('lexicon-MNB', pipe46))

    # count using lexicon positive
    pipe5 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon)),
                          SVC(kernel='linear', probability=True))
    pipelines.append(('lexiconp-SVM', pipe5))

    #pipe52 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon)),
    #                       KNeighborsClassifier(3))
    #pipelines.append(('lexiconp-KNN', pipe52))

    pipe53 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon)),
                           DecisionTreeClassifier())
    pipelines.append(('lexiconp-CART', pipe53))

    pipe54 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon)),
                           RandomForestClassifier())
    pipelines.append(('lexiconp-RF', pipe54))

    pipe55 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon)),
                           AdaBoostClassifier())
    pipelines.append(('lexiconp-ADA', pipe55))

    pipe56 = make_pipeline(CountVectorizer(vocabulary=set(pos_lexicon)),
                           MultinomialNB())
    pipelines.append(('lexiconp-MNB', pipe56))
    """

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
    print('F1 score: %.5f' % f1_score(y_test, y_pred, average='macro'))
    """

    # ensemble, hard for majority of votes, soft uses argmax of probabilities
    eclf = VotingClassifier(estimators=pipelines, voting='soft', n_jobs=6)

    eclf.fit(X_train, y_train)

    print('Ensemble ====')
    print('Accuracy (train): %.5f' % accuracy_score(y_train, eclf.predict(X_train)))
    print('Accuracy (dev): %.5f' % accuracy_score(y_dev, eclf.predict(X_dev)))
    print('Accuracy (test): %.5f' % accuracy_score(y_test, eclf.predict(X_test)))
    print('F1 score: %.5f' % f1_score(y_test, eclf.predict(X_test), average='macro'))

    # Individual Evaluation
    """
    print()
    print('Individual Evaluation')

    for estimator, (name, _) in zip(eclf.estimators_, eclf.estimators):
        print('Accuracy (%s): %.5f' % (name, accuracy_score(y_test, estimator.predict(X_test))))
    """


    # Final evaluation
    #"""

    X_new = np.concatenate((X_train, X_dev))#, X_test))
    y_new = np.concatenate((y_train, y_dev))#, y_test))

    #eclf = VotingClassifier(estimators=pipelines, voting='hard', n_jobs=4)

    #eclf.fit(X_new, y_new)

    df_evaldev = pd.read_csv(PROC_DIR + 'eval-dev.csv')
    df_evalfinal = pd.read_csv(PROC_DIR + 'eval-final.csv')

    X_evaldev = df_evaldev['text'].values

    X_evalfinal = df_evalfinal['text'].values


    df_saida = pd.DataFrame({'id': df_evaldev['id'].values, 'class': le.inverse_transform(eclf.predict(X_evaldev))})
    df_saida.to_csv('../data/results/eval-dev.txt', header=False, index=False, sep='\t', columns=['id', 'class'])

    df_saida = pd.DataFrame({'id': df_evalfinal['id'].values, 'class': le.inverse_transform(eclf.predict(X_evalfinal))})
    df_saida.to_csv('../data/results/eval-final.txt', header=False, index=False, sep='\t', columns=['id', 'class'])
    #"""

if __name__ == '__main__':
    main()