import sqlite3
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF as NMF_sklearn


def load_file():
    con = sqlite3.connect("Data/database.sqlite")
    df_content = pd.read_sql_query("SELECT * from content", con)
    df_artist = pd.read_sql_query("SELECT * from artists", con)
    df_reviews =pd.read_sql_query("SELECT * from reviews", con)
    df_years=pd.read_sql_query("SELECT * from years", con)
    
    df=df_content.join(df_reviews.set_index('reviewid'),on='reviewid')
    return df


def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):
    
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    stop_set = set(stopwords.words('english'))

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary


def softmax(v, temperature=1.0):
    
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s


def hand_label_topics(H, vocabulary):
   
    hand_labels = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:20]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_five]))
        label = input('please label this topic: ')
        hand_labels.append(label)
        print()
    return hand_labels


def analyze_article(article_index, contents, web_urls, W, hand_labels):
    
    #print(web_urls[article_index])
    #print(contents[article_index])
    probs = softmax(W[article_index], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        print('--> {:.2f}% {}'.format(prob * 100, label))
    print()


import numpy as np


class NMF(object):
    '''
    A Non-Negative Matrix Factorization (NMF) model using the Alternating Least
    Squares (ALS) algorithm.

    This class represents an NMF model, which is a useful unsupervised data
    mining tool; e.g. for finding latent topics in a text corpus such as NYT
    articles.
    '''

    def __init__(self, k, max_iters=50, alpha=0.5, eps=1e-6):
        '''
        Constructs an NMF object which will mine for `k` latent topics.
        The `max_iters` parameter gives the maximum number of ALS iterations
        to perform. The `alpha` parameter is the learning rate, it should range
        in (0.0, 1.0]. `alpha` near 0.0 causes the model parameters to be
        learned slowly over many many ALS iterations, while an alpha near 1.0
        causes model parameters to be fit quickly over very few ALS iterations.
        '''
        self.k = k
        self.max_iters = max_iters
        self.alpha = alpha
        self.eps = eps

    def _fit_one(self, V):
        '''
        Do one ALS iteration. This method updates self.H and self.W
        and returns None.
        '''
        # Fit H while holding W constant:
        H_new = np.linalg.lstsq(self.W, V, rcond=None)[0].clip(min=self.eps)
        self.H = self.H * (1.0 - self.alpha) + H_new * self.alpha

        # Fit W while holding H constant:
        W_new = np.linalg.lstsq(self.H.T, V.T, rcond=None)[0].T.clip(min=self.eps)
        self.W = self.W * (1.0 - self.alpha) + W_new * self.alpha

    def fit(self, V, verbose = False):
        '''
        Do many ALS iterations to factorize `V` into matrices `W` and `H`.

        Let `V` be a matrix (`n` x `m`) where each row is an observation
        and each column is a feature. `V` will be factorized into a the matrix
        `W` (`n` x `k`) and the matrix `H` (`k` x `m`) such that `WH` approximates
        `V`.

        This method returns the tuple (W, H); `W` and `H` are each ndarrays.
        '''
        n, m = V.shape
        self.W = np.random.uniform(low=0.0, high=1.0 / self.k, size=(n, self.k))
        self.H = np.random.uniform(low=0.0, high=1.0 / self.k, size=(self.k, m))
        for i in range(self.max_iters):
            if verbose:
                print('iter', i, ': reconstruction error:', self.reconstruction_error(V))
            self._fit_one(V)
        if verbose:
            print('FINAL reconstruction error:', self.reconstruction_error(V), '\n')
        return self.W, self.H

    def reconstruction_error(self, V):
        '''
        Compute and return the reconstruction error of `V` as the
        matrix L2-norm of the residual matrix.
        See: https://en.wikipedia.org/wiki/Matrix_norm
        See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
        '''
        return np.linalg.norm(V - self.W.dot(self.H))

def main():
    '''
    
    '''
    # Load the corpus.
    df = load_file()
    contents = df.content
    web_urls = df.web_url

    # Build our text-to-vector vectorizer, then vectorize our corpus.
    vectorizer, vocabulary = build_text_vectorizer(contents,
                                 use_tfidf=True,
                                 use_stemmer=False,
                                 max_features=5000)
    X = vectorizer(contents)

    # We'd like to see consistent results, so set the seed.
    np.random.seed(12345)

    # Find latent topics using our NMF model.
    factorizer = NMF(k=7, max_iters=35, alpha=0.5)
    W, H = factorizer.fit(X, verbose=True)

  
    hand_labels = hand_label_topics(H, vocabulary)
    rand_articles = np.random.choice(list(range(len(W))), 15)
    for i in rand_articles:
        analyze_article(i, contents, web_urls, W, hand_labels)

    # Do it all again, this time using scikit-learn.
    nmf = NMF_sklearn(n_components=7, max_iter=100, random_state=12345, alpha=0.0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    print('reconstruction error:', nmf.reconstruction_err_)
    hand_labels = hand_label_topics(H, vocabulary)
    for i in rand_articles:
        analyze_article(i, contents, web_urls, W, hand_labels)

if __name__=='__main__':
    df_artist, df_content, df_reviews, df_years=load_file()
    df_content
