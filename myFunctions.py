
import re
import string
from gensim.models.phrases import Phrases, Phraser
from gensim import matutils

import numpy as np
from numpy import ndarray, dot, array, float32 as REAL

'''
Clean String
'''
def cleanString(s : str):

    s = s.lower()

    # remove emojis
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    s = regrex_pattern.sub(r'',s)

    # remove non-alphabet characters
    myPunc = '!"#%&$()*+-./:;,<=>?@[\\]^_`{|}~'
    s = s.translate(s.maketrans(myPunc, ' '*len(myPunc)))
    s = s.translate(s.maketrans(string.digits, ' '*len(string.digits)))

    # lets actually keep numbers, but instead lets turn them all into one string.
    # s = re.sub('\d+','00000',s)

    # clean up extra white-space
    s = re.sub('\s+',' ',s)

    return s.strip()

    # note I removed '$' and '\'' from the punctuation to remove.

'''
Find Bigrams
'''
def findBigrams(corpus : list, min_count : int):

    corpusPhrased = []
    # this actually finds the bigrams
    phrases = Phrases(corpus, min_count = min_count, delimiter='_')

    for sent in corpus:
        phrased = phrases[sent]
        corpusPhrased.append(phrased)

    return corpusPhrased


def processSentences(sentences : list, minStringSize = 5, minTokenCount = 3, splitonPeriod = False, phraseMinCount = 25, bigrams=True, trigrams=True):

    # apply string cleaning
    sentences_clean = list(map(cleanString, sentences))

    # lets filter out any short strings, as they likely wont hold enough information.
    sentences_clean = list(filter(lambda x: len(x) > minStringSize, sentences_clean))


    # okay so now lets go ahead and tokenize our sentences into words.
    # were going to ignore any sentences with fewer than 3 words.

    sentences_tokenized = []

    for sentence in sentences_clean:

        # for some of the larger bodies of text we are first going to split on periods to get 'real' sentences.
        if splitonPeriod:
            sents_split = sentence.split('.')

            for sent in sents_split:
                tokens = sent.split(' ')
                # were only going to care about 3 or greater.
                if len(tokens) >= minTokenCount:

                    # i dont want any '$' completely by themselves
                    # try:
                    #     while True:
                    #         tokens.remove('$')
                    # except ValueError:
                    #     pass

                    sentences_tokenized.append(tokens)

        # dont split on period.
        else:
            tokens = sentence.split(' ')
            if len(tokens) >= minTokenCount:
                sentences_tokenized.append(tokens)


    # Time to find Bigrams and Trigrams
    if bigrams:
        sentences_tokenized = findBigrams(sentences_tokenized, phraseMinCount)

    if trigrams:
        sentences_tokenized = findBigrams(sentences_tokenized, phraseMinCount)

    return sentences_tokenized

####
####            Model stuff
####


'''
FUNCTION : meanVector(...)
INPUT :
        keyedVectors : word vectors or keyed vectors from gensim model, (model.wv)
        positive : list of words or vectors to be applied positively [default = list()]
        negative : list of words or vectors to be applied negatively [default = list()]
OUTPUT :
        averaged word vector, [type = numpy.ndarray]
DESCRIPTION :
        allows for simple averaging of positive and negative words and vectors given a gensim model's word vector library.
        NOTE: this code is pulled from gensim's word2vec repo, I just edited it to return the averaged vector.
'''

KEY_TYPES = (str, int, np.integer)

def meanVector(keyedVectors, positive=list(), negative=list()):

    # remove any words that arent in the vocabulary
    positive = list(filter(lambda x: (x in keyedVectors), positive))
    negative = list(filter(lambda x: (x in keyedVectors), negative))

    positive = [
            (item, 1.0) if isinstance(item, KEY_TYPES + (ndarray,))
            else item for item in positive
            ]
    negative = [
            (item, -1.0) if isinstance(item, KEY_TYPES + (ndarray,))
            else item for item in negative
            ]

    # compute the weighted average of all keys
    all_keys, mean = set(), []
    for key, weight in positive + negative:
        if isinstance(key, ndarray):
            mean.append(weight * key)
        else:
            mean.append(weight * keyedVectors.get_vector(key, norm=True))
            if keyedVectors.has_index_for(key):
                all_keys.add(keyedVectors.get_index(key))
        if not mean:
            raise ValueError("cannot compute similarity with no input")

    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    return mean

'''
FUNCTION : cosineDistance(...)
'''
def cosineDistance(v1, v2):
    # matutils.unitvec scales the vectors
    return dot(matutils.unitvec(v1), matutils.unitvec(v2))
