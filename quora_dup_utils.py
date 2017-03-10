""" Helper functions for exploring quora duplicate questions data set.
"""


import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
# libraries for timing
from contextlib import contextmanager
from timeit import default_timer
import time

""" Functions to calculate ROC AUC score for model
"""

def calculate_AUC(model, doc_names_and_duplicate_class):
    """ Return area under ROC curve for model. This is done by simply taking
        cosine similarity between
        document vectors to predict whether they are duplicate questions or not.
    """
    doc_distances = []

    for i in range(len(doc_names_and_duplicate_class)):
        # get word vectors for given pair
        vec1_name = doc_names_and_duplicate_class[i][0]
        vec2_name = doc_names_and_duplicate_class[i][1]
        vec1 = model.docvecs[vec1_name]
        vec2 = model.docvecs[vec2_name]
        # take cosine distance between them
        distance = cosine_similarity(vec1, vec2)
        doc_distances.append(distance)

    doc_distances = np.array(doc_distances)
    doc_scores = np.array([x[2] for x in doc_names_and_duplicate_class])

    return roc_auc_score(doc_scores, doc_distances)

def cosine_similarity(vec1, vec2):
    """return cosine angle between numpy vectors v1 and v2
    """
    def unit_vector(vec):
        return vec/np.linalg.norm(vec)
    vec1_u, vec2_u = unit_vector(vec1), unit_vector(vec2)
    return np.dot(vec1_u, vec2_u)


""" helper function for recording time of computations
"""
@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


"""
functions to find best accuracy threshold given the cosine similarities
between document vectors; the function to call in the notebook is
report_accuracy_prec_recall_F1

The function get_model_distances_and_scores returns the true tag (1 or 0)
for each pair of documents along with the cosine similarity (float between
-1 and 1) for each pair of documents.

"""

def max_accuracy(y_target, y_pred, thresh_number=5000):
    # find the maximum accuracy that can be achieved with y_pred by
    # choosing appropriate threshold

    # returns (max_accuracy, max_accuracy_threshold, max_accuracy_predictions)

    min_thresh, max_thresh = min(y_pred), max(y_pred)
    thresholds = np.linspace(min_thresh, max_thresh,thresh_number)
    best_thresh, best_acc = 0, 0
    best_preds = y_pred
    for thresh in thresholds:
        # make predictions list
        y_pred_vals = np.array([0 if x<thresh else 1 for x in y_pred])
        # compute accuracy
        acc = get_accuracy(y_target, y_pred_vals)
        if acc > best_acc:
            best_thresh, best_acc = thresh, acc
            best_preds = y_pred_vals
    print "Best accuracy:", round(best_acc,4)
    return (round(best_acc,4), best_thresh, best_preds)

def get_accuracy(y_target, y_pred_vals):
    # get accuracy between vector of targets and vector of definite predictions
    assert len(y_target) == len(y_pred_vals)
    num_correct = 0
    for i in range(len(y_target)):
        if y_target[i] == y_pred_vals[i]:
            num_correct += 1
    return float(num_correct)/float(len(y_target))

def report_accuracy_prec_recall_F1(y_target, y_pred):
    (best_acc, best_thresh, best_preds) = max_accuracy(y_target, y_pred)
    (precision, recall, F1, support) = precision_recall_fscore_support(y_target,
                                                best_preds, average='binary')
    print "Precision:", precision
    print "Recall:", recall
    print "F1-score:", round(F1, 4)

def get_model_distances_and_scores(model, doc_names_and_duplicate_class):
    """ Return (y_target, y_pred) for model and given documents
    y_pred is number between -1 and 1
    """
    doc_distances = []

    for i in range(len(doc_names_and_duplicate_class)):
        # get word vectors for given pair
        vec1_name = doc_names_and_duplicate_class[i][0]
        vec2_name = doc_names_and_duplicate_class[i][1]
        vec1 = model.docvecs[vec1_name]
        vec2 = model.docvecs[vec2_name]
        # take cosine distance between them
        distance = cosine_similarity(vec1, vec2)
        doc_distances.append(distance)

    doc_distances = np.array(doc_distances)
    doc_scores = np.array([x[2] for x in doc_names_and_duplicate_class])

    return (doc_scores, doc_distances)


""" function that takes sentence (list of words) and word2vec model and
returns the average of the word2vec vectors of the words in the sentence
"""

def make_question_vectors(model, sentence):
    # return numpy document vector by averaging constituent word vectors

    # model is pretrained gensim word2vec model

    # sentence is a list of words in same style as iterator makes for
    # entering into word2vec
    word_vecs = []
    for word in sentence:
        try:
            new_word = model[word]
        except KeyError:
            continue
        # check whether array has nan before appending
        if not np.isnan(np.sum(new_word)):
            word_vecs.append(new_word)
    # if no appropriate word vectors found, return array of zeros
    if not word_vecs:
        return np.zeros(model.layer1_size)
    word_vecs = np.array(word_vecs)
    return word_vecs.mean(axis=0)
