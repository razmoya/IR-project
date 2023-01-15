
from collections import Counter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import pickle
import numpy as np
from google.cloud import storage
from numpy.linalg import norm
from inverted_index_gcp import *
import math
from contextlib import closing
from inverted_index_gcp import MultiFileReader
import pandas as pd


def read_pickle_in(path):
    """
      this function reading a pickle file from disk to object in memory
      Parameters:
      -----------
      file_name: string , file name.
      Returns:
      -----------
      object
      """
    stream = open(f'{path}', 'rb')
    pick = pickle.load(stream)
    stream.close()
    # print(f'{file_name} loaded')
    return pick


"dictionary of doc_id: title"
doc_title_dict = read_pickle_in("./postings_gcp/title_dict.pickle")

"dictionary of doc_id: doc_len"
DL = read_pickle_in("./postings_gcp/DL.pickle")

"page view counter"
pageview = read_pickle_in("./postings_gcp/pageviews-202108-user.pkl")

"term total dictionary"
term_total = read_pickle_in("./postings_gcp/term_total.pickle")

"dictionary for normalization of term frequency"
norm_dict = read_pickle_in("./postings_gcp/norm_dict.pickle")

"page rank"
pagerank = read_pickle_in('./postings_gcp/pagerank.pickle')

"indexes"
body_index = read_pickle_in("./postings_gcp/body.pkl")
title_index = read_pickle_in("./postings_gcp/title.pkl")
anchor_index = read_pickle_in("./postings_gcp/anchor_text.pkl")



nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
question_words = ["who", "what", "when", "where", "why", "how"]


def is_question(query):
    if query.endswith("?"):
        return True
    else:
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        if any(word in question_words for word in tokens):
            return True
    return False


def tokenize(text):
    """
    this method creates a list of tokens from a given text + removes stop words
    :param text: text
    :return: list of tokens
    """

    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    new_tokens = []
    for token in tokens:
        if token in all_stopwords:
            continue
        else:
            new_tokens.append(token)
    return new_tokens




def get_candidate_documents_and_scores(query_to_search, index, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in pls.keys():
            list_of_doc = pls[term]
            normlized_tfidf = []
            for doc_id, freq in list_of_doc:
                if doc_id in DL.keys():
                    if DL[doc_id] == 0:
                        continue
                    else:
                        normlized_tfidf.append((doc_id, (freq / DL[doc_id]) * np.math.log(len(DL) / index.df[term], 10)))
                else:
                    continue

            #             normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * np.math.log(len(DL) / index.df[term], 10)) for doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates



def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df.get(w, 0) * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df.get(w, 0)):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list



def cosine_sim(index, query, pls):
    # calc tf-idf for each term of the query
    epsilon = .0000001
    q_tfidf = {}
    counter = Counter(query)
    for token in np.unique(query):
        #         if token in term_total.keys():  # avoid terms that do not appear in the index.
        tf = counter[token] / len(query)  # term frequency divded by the length of the query
        df = index.df.get(token, 0)
        idf = np.math.log((len(DL)) / (df + epsilon), 10)  # smoothing
        q_tfidf[token] = tf * idf

    #     dictionary : key= doc_id value= sim score
    sim = {}
    for term in query:
        posting = pls[term]
        for doc_id, weight in posting:
            if doc_id in sim.keys():
                sim[doc_id] += q_tfidf[term] * weight
            else:
                sim[doc_id] = q_tfidf[term] * weight
    for doc_id in sim.keys():
        sim[doc_id] = sim[doc_id] * (1 / len(query)) * (1 / norm_dict[doc_id])

    return sim



def calc_idf(index, list_of_tokens):
    """
    This function calculate the idf values according to the BM25 idf formula for each term in the query.

    Parameters:
    -----------
    query: list of token representing the query. For example: ['look', 'blue', 'sky']

    Returns:
    -----------
    idf: dictionary of idf scores. As follows:
                                                key: term
                                                value: bm25 idf score
    """
    N = len(DL)
    idf = {}
    for term in list_of_tokens:
        if term in index.df.keys():
            n_ti = index.df[term]
            idf[term] = math.log(1 + (N - n_ti + 0.5) / (n_ti + 0.5))
        else:
            pass
    return idf


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """

    # YOUR CODE HERE

    return_dict = {}

    for i in title_scores.keys():
        return_dict[i] = title_scores[i] * title_weight

    for i in body_scores.keys():
        if i not in return_dict.keys():
            return_dict[i] = body_scores[i] * text_weight
        else:
            return_dict[i] += body_scores[i] * text_weight

    return return_dict


def bm25(index, query, pls, b=0.75, k1=1.2, k3=1.5):
    N = len(DL)
    AVGDL = sum(DL.values()) / N
    freq_q = Counter(query)
    len_q = len(query)
    idf = calc_idf(index,query)

    scores = Counter()
    for term in query:
        postings = pls[term]
        term_frequencies = {}
        for doc_id, value in postings:
            term_frequencies[doc_id] = term_frequencies.get(doc_id, 0) + value
        for doc_id, value in term_frequencies.items():
            doc_len = DL[doc_id]
            freq = value
            qtf = freq_q[term] / len_q
            numerator = idf.get(term,0) * freq * (k1 + 1) * ((k3 + 1) * qtf/(k3 + qtf))
            denominator = freq + k1 * (1 - b + b * doc_len / AVGDL)
            score = round((numerator / denominator), 5)
            scores[doc_id] = round(scores.get(doc_id, 0) + score, 5)
    return scores


min_pagerank = min(pagerank.values())
max_pagerank = max(pagerank.values())
min_pageview = min(pageview.values())
max_pageview = max(pageview.values())

def normalized_pagerank(score):
    return (score - min_pagerank) / (max_pagerank - min_pagerank)

def normalized_pageview(score):
    return (score - min_pageview) / (max_pageview - min_pageview)


def search_bm(query):
    q_tokens = tokenize(query)
    pls = {}
    pls_title = {}
    for i in q_tokens:
        pls[i] = read_posting_list(body_index, i)
        pls_title[i] = read_posting_list(title_index, i)

    if is_question(query):
        print(query)
        print("bm25")
        body_scores = bm25(body_index, q_tokens, pls)
        lst = body_scores.most_common(100)
        new = [(j[0], 0.5*normalized_pageview(pageview.get(j[0],0)) + 0.5*normalized_pagerank(pagerank.get(j[0],0))) for j in lst]
        sort = sorted(new, key=lambda x: x[1], reverse=True)
        res = [(j[0], doc_title_dict[j[0]]) for j in sort]

    else:
        #title:
        candidates = get_candidate_documents_and_scores(q_tokens, title_index, pls_title)
        if candidates:
            print(query)
            print("title binary")
            can = Counter(elem[0] for elem in candidates.keys())
            for i in can:
                can[i] = can[i]/len(q_tokens)
            normalized = can.items()
            # max_number = max(normalized, key=lambda x: x[1])[1]
            title_lst = [(doc_id, 0.5*normalized_pageview(pageview.get(doc_id,0)) + 0.5*normalized_pagerank(pagerank.get(doc_id,0))) for doc_id, number in normalized if number > 0.7]
            body_scores = bm25(body_index, q_tokens, pls)
            body_lst = [(doc_id, 0.5*normalized_pageview(pageview.get(doc_id,0)) + 0.5*normalized_pagerank(pagerank.get(doc_id,0))) for doc_id, score in body_scores.most_common(100)]
            merge = title_lst
            for i in body_lst:
                if i not in merge:
                    merge.append(i)
            sort = sorted(merge, key=lambda x: x[1], reverse=True)
            res = [(j[0], doc_title_dict[j[0]]) for j in sort[:100]]
        else:
            print(query)
            print("bm25")
            body_scores = bm25(body_index, q_tokens, pls)
            lst = body_scores.most_common(100)
            new = [(j[0], 0.5*normalized_pageview(pageview.get(j[0],0)) + 0.5*normalized_pagerank(pagerank.get(j[0],0))) for j in lst]
            sort = sorted(new, key=lambda x: x[1], reverse=True)
            res = [(j[0], doc_title_dict[j[0]]) for j in sort]

    return res


def search_body_back(query):
    q_tokens = tokenize(query)
    pls = {}
    for i in q_tokens:
        pls[i] = read_posting_list(body_index, i)
    sim_dict = cosine_sim(body_index, q_tokens, pls)
    lst = get_top_n(sim_dict, N=100)
    res = [(j[0], doc_title_dict[j[0]]) for j in lst]
    return res


def search_title_back(query):
    q_tokens = tokenize(query)
    pls = {}
    for i in q_tokens:
        pls[i] = read_posting_list(title_index, i)
    candidates = get_candidate_documents_and_scores(q_tokens, title_index, pls)
    can = Counter(elem[0] for elem in candidates.keys())
    lst = can.most_common()
    res = []
    for j in lst:
        if j[0] in doc_title_dict.keys():
            res.append((j[0], doc_title_dict[j[0]]))
        else:
            res.append((j[0], "title not found"))
    return res


def search_anchor_back(query):
    q_tokens = tokenize(query)
    pls = {}
    print(q_tokens)
    for i in q_tokens:
        pls[i] = read_posting_list(anchor_index, i)
    candidates = get_candidate_documents_and_scores(q_tokens, anchor_index, pls)
    can = Counter(elem[0] for elem in candidates.keys())
    lst = can.most_common()
    res = [(j[0], doc_title_dict[j[0]]) for j in lst]
    return res


def binary_body_search(query):
    q_tokens = tokenize(query)
    pls = {}
    for i in q_tokens:
        pls[i] = read_posting_list(anchor_index, i)
    candidates = get_candidate_documents_and_scores(q_tokens, anchor_index, pls)
    can = Counter(elem[0] for elem in candidates.keys())
    lst = can.most_common(100)
    res = [(j[0], doc_title_dict[j[0]]) for j in lst]
    return res


def search_backend(query):
    # q_tokens = tokenize(query)
    # body = bm_25_search(body_index, q_tokens)
    # print("1")
    # # print(body.keys())
    # title = bm_25_search(title_index, q_tokens)
    # print("2")
    # # print(title.keys())
    # merge = merge_results(body, title)
    # print("3")
    # # lst = merge.keys()
    # # print(lst)
    # res = [(j, doc_title_dict[j]) for j in merge.keys()]
    body_res = search_body_back(query)
    title_res = search_title_back(query)
    body = [(j, pagerank.get(j[0], 0)) for j in body_res]
    title = [(j, pagerank.get(j[0], 0)) for j in title_res]
    merge = list(set(body + title))
    sort = sorted(merge, key=lambda x: x[1], reverse=True)
    res = [i[0] for i in sort][:100]
    return res


def get_pagerank_back(wiki_ids):
    res = []
    for i in wiki_ids:
        res.append(pagerank[i])

    return res


def get_pageview_back(wiki_ids):
    res = []
    for i in wiki_ids:
        res.append(pageview[i])

    return res
