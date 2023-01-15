
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


def read_pickle(bucket, path):
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket)

    # Get the blob (pickle file)
    blob = bucket.get_blob(path)

    # Read the contents of the blob
    content = blob.download_as_string()

    # Load the pickle file
    obj = pickle.loads(content)
    return obj


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


"TO DO: save a dictionary of doc_id: title"
doc_title_dict = read_pickle_in("./postings_gcp/title_dict.pickle")

"TO DO: save a dictionary of doc_id: doc_len"
DL = read_pickle_in("./postings_gcp/DL.pickle")

"TO DO: save page view counter"
pageview = read_pickle_in("./postings_gcp/pageviews-202108-user.pkl")

term_total = read_pickle_in("./postings_gcp/term_total.pickle")

norm_dict = read_pickle_in("./postings_gcp/norm_dict.pickle")

"page rank"
pagerank = read_pickle_in('./postings_gcp/pagerank.pickle')
# df = pd.read_csv('./postings_gcp/pagerank.pickle')
# df.columns = ['id', 'page rank']
# pagerank = df.set_index('id')
# pagerank = pagerank.to_dict()['page rank']

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


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(term_total)
    Q = np.zeros(total_vocab_size)
    term_vector = list(term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = np.math.log((len(DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_posting_iter(index):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


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


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    sim_dict = {}

    for index, row in D.iterrows():
        # for column in D:
        cosine = round(np.dot(row, Q) / (np.linalg.norm(row) * np.linalg.norm(Q)))
        sim_dict[index] = cosine

    return sim_dict


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

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


# class BM25_from_index:
#     """
#     Best Match 25.
#     ----------
#     k1 : float, default 1.5
#
#     b : float, default 0.75
#
#     index: inverted index
#     """
#
#     def __init__(self, index, k1=1.5, b=0.75):
#         self.b = b
#         self.k1 = k1
#         self.index = index
#         self.N = len(DL)
#         self.AVGDL = sum(DL.values()) / self.N
#         self.words, self.pls = zip(*self.index.posting_lists_iter())
#
#     def calc_idf(self, list_of_tokens):
#         """
#         This function calculate the idf values according to the BM25 idf formula for each term in the query.
#
#         Parameters:
#         -----------
#         query: list of token representing the query. For example: ['look', 'blue', 'sky']
#
#         Returns:
#         -----------
#         idf: dictionary of idf scores. As follows:
#                                                     key: term
#                                                     value: bm25 idf score
#         """
#         idf = {}
#         for term in list_of_tokens:
#             if term in self.index.df.keys():
#                 n_ti = self.index.df[term]
#                 idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
#             else:
#                 pass
#         return idf
#
#     def search(self, queries, N=3):
#         """
#         This function calculate the bm25 score for given query and document.
#         We need to check only documents which are 'candidates' for a given query.
#         This function return a dictionary of scores as the following:
#                                                                     key: query_id
#                                                                     value: a ranked list of pairs (doc_id, score) in the length of N.
#
#         Parameters:
#         -----------
#         query: list of token representing the query. For example: ['look', 'blue', 'sky']
#         doc_id: integer, document id.
#
#         Returns:
#         -----------
#         score: float, bm25 score.
#         """
#         # YOUR CODE HERE
#
#         scores_d = {}
#         for i in queries:
#             q = list(set(queries[i]))
#             self.idf = self.calc_idf(q)
#             docs = get_candidate_documents_and_scores(q, self.index, self.words, self.pls)
#             candidates = {id: docs[(id, term)] for (id, term) in docs}
#             scores = []
#             for j in candidates:
#                 score = self._score(q, j)
#                 # if score not in scores:
#                 scores.append((j, score))
#             sort = sorted(scores, key=lambda tup: tup[1], reverse=True)
#             scores_d[i] = sort[:N]
#         return scores_d
#
#     def _score(self, query, doc_id):
#         """
#         This function calculate the bm25 score for given query and document.
#
#         Parameters:
#         -----------
#         query: list of token representing the query. For example: ['look', 'blue', 'sky']
#         doc_id: integer, document id.
#
#         Returns:
#         -----------
#         score: float, bm25 score.
#         """
#         score = 0.0
#         doc_len = DL[doc_id]
#
#         for term in query:
#             if term in self.term_total.keys():
#                 term_frequencies = dict(self.pls[self.words.index(term)])
#                 if doc_id in term_frequencies.keys():
#                     freq = term_frequencies[doc_id]
#                     numerator = self.idf[term] * freq * (self.k1 + 1)
#                     denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
#                     score += (numerator / denominator)
#         return score


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


# def calc_idf(index, list_of_tokens):
#     """
#     This function calculate the idf values according to the BM25 idf formula for each term in the query.
#
#     Parameters:
#     -----------
#     query: list of token representing the query. For example: ['look', 'blue', 'sky']
#
#     Returns:
#     -----------
#     idf: dictionary of idf scores. As follows:
#                                                 key: term
#                                                 value: bm25 idf score
#     """
#     N = len(DL)
#     idf = {}
#     for term in list_of_tokens:
#         if term in index.df.keys():
#             n_ti = index.df[term]
#             idf[term] = math.log(1 + (N - n_ti + 0.5) / (n_ti + 0.5))
#         else:
#             pass
#     return idf
#
#
# def all_scores(index, query, doc_ids, b=0.75, k1=1.2, k3=1.5):
#     """
#     This function calculate the bm25 score for given query and document.
#
#     Parameters:
#     -----------
#     query: list of token representing the query. For example: ['look', 'blue', 'sky']
#     doc_id: integer, document id.
#
#     Returns:
#     -----------
#     score: float, bm25 score.
#     """
#     N = len(DL)
#     AVGDL = sum(DL.values()) / N
#     freq_q = Counter(query)
#     len_q = len(query)
#
#     all_scores = {}
#     # score(D,Q) = Σ(i=1 to n)[IDF(qi) * ((tf(qi,D) * (k1 + 1)) / (tf(qi,D) + k1 * (1 - b + b * |D|/avgdl)) * (k3 + 1) * qtf(qi) / (k3 + qtf(qi))]
#
#     idf = calc_idf(index, query)
#
#     for doc_id in doc_ids:
#         score = 0.0
#         # flag = False
#         for term in query:
#             term_frequencies = dict(read_posting_list(index, term))
#             if doc_id in term_frequencies.keys():
#                 # flag = True
#                 doc_len = DL[doc_id]
#                 freq = term_frequencies[doc_id]
#                 qtf = freq_q[term] / len_q
#                 denominator = freq + k1 * (1 - b + b * doc_len / AVGDL)
#                 numerator = idf.get(term, 0) * (freq * (k1 + 1) / denominator) * ((k3 + 1) * qtf / (k3 + qtf))
#                 score += numerator
#         # if score != 0.0 and flag:
#         all_scores[doc_id] = score
#
#     return all_scores
#
#
# def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5):
#     """
#     This function merge and sort documents retrieved by its weighte score (e.g., title and body).
#
#     Parameters:
#     -----------
#     title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
#                                                                             key: query_id
#                                                                             value: list of pairs in the following format:(doc_id,score)
#
#     body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
#                                                                             key: query_id
#                                                                             value: list of pairs in the following format:(doc_id,score)
#     title_weight: float, for weigted average utilizing title and body scores
#     text_weight: float, for weigted average utilizing title and body scores
#     N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.
#
#     Returns:
#     -----------
#     dictionary of querires and topN pairs as follows:
#                                                         key: query_id
#                                                         value: list of pairs in the following format:(doc_id,score).
#     """
#
#     # YOUR CODE HERE
#
#     return_dict = {}
#
#     for ID in title_scores.keys():
#         return_dict[ID] = title_scores[ID] * title_weight
#
#     for ID in body_scores.keys():
#         if ID not in return_dict.keys():
#             return_dict[ID] = body_scores[ID] * text_weight
#         else:
#             return_dict[ID] += body_scores[ID] * text_weight
#
#     return return_dict
#
#
# def bm_25_search(index, query, N=100):
#     """
#     This function calculate the bm25 score for given query and document.
#     We need to check only documents which are 'candidates' for a given query.
#     This function return a dictionary of scores as the following:
#                                                                 key: query_id
#                                                                 value: a ranked list of pairs (doc_id, score) in the length of N.
#
#     Parameters:
#     -----------
#     query: list of token representing the query. For example: ['look', 'blue', 'sky']
#     doc_id: integer, document id.
#
#     Returns:
#     -----------
#     score: float, bm25 score.
#     """
#
#     pls = {}
#     for i in query:
#         pls[i] = read_posting_list(body_index, i)
#
#     lst = []
#     c_docs = get_candidate_documents_and_scores(query, index, pls)
#     candidates_1 = [j[0] for j in c_docs.keys()]
#     # print("candidates_1" + str(candidates_1))
#     candidates = set(candidates_1)
#     AllScores = all_scores(index, query, candidates, 0.75, 1.2, 1.5)
#
#     for id_doc in candidates:
#         score = AllScores[id_doc]
#         if (id_doc, score) not in lst:
#             lst.append((id_doc, score))
#         else:
#             continue
#
#     sorted_by_second = sorted(lst, key=lambda tup: tup[1], reverse=True)[:N]
#     to_return = dict(sorted_by_second)
#
#     # print(queries)
#     return to_return


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


def all_scores(index, query, doc_ids, b=0.75, k1=1.2, k3=1.5):
    """
    This function calculate the bm25 score for given query and document.

    Parameters:
    -----------
    query: list of token representing the query. For example: ['look', 'blue', 'sky']
    doc_id: integer, document id.

    Returns:
    -----------
    score: float, bm25 score.
    """
    N = len(DL)
    AVGDL = sum(DL.values()) / N
    freq_q = Counter(query)
    len_q = len(query)

    total_scores = {}

    idf = calc_idf(index, query)

    for doc_id in doc_ids:
        score = 0.0
        # flag = False
        for term in query:
            term_frequencies = dict(read_posting_list(index, term))
            if doc_id in term_frequencies.keys():
                # flag = True
                doc_len = DL[doc_id]
                freq = term_frequencies[doc_id]
                qtf = freq_q[term] / len_q
                denominator = freq + k1 * (1 - b + b * doc_len / AVGDL)
                numerator = idf.get(term, 0) * (freq * (k1 + 1) / denominator) * ((k3 + 1) * qtf / (k3 + qtf))
                score += numerator
        # if score != 0.0 and flag:
        total_scores[doc_id] = score

    return total_scores


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


def bm_25_search(index, query, N=100):
    """
    This function calculate the bm25 score for given query and document.
    We need to check only documents which are 'candidates' for a given query.
    This function return a dictionary of scores as the following:
                                                                key: query_id
                                                                value: a ranked list of pairs (doc_id, score) in the length of N.

    Parameters:
    -----------
    query: list of token representing the query. For example: ['look', 'blue', 'sky']
    doc_id: integer, document id.

    Returns:
    -----------
    score: float, bm25 score.
    """

    pls = {}
    for i in query:
        pls[i] = read_posting_list(body_index, i)

    c_docs = get_candidate_documents_and_scores(query, index, pls)
    candidates = set([j[0] for j in c_docs.keys()])
    AllScores = all_scores(index, query, candidates, 0.75, 1.2, 1.5)
    sorted_by_second = sorted(AllScores.items(), key=lambda tup: tup[1], reverse=True)[:N]
    to_return = dict(sorted_by_second)
    return to_return


def bm25(index, query, pls, b=0.75, k1=1.2):
    N = len(DL)
    AVGDL = sum(DL.values()) / N
    freq_q = Counter(query)
    len_q = len(query)
    idf = calc_idf(index,query)

    score_ret = Counter()
    for term in query:
        postings = pls[term]
        dic = {}
        for doc_id, value in postings:
            dic[doc_id] = dic.get(doc_id, 0) + value
        for doc_id, value in dic.items():
            score = 0.0
            doc_len = DL[doc_id]
            freq = value
            numerator = idf.get(term,0) * freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / AVGDL)
            score = round((numerator / denominator), 5)
            score_ret[doc_id] = round(score_ret.get(doc_id, 0) + score, 5)
    return score_ret


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
