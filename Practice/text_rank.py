import numpy as np
from nltk.corpus import brown, stopwords
from nltk.cluster.util import cosine_distance

def build_similarity_matrix(sentences, stopwords):
    S = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stopwords)

    for idx in range(len(S)):
        S[idx] /= S[idx].sum()

    return S

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def pagerank(A, eps=0.00001, d = 0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


def build_transition_matrix(links, index):
    total_links = 0
    A = np.zeros((len(index), len(index)))
    for webpage in links:
        if not links[webpage]:
            A[index[webpage]] = np.ones(len(index)) / len(index)
        else:
            for dest_webpage in links[webpage]:
                total_links += 1
                A[index[webpage]][index[dest_webpage]] = 1.0 / len(links[webpage])
    return A


def build_index(links):
    website_list = links.keys()
    return {website: index for index, website in enumerate(website_list)}

links = {
    'webpage-1': set(['webpage-2', 'webpage-4', 'webpage-5', 'webpage-6', 'webpage-8', 'webpage-9', 'webpage-10']),
    'webpage-2': set(['webpage-5', 'webpage-6']),
    'webpage-3': set(['webpage-10']),
    'webpage-4': set(['webpage-9']),
    'webpage-5': set(['webpage-2', 'webpage-4']),
    'webpage-6': set([]), # dangling page
    'webpage-7': set(['webpage-1', 'webpage-3', 'webpage-4']),
    'webpage-8': set(['webpage-1']),
    'webpage-9': set(['webpage-1', 'webpage-2', 'webpage-3', 'webpage-8', 'webpage-10']),
    'webpage-10': set(['webpage-2', 'webpage-3', 'webpage-8', 'webpage-9']),
}

if __name__ == '__main__':
    website_index = build_index(links)
    # print(website_index)
    A = build_transition_matrix(links, website_index)
    # print(A)
    # results = pagerank(A)
    # print("Results: ", results)
    # print(sum(results))
    # print([item[0] for item in sorted(enumerate(results), key=lambda item: -item[1])])

    # print(sentence_similarity("This is a good sentence".split(), "This is a bad sentence".split()))
    #
    # # One out of 2 non-stop words differ => 0.5 similarity
    # print(sentence_similarity("This is a good sentence".split(), "This is a bad sentence".split(),
    #                           stopwords.words('english')))
    #
    # # 0 out of 2 non-stop words differ => 1 similarity (identical sentences)
    # print(sentence_similarity("This is a good sentence".split(), "This is a good sentence".split(),
    #                           stopwords.words('english')))
    #
    # # Completely different sentences=> 0.0
    # print(sentence_similarity("This is a good sentence".split(), "I want to go to the market".split(),
    #                           stopwords.words('english')))

    sentences = brown.sents('ca01')
    print(sentences)
    print(len(sentences))
    stop_words = stopwords.words('english')
    S = build_similarity_matrix(sentences, stop_words)
    print(S)
