from scipy import sparse


def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    --vocab: dict form word2id
    """
    vocab_size = len(vocab)
    id2word = dict((i, word) for (word, i) in word2id.items())

    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)

    for i, line in enumerate(corpus):
        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]
        for pair in token_ids:
            cooccurrences[pair[0], pair[1]] += 1

    return cooccurrences
    


if __name__=='__main__':
    vocab = sys.argv[1]
