import numpy as np

def create_embedding_matrix(embedding_path, word_index, embedding_dim=300, output_file_name='saved_embedding_matrix.npy'):
    word_index = word_index
    num_words = len(word_index)

    embeddings_index = {}
    with open(embedding_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def create_positional_encoding(length, dim_model, wlength = 10000.):
    assert(dim_model % 2 == 0)
    position = np.broadcast_to(np.arange(length)[:, None], (length, dim_model // 2))
    unit     = np.broadcast_to(np.arange(dim_model // 2)[None, :], (length, dim_model // 2))
    rad = position / (wlength *1.) ** (unit / (dim_model // 2))

    sin = np.sin(rad)
    cos = np.cos(rad)

    con = []
    for i in range(sin.shape[0]):
        con.append(np.concatenate([sin[i], cos[i]]))

    return np.array(con)
