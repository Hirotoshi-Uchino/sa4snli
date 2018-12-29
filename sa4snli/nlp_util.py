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
    # save word_index
    # np.save(output_file_name, embedding_matrix)
