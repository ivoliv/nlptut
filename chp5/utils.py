import torch
import numpy as np

def load_glove_from_file(glove_filepath):
    """Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarray)
    """
    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddings
        words (list): list of words in the dataset
    Returns:
        final_embeddings (numpy.ndarray): embedding matrix
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    print('embedding dimension: {:,} '.format(glove_embeddings.shape[1]))
    print('original embedding : {:,} words'.format(glove_embeddings.shape[0]))
    embedding_size = glove_embeddings.shape[1]
    final_embeddings = np.zeros((len(words), embedding_size))

    new_words = 0

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            if word == '<PAD>':
                embedding_i = torch.zeros(1, embedding_size)
            else:
                embedding_i = torch.ones(1, embedding_size)
                torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i
            new_words += 1

    print('final embedding    : {:,} words'.format(final_embeddings.shape[0]))
    print('new embeddings     : {:,} words'.format(new_words))

    return final_embeddings