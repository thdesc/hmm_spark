import numpy as np
from pyspark import SparkContext
import re

""""
TO DO:

- remove loops from loop_t function 
- improve the way of creating list index : it means create this function:
    input: sentence x, list of unique words in corpus: vocabulary
    return list of index in vocabulary for each word in x
    exemple: x = ['je', 'suis'], vocab = ['je', il', 'fait', beau', 'suis']
             return [0, 4]

"""


def split_into_sentences(text):
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = ("(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|"
                "However\s|That\s|This\s|Wherever)")
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
                  "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>",
                  text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2",
                  text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def vectorized_alpha_fn(input_sequence, t, a, b, pi):
    """
    t correspond to the time sequence. It can go from 1 to len(sequence)
    """
    if t == 1:
        return np.matmul(pi,
                         np.diag(b[:, vocabulary.index(input_sequence[t - 1])]))
    else:
        tmp = np.matmul(a,
                        np.diag(b[:, vocabulary.index(input_sequence[t - 1])]))
        return np.matmul(
            vectorized_alpha_fn(input_sequence[:t - 1], t - 1, a, b, pi), tmp)


def compute_alpha_matrix(input_sequence, a, b, pi):
    alpha = np.empty((len(input_sequence), a.shape[0]))
    for t in range(len(input_sequence)):
        alpha[t, :] = vectorized_alpha_fn(input_sequence, t + 1, a, b, pi)
    return alpha


def vectorized_beta_fn(input_sequence, t, a, b):
    """
    t correspond to the time sequence. It can go from 1 to len(sequence)
    """
    if t == len(input_sequence):
        return np.ones(a.shape[0])
    else:
        tmp = np.dot(a, np.diag(b[:, vocabulary.index(input_sequence[t])]))
        return np.dot(tmp, vectorized_beta_fn(input_sequence, t + 1, a, b))


def compute_beta_matrix(input_sequence, a, b):
    beta = np.empty((len(input_sequence), a.shape[0]))
    for t in range(len(input_sequence)):
        beta[t, :] = vectorized_beta_fn(input_sequence, t + 1, a, b)
    return beta


def sentence_index(x, voc):
    x_index = []
    for word in x:
        x_index.append(voc.index(word))
    return x_index


def loop_i(alpha_matrix, beta_matrix):
    return np.multiply(alpha_matrix[0, :], beta_matrix[0, :])


def loop_o(list_index, alpha_matrix, beta_matrix, voc):
    matrix_o = np.zeros((len(voc), alpha_matrix.shape[1]))
    matrix_o[list_index] = np.multiply(alpha_matrix, beta_matrix)
    return matrix_o.transpose()


def loop_t(list_index, a, b, alpha_matrix, beta_matrix):
    nb_state = alpha_matrix.shape[1]
    matrix_t = np.zeros((nb_state, nb_state))
    for t in range(len(list_index) - 1):
        for q in range(alpha_matrix.shape[1]):
            tmp1 = np.multiply(alpha_matrix[t, :], a[q, :])
            tmp2 = np.multiply(tmp1, b[:, list_index[t + 1]])
            matrix_t[t, :] = np.multiply(tmp2, beta_matrix[t + 1, :])
    return matrix_t


if __name__ == '__main__':
    sc = SparkContext()

    list_states = ['DET', 'ADJ']
    n_states = len(list_states)

    text_file = sc.textFile('data/file.txt')

    rdd_data = (text_file
                .flatMap(split_into_sentences)
                .map(lambda s: re.findall(r"[\w']+|[.,!?;]", s))
                # Empty removal
                .map(lambda x: x)
                )

    rdd_data.persist()
    rdd_data.collect()

    # Identity map for flattening
    list_token = rdd_data.flatMap(lambda x: x).collect()
    # one array compare with rdd_data array of sentences
    set_token = set(list_token)
    n_token = len(set_token)
    vocabulary = list(set_token)
    print(f'File contains {len(list_token)} tokens.')
    print(f'File contains {len(set_token)} unique tokens.')

    rdd_data_index = rdd_data.map(lambda x: sentence_index(x, vocabulary))

    # random initialization
    init_probability = np.random.rand(n_states)
    init_probability /= np.sum(init_probability)

    transition_matrix = np.random.rand(n_states, n_states)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    emission_matrix = np.random.rand(n_states, n_token)
    emission_matrix /= emission_matrix.sum(axis=1, keepdims=True)

    # Broadcast states and initial matrix.
    bc_init_vec = sc.broadcast(init_probability)
    bc_transition_mat = sc.broadcast(transition_matrix)
    bc_emission_mat = sc.broadcast(emission_matrix)

    n_iterations = 1

    for n in range(n_iterations):

        import time
        rdd_list_alpha = rdd_data.map(
                lambda x: compute_alpha_matrix(x, a=bc_transition_mat.value,
                                               b=bc_emission_mat.value,
                                               pi=bc_init_vec.value)
                )

        start = time.time()

        rdd_list_alpha.collect()

        print(time.time() - start)

        rdd_list_alpha.persist()

        # Compute beta variables
        rdd_list_beta = rdd_data.map(
                lambda x: compute_beta_matrix(x, a=bc_transition_mat.value,
                                              b=bc_emission_mat.value)
                )
        rdd_list_beta.persist()
        rdd_list_beta.collect()

        rdd_list_alpha_beta = rdd_list_alpha.zip(rdd_list_beta)
        rdd_list_alpha_beta.persist()
        rdd_list_alpha_beta_data = rdd_list_alpha_beta.zip(rdd_data_index)
        rdd_list_alpha_beta_data.persist()

        rdd_i = (rdd_list_alpha_beta
                 .map(lambda x: loop_i(x[0],
                                       x[1])))
        rdd_reduced_i = rdd_i.reduce(lambda x, y: x + y)
        rdd_normalized_i = rdd_reduced_i / np.sum(rdd_reduced_i)

        rdd_o = (rdd_list_alpha_beta_data
                 .map(lambda x: loop_o(x[1], x[0][0], x[0][1], vocabulary)))
        rdd_reduced_o = rdd_o.reduce(lambda x, y: x + y)
        rdd_normalized_o = rdd_reduced_o / rdd_reduced_o.sum(axis=1,
                                                             keepdims=True)

        rdd_t = (rdd_list_alpha_beta_data
                 .map(lambda x: loop_t(x[1],
                                       bc_transition_mat.value,
                                       bc_emission_mat.value,
                                       x[0][0],
                                       x[0][1],
                                       )))

        rdd_reduced_t = (rdd_t.reduce(lambda x, y: x + y))
        rdd_normalized_t = rdd_reduced_t / rdd_reduced_t.sum(axis=1,
                                                             keepdims=True)

        # Un persist model parameters
        bc_init_vec.unpersist()
        bc_emission_mat.unpersist()
        bc_transition_mat.unpersist()

        # Update with new model parameters
        bc_init_vec = sc.broadcast(rdd_normalized_i)
        bc_transition_mat = sc.broadcast(rdd_normalized_t)
        bc_emission_mat = sc.broadcast(rdd_normalized_o)

        print(bc_init_vec.value.round(3))
        print(bc_transition_mat.value.round(3))
        print(bc_emission_mat.value.round(3))
        print('--------------------------------')
