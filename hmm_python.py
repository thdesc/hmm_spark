from pyspark import SparkContext
import re
import random

""""
TO DO:

- Break forward
- Documentation
 
"""


def forward(list_observations,
            list_states,
            dict_initial_distribution,
            dict_transition_matrix,
            dict_emission_matrix):
    """
    Forward
    """
    list_alpha = []
    # Initialization part
    dict_alpha = {}
    for str_state in list_states:
        # Compute initial alpha value
        str_first_observation = list_observations[0]
        pi = dict_initial_distribution[str_state]
        b = dict_emission_matrix[str_state][str_first_observation]
        # Initial alpha value
        alpha_zero = pi * b
        dict_alpha[str_state] = alpha_zero
    list_alpha.append(dict_alpha)

    # Recursive part
    for i, str_observation in enumerate(list_observations[1:], 1):
        dict_alpha = {}
        for str_state in list_states:
            # Compute alpha value
            # noinspection PyUnresolvedReferences
            list_values = [list_alpha[i - 1][str_new_state] *
                           dict_transition_matrix[str_new_state][str_state]
                           for str_new_state in list_states]
            total_sum = sum(list_values)
            b = dict_emission_matrix[str_state][str_observation]
            alpha = b * total_sum
            dict_alpha[str_state] = alpha

        list_alpha.append(dict_alpha)

    return list_alpha


def backward(list_observations,
             list_states,
             dict_transition_matrix,
             dict_emission_matrix):
    """
    Backward
    """
    list_beta = []
    # Terminal state
    dict_beta = {}
    for str_state in list_states:
        dict_beta[str_state] = 1
    list_beta.append(dict_beta)

    for i, str_observation in enumerate(reversed(list_observations[1:]), 1):
        dict_beta = {}
        for str_state in list_states:
            # Compute alpha value
            list_values = [list_beta[i - 1][str_state_k] *
                           dict_emission_matrix[str_state_k][str_observation] *
                           dict_transition_matrix[str_state][str_state_k]
                           for str_state_k in list_states]

            beta = sum(list_values)
            dict_beta[str_state] = beta
        list_beta.append(dict_beta)

    # need to reverse list_beta
    list_beta.reverse()

    return list_beta


def loop_i(list_states, list_alpha, list_beta):
    dict_i = {}
    dict_alpha_0 = list_alpha[0]
    dict_beta_0 = list_beta[0]
    # To remove and set before
    for key in list_states:
        dict_i[key] = dict_alpha_0[key] * dict_beta_0[key]
    return dict_i


def loop_o(list_states,
           list_alpha,
           list_beta,
           list_observations):
    assert len(list_alpha) == len(list_beta)
    dict_o = {key: {key_2: 0 for key_2 in list_observations}
              for key in list_states}

    for i, observation in enumerate(list_observations):
        dict_alpha_i = list_alpha[i]
        dict_beta_i = list_beta[i]

        for key_state in list_states:
            alpha_beta = dict_alpha_i[key_state] * dict_beta_i[key_state]
            dict_o[key_state][observation] += alpha_beta

    return dict_o


def loop_t(list_states,
           list_alpha,
           list_beta,
           list_observations,
           dict_transition,
           dict_emission):
    """
    Emission
    """

    assert len(list_alpha) == len(list_beta)
    dict_t = {key: {key2: 0 for key2 in list_states} for key in list_states}
    # The first observation of the list is useless for the emission computation
    for i, observation in enumerate(list_observations[1:]):
        dict_alpha_i = list_alpha[i]
        dict_beta_i_plus_1 = list_beta[i + 1]
        for key_state in list_states:
            for key_state2 in list_states:
                value = dict_alpha_i[key_state]
                value *= dict_beta_i_plus_1[key_state2]
                value *= dict_transition[key_state][key_state2]
                value *= dict_emission[key_state2][observation]
                dict_t[key_state][key_state2] += value

    return dict_t


def reduce_dict_sum(dict_1, dict_2):
    dict_summed = {key: dict_1.get(key, 0) + dict_2.get(key, 0)
                   for key
                   in set(dict_1) | set(dict_2)}
    return dict_summed


def reduce_nested_dict_sum(list_states, nested_dict_1, nested_dict_2):
    nested_dict_summed = {}
    for state in list_states:

        dict_1 = nested_dict_1[state]
        dict_2 = nested_dict_2[state]

        nested_dict_summed[state] = reduce_dict_sum(dict_1, dict_2)
    return nested_dict_summed


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


def nested_dict_normalization(list_states, nested_dict):
    nested_dict_normalized = {}
    for state in list_states:
        dictionary = nested_dict[state]
        nested_dict_normalized[state] = dict_normalization(dictionary)

    return nested_dict_normalized


def nested_random_matrix(i, j):
    nested_list_random = [[random.random() for _ in range(j)]
                          for _ in range(i)]

    nested_list_normalized = [[val / sum(list_random) for val in list_random]
                              for list_random in nested_list_random]

    return nested_list_normalized


def dict_normalization(dictionary):
    dict_normalized = {key: val / sum(dictionary.values())
                       for key, val
                       in dictionary.items()}
    return dict_normalized


if __name__ == '__main__':

    sc = SparkContext()
    #
    # list_states = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN',
    #                'NUM', 'PRT', 'PRON', 'PUNC', 'VERB', 'X']

    list_states = ['A', 'B', 'C']
    n_states = len(list_states)
    # Data loading and initialization

    # Use your own data here, sentences must be ended by a punctuation mark Q
    text_file = sc.textFile('./data/file.txt')

    rdd_data = (text_file
                .flatMap(split_into_sentences)
                .map(lambda s: re.findall(r"[\w']+|[.,!?;]", s))
                # Empty removal
                .map(lambda x: x)
                )

    rdd_data.persist()
    # rdd_data.collect()

    # Identity map for flattening
    list_token = (rdd_data.flatMap(lambda x: x)
                  .collect())
    set_token = set(list_token)
    n_token = len(set_token)
    print(f'File contains {len(list_token)} tokens.')
    print(f'File contains {len(set_token)} unique tokens.')

    # Generate random matrix for initialization
    nested_random_start = nested_random_matrix(1, n_states)
    nested_random_transition = nested_random_matrix(n_states, n_states)
    nested_random_emission = nested_random_matrix(n_states, n_token)

    # Define random transition and emission matrix from random values
    dict_start_probability = dict(zip(list_states, nested_random_start[0]))
    dict_transition_probability = {
            list_states[i]: dict(zip(list_states, nested_random_transition[i]))
            for i in range(n_states)}
    dict_emission_probability = {
            list_states[i]: dict(zip(set_token, nested_random_emission[i]))
            for i in range(n_states)}

    # Broadcast states and initial matrix.
    bc_list_states = sc.broadcast(list_states)
    bc_dict_start_hat = sc.broadcast(dict_start_probability)
    bc_dict_transition_hat = sc.broadcast(dict_transition_probability)
    bc_dict_emission_hat = sc.broadcast(dict_emission_probability)

    n_iterations = 10

    for n in range(n_iterations):
        # Compute alpha variables
        rdd_list_alpha = rdd_data.map(
                lambda x: forward(x,
                                  bc_list_states.value,
                                  bc_dict_start_hat.value,
                                  bc_dict_transition_hat.value,
                                  bc_dict_emission_hat.value))
        rdd_list_alpha.persist()
        # rdd_list_alpha.collect()

        # Compute beta variables
        rdd_list_beta = rdd_data.map(
                lambda x: backward(x,
                                   bc_list_states.value,
                                   bc_dict_transition_hat.value,
                                   bc_dict_emission_hat.value))
        rdd_list_beta.persist()
        # rdd_list_beta.collect()

        # Compute initial probability vector statistics
        rdd_list_alpha_beta = rdd_list_alpha.zip(rdd_list_beta)
        rdd_list_alpha_beta.persist()
        rdd_list_alpha_beta_data = rdd_list_alpha_beta.zip(rdd_data)
        rdd_list_alpha_beta_data.persist()

        rdd_i = (rdd_list_alpha_beta
                 .map(lambda x: loop_i(bc_list_states.value,
                                       x[0],
                                       x[1])))
        dict_reduced_i = rdd_i.reduce(reduce_dict_sum)

        rdd_o = (rdd_list_alpha_beta_data
                 .map(lambda x: loop_o(bc_list_states.value,
                                       x[0][0],
                                       x[0][1],
                                       x[1])))

        dict_reduced_o = rdd_o.reduce(
                lambda x, y: reduce_nested_dict_sum(bc_list_states.value,
                                                    x,
                                                    y))

        rdd_t = (rdd_list_alpha_beta_data
                 .map(lambda x: loop_t(bc_list_states.value,
                                       x[0][0],
                                       x[0][1],
                                       x[1],
                                       bc_dict_transition_hat.value,
                                       bc_dict_emission_hat.value)))

        dict_reduced_t = rdd_t.reduce(
                lambda x, y: reduce_nested_dict_sum(bc_list_states.value,
                                                    x,
                                                    y))

        # Normalization of the values
        dict_normalized_i = dict_normalization(dict_reduced_i)
        dict_normalized_t = nested_dict_normalization(bc_list_states.value,
                                                      dict_reduced_t)
        dict_normalized_o = nested_dict_normalization(bc_list_states.value,
                                                      dict_reduced_o)

        # Un persist model parameters
        bc_dict_start_hat.unpersist()
        bc_dict_emission_hat.unpersist()
        bc_dict_transition_hat.unpersist()

        # Update with new model parameters
        bc_dict_start_hat = sc.broadcast(dict_normalized_i)
        bc_dict_transition_hat = sc.broadcast(dict_normalized_t)
        bc_dict_emission_hat = sc.broadcast(dict_normalized_o)

"""
Vanishing problem hence need to split into sentences.


6:53 7:08

"""
