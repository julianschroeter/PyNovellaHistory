import itertools


def first_n_dict_entries(n, dictionary):
    "Returns a new dictionary with the first n entries"
    return dict(itertools.islice(dictionary.items(), n))

def from_n_dict_entries(n, dictionary):
    "Returns a new dictionary from the n-th entry on"
    return dict(itertools.islice(dictionary.items(), n, len(dictionary)))

