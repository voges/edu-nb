"""Code to compute the entropy of arbitrary data."""


import collections
import math


def entropy(data, base=2):
    """Compute the entropy of the data."""
    if len(data) <= 1:
        return 0.0

    counts = collections.Counter()
    for datum in data:
        counts[datum] += 1

    eta = 0.0
    probs = [(float(c) / len(data)) for c in counts.values()]
    for prob in probs:
        if prob > 0.0:
            eta -= prob * math.log(prob, base)

    return eta
