import numpy as np


def discount_utility(df, disc_fac):
    v = 0.0
    for i in df.index.levels[0]:
        v += np.sum(np.multiply(disc_fac ** df.xs([i]).index, df.xs([i])["utilities"]))
    return v / len(df.index.levels[0])
