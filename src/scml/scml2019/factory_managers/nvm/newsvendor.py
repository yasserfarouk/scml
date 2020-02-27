import sys

sys.path.append("/".join(__file__.split("/")[:-1]))
import pandas as pd
import matplotlib.pyplot as plt


def compute_min_expectation(dict_data, size) -> dict:
    """
    Compute the expectation of min(y, X) for all values of y in the support of X where X is a discrete random variable. Returns a dictionary.
    :param dict_data:
    :param size:
    :return:
    """
    # What if we received an empty dictionary? Then we assume the random variable X has no support.
    ret = {"min_0": 0}
    temp = 1
    for i in range(1, size):
        # The dictionary only stores the values where X has positive probability. All other values are assumed to be zero.
        temp -= dict_data[str(i - 1)] if str(i - 1) in dict_data else 0
        ret["min_" + str(i)] = ret["min_" + str(i - 1)] + temp
    return ret


def solve_nvp(dict_q_out, dict_q_inn, p_out, p_inn, size):
    """
    Solves the NVM with our fast algorithm.
    :param dict_q_out: a mapping from q -> E[min(q, Q_out)].
    :param dict_q_inn: a mapping from q -> E[min(q, Q_in)].
    :param p_out: the selling per-unit price.
    :param p_inn: the buying per-unit price.
    :param size: the support of the distributions, from 0, ..., size
    :return: the optimal quantity to buy. This is one value in 0, ..., size.
    """
    y_opt_temp = 0
    opt_revenue = 0
    print(f"**** SOLVING NVP: p_out = {p_out}, p_inn = {p_inn}, size = {size} ****")
    data = []
    for q in range(0, size):
        temp = p_out * dict_q_out["min_" + str(q)] - p_inn * dict_q_inn["min_" + str(q)]
        print(q, "\t", temp)
        data += [(q, temp)]
        if temp >= opt_revenue:
            y_opt_temp = q
            opt_revenue = temp
    df = pd.DataFrame(data, columns=["y", "expected_revenue"])
    plt.scatter(df["y"], df["expected_revenue"])
    plt.axhline(0, color="black")
    plt.show()
    return y_opt_temp
