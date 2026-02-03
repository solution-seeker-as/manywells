"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 15 May 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Various functions implemented using Casadi
"""

import casadi as ca


def ca_max_approx(x, y, eps: float = 1e-6):
    """
    A differential approximation of the max function: max(x, y)
        max(x, y) ≃ (1/2) * (x + y + sqrt((x-y)**2 + eps)),
    where eps is a small number.

    :param x: First argument
    :param y: Second argument
    :param eps: Small positive number
    :return: Approximation of max(x, y)
    """
    return 0.5 * (x + y + ca.sqrt((x - y) ** 2 + eps))

def ca_min_approx(x, y, eps: float = 1e-6):
    """
    A differential approximation of the min function: min(x, y)
        min(x, y) ≃ (1/2) * (x + y - sqrt((x-y)**2 + eps)),
    where eps is a small number.
    :param x: First argument
    :param y: Second argument
    :param eps: Small positive number
    :return: Approximation of min(x, y)
    """
    return 0.5 * (x + y - ca.sqrt((x - y) ** 2 + eps))


def ca_softmax(x):
    """
    Implements the softmax function
    :param x: A vector of reals
    :return: softmax(x)
    """
    exp_logits = ca.exp(x)
    return exp_logits / ca.sum1(exp_logits)


def ca_sigmoid(x, a, k):
    """
    Sigmoid function in variable x

    :param x: Input variable
    :param a: Inflection point
    :param k: Rate parameter
    :return: Sigmoid function evaluated at x
    """
    return 1 / (1 + ca.exp(-k * (x - a)))


def ca_double_sigmoid(x, l1: float, l2: float, l3: float, a: float, b: float, k: float = 20):
    """
    Double sigmoid function with two transition/inflection points at x = a and x = b.

    The levels are specified by l1, l2, l3, so that the function two transitions: from l1 to l2 at x = a,
    and from l2 to l3 at x = b.

    The steepness of the transitions are determined by the rate parameter k. NOTE: Ipopt is a bit sensitive to this
    parameter since high values lead to steep gradients. (Reducing default value from 30 to 20 seems to help Ipopt.)

    Illustration of function:

                        ---l3---
                       /
              ---l2---/
             /
    ---l1---/

    --------a---------b--------> x

    :param x: Input variable
    :param l1: First level
    :param l2: Second level
    :param l3: Third level
    :param a: First inflection point for transition l1 -> l2
    :param b: Second inflection point for transition l2 -> l3
    :param k: Rate parameter
    :return: Double sigmoid function evaluated at x
    """

    assert a < b, 'a must be less than b'

    s1 = ca_sigmoid(x, a, k)
    s2 = ca_sigmoid(x, b, k)

    return l1 + ((l2 - l1) * s1 + (l3 - l2) * s2)
