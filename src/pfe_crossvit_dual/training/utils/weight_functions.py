import torch
import sys


def get_weight_function(name):
    match name:
        case "linear":
            return linear_
        case "sqrt":
            return power_0_5
        case "power2":
            return power2
        case "ln":
            return ln_
        case "sigmoid":
            return sigmoid_
        case "threshold":
            return threshold_
        case "parabole":
            return parabole_beta
        case _:
            print(f"{name} is not a valid weight function")
            sys.exit(0)


def linear_(x):
    return x + 1e-7


def power_0_5(x):
    return (x + 1e-7) ** 0.5


def power2(x):
    return (x + 1e-7) ** 2


def ln_(x):
    return torch.log(1 + 10 * x)


def sigmoid_(x):
    return torch.sigmoid(20 * (x - 0.1))


def threshold_(x):
    return (x > 0.05).float()


def parabole_beta(x, k=3):
    """Parabole de beta avec k=3 - proche d'une gaussienne mais plus legere en calculs"""
    return (4 * x * (1 - x)) ** k
