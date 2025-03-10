from .fd_ucb import fd_ucb
from .is_ucb import is_ucb
from .baselines import fd_naive_greedy, fd_naive_ucb
from .baselines import is_naive_greedy, is_naive_ucb


EVALUATOR = {
    "fd-ucb": fd_ucb,
    "fd-naive-ucb": fd_naive_ucb,
    "fd-naive-greedy": fd_naive_greedy,

    "is-ucb": is_ucb,
    "is-naive-ucb": is_naive_ucb,
    "is-naive-greedy": is_naive_greedy,
}

