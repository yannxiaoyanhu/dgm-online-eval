import numpy as np
from termcolor import colored
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from algorithms.aux import EVALUATOR


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--total_nsamples', type=int, default=5000,
                    help='Total number of generated samples')
parser.add_argument('--eval_epochs', type=int, default=20,
                    help='Epochs of evaluation')
parser.add_argument('-bs', '--batch_size', type=int, default=5,
                    help='Batch size at each step')
parser.add_argument('--evaluator_name', type=str, default='is-ucb', choices=EVALUATOR.keys(),
                    help='Evaluation algorithm')
parser.add_argument('--num_model', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed')


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)

    G = args.num_model
    num_cls = 1000
    T = args.total_nsamples // args.batch_size
    bs = args.batch_size
    num_epoch = args.eval_epochs

    mu_G = np.random.randn(G, num_cls, )
    cov_G = np.empty((G, num_cls, num_cls,))
    for g in range(G):
        cov_G[g] = np.atleast_2d(np.cov(np.random.randn(10, num_cls,), rowvar=False)) + np.identity(num_cls)

    print(colored(f'[IS-online-eval Configs] evaluator: {args.evaluator_name}, bs: {args.batch_size}, T: {T}',
                  'blue'), '\n')

    # Track statistics
    visitation = np.zeros((T, G,))

    for epoch in range(1, num_epoch + 1):

        evaluator = EVALUATOR[args.evaluator_name](G=G, bs=bs, total_nsamples=args.total_nsamples)

        for t in range(T):
            gt = evaluator.select_arm()
            ft = np.random.multivariate_normal(mean=mu_G[gt], cov=cov_G[gt], size=bs)
            exp_ft = np.exp(ft - np.max(ft, axis=1, keepdims=True))
            ft = exp_ft / np.sum(exp_ft, axis=1, keepdims=True)
            evaluator.update_stats(g=gt, batch_preds=ft)

            visitation[t:, gt] += 1.

            if (t + 1) % G == 0:
                print(colored(f'Inception Score, evaluator: {args.evaluator_name}, '
                              f'epoch {epoch}, step {t + 1}, tilde.IS: {evaluator.tilde_is}', 'red'))
                print(colored(f'avg.visitation: {visitation[t] / ((t + 1) * epoch)}', 'blue'), '\n')


if __name__ == '__main__':
    main()
