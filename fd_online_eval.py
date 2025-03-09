import numpy as np
from termcolor import colored
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from algorithms.aux import EVALUATOR
from metrics.fd import compute_FD_with_stats


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--total_nsamples', type=int, default=5000,
                    help='Total number of generated samples')
parser.add_argument('--burn_in_nsamples', type=int, default=0)
parser.add_argument('--eval_epochs', type=int, default=20,
                    help='Epochs of evaluation')
parser.add_argument('-bs', '--batch_size', type=int, default=5,
                    help='Batch size at each step')
parser.add_argument('--num_dim', type=int, default=128)
parser.add_argument('--num_model', type=int, default=3)

parser.add_argument('--evaluator_name', type=str, default='fd-ucb', choices=EVALUATOR.keys(),
                    help='Evaluation algorithm')
parser.add_argument('--fd_ucb_M', type=float, default=0.,
                    help='Parameter M for FD-UCB')
parser.add_argument('--fd_ucb_kappa', type=float, default=1.,
                    help='Parameter kappa for FD-UCB')
parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed')


def main():
    # Parameters
    args = parser.parse_args()
    np.random.seed(args.seed)

    G = args.num_model
    num_dim = args.num_dim
    num_epoch = args.eval_epochs
    burn_in_nsamples = args.burn_in_nsamples
    T = (args.total_nsamples - int(G * burn_in_nsamples)) // args.batch_size
    bs = args.batch_size

    # Real data
    mu_real = np.random.randn(num_dim,)
    sigma_real = np.atleast_2d(np.cov(np.random.randn(10, num_dim,), rowvar=False)) + np.identity(num_dim)

    # Generated data
    mu_G = np.random.randn(G, num_dim,)
    cov_G = np.empty((G, num_dim, num_dim,))
    FD_G = np.empty((G,))
    for g in range(G):
        cov_G[g] = np.atleast_2d(np.cov(np.random.randn(10, num_dim,), rowvar=False)) + np.identity(num_dim)
        FD_G[g] = compute_FD_with_stats(mu1=mu_G[g], mu2=mu_real, sigma1=cov_G[g], sigma2=sigma_real)
    instant_regret = FD_G - np.min(FD_G)

    print(colored(f'[FD-online-eval Configs] evaluator: {args.evaluator_name}, bs: {args.batch_size}, T: {T}',
                  'blue'), '\n')

    # Track statistics
    cum_regret = np.zeros((T,))
    op = np.zeros((T,))
    visitation = np.zeros((T, G,))

    for epoch in range(1, num_epoch + 1):

        evaluator = EVALUATOR[args.evaluator_name](G=G, num_dim=num_dim, bs=bs,
                                                   mu_r=mu_real, sigma_r=sigma_real,
                                                   burn_in_nsamples=burn_in_nsamples,
                                                   M=args.fd_ucb_M, kappa=args.fd_ucb_kappa)

        # Burn-in sampling
        if burn_in_nsamples > 0:
            for g in range(G):
                f = np.random.multivariate_normal(mean=mu_G[g], cov=cov_G[g], size=burn_in_nsamples)
                evaluator.update_stats(g=g, batch_feat=f)

        for t in range(T):
            gt = evaluator.select_arm()
            ft = np.random.multivariate_normal(mean=mu_G[gt], cov=cov_G[gt], size=burn_in_nsamples)
            evaluator.update_stats(g=gt, batch_feat=ft)

            cum_regret[t:] += instant_regret[gt]
            op[t:] += 1. if FD_G[gt] == np.min(FD_G) else 0.
            visitation[t:, gt] += 1.

            if (t + 1) % G == 0:
                print(colored(f'evaluator: {args.evaluator_name}, '
                              f'epoch {epoch}, step {t + 1}, ', 'red'))
                print(colored(f'avg.regret: {cum_regret[t] / ((t + 1) * epoch)}, '
                              f'OPR: {op[t] / ((t + 1) * epoch)}, '
                              f'avg.visitation: {visitation[t] / ((t + 1) * epoch)}', 'blue'), '\n')


if __name__ == '__main__':
    main()
