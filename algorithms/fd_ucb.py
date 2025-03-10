import numpy as np
from copy import deepcopy
from scipy.linalg import eigh

from metrics.fd import compute_FD_with_stats, compute_efficient_FD_with_stats


def update_mu_cov(mu, cov, feat, nt):
    bs = feat.shape[0]
    for j in range(1, bs + 1):
        m = nt - bs + j
        if m > 1:
            z = feat[j - 1] - mu
            cov = (m - 2) / (m - 1) * cov + 1 / m * np.outer(z, z)
        mu = (m - 1) / m * mu + 1 / m * feat[j - 1]
    return mu, cov


class fd_ucb:
    def __init__(self, G, num_dim, bs, mu_r, sigma_r, burn_in_nsamples, delta=0.05, M=2., kappa=1.,
                 epsilon_greedy=False, epsilon=0.):
        self.G = G
        self.bs = bs
        self.visitation = np.zeros((G,))
        self.num_dim = num_dim
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.tilde_fid = np.empty((G,))
        self.hat_fid = np.array([-np.inf for _ in range(G)])
        self.hat_mu = np.zeros((G, num_dim,))
        self.hat_cov = np.zeros((G, num_dim, num_dim,))
        self.delta = delta
        self.kappa = kappa
        self.burn_in_phase = np.array([True for _ in range(G)])
        self.burn_in_nsamples = burn_in_nsamples

        self.tr_sigmar_r = np.trace(sigma_r)
        self.M = M
        w_sigma_r = eigh(sigma_r, eigvals_only=True)
        self.norm2_sigma_r = w_sigma_r[-1]
        self.tr_root_sigma_r = np.sum(np.sqrt(w_sigma_r))

    def select_arm(self):
        if not np.all(self.visitation):
            return np.random.choice(np.where(self.visitation == 0)[0])
        return np.random.choice(np.where(self.hat_fid == np.min(self.hat_fid))[0])

    def update_stats(self, g, batch_feat):
        if self.burn_in_phase[g] and self.burn_in_nsamples > 0:
            self.hat_mu[g], self.hat_cov[g] = update_mu_cov(mu=self.hat_mu[g], cov=self.hat_cov[g],
                                                            feat=batch_feat, nt=self.burn_in_nsamples)
            self.burn_in_phase[g] = False
        else:
            self.visitation[g] += 1
            n_g = int(self.visitation[g] * self.bs) + self.burn_in_nsamples
            self.hat_mu[g], self.hat_cov[g] = update_mu_cov(mu=self.hat_mu[g], cov=self.hat_cov[g],
                                                            feat=batch_feat, nt=n_g)
            if self.visitation[g] * self.bs >= 2:
                self.tilde_fid[g] = compute_FD_with_stats(mu1=self.hat_mu[g], sigma1=self.hat_cov[g],
                                                          mu2=self.mu_r, sigma2=self.sigma_r)
                hat_fid_g = self.tilde_fid[g] - self.bonus(g)
                self.hat_fid[g] = -np.inf if np.isnan(hat_fid_g) else hat_fid_g

    def bonus(self, g):
        mu_g, cov_g = deepcopy(self.hat_mu[g]), deepcopy(self.hat_cov[g])
        n_g = int(self.visitation[g] * self.bs) + self.burn_in_nsamples

        # Thresholding method for sample covariance matrix (Cai, T. and Liu, W., 2011)
        tau = 2. * np.outer(np.diag(cov_g), np.diag(cov_g))
        tau = self.M * np.sqrt(tau * np.log(self.num_dim) / n_g)
        trunc_cov = cov_g
        trunc_cov[np.abs(trunc_cov) < tau] = 0.

        w_cov_g = eigh(trunc_cov, eigvals_only=True)
        norm2_cov_g = w_cov_g[-1]
        tr_cov2_g = np.sum(w_cov_g ** 2)
        eff_rank_cov_g = np.trace(trunc_cov) / norm2_cov_g

        L = np.log(1. / self.delta)
        L_prime = np.log(1. / self.delta)

        Delta_mu_g = np.sqrt(1. / n_g * (np.sqrt(tr_cov2_g * L) + norm2_cov_g * L))
        Delta_cov_g = (self.kappa ** 2) * norm2_cov_g * np.sqrt((eff_rank_cov_g + L_prime) / n_g) + \
                      (Delta_mu_g ** 2)

        term1 = Delta_mu_g * (Delta_mu_g + np.linalg.norm(mu_g - self.mu_r))
        term2 = self.tr_root_sigma_r * np.sqrt(Delta_cov_g)
        term3 = np.trace(trunc_cov) * np.sqrt(1. / n_g * L)
        term4 = norm2_cov_g / n_g * L

        return term1 + term2 + term3 + term4
