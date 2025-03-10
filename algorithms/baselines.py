import numpy as np
from copy import deepcopy
from scipy.linalg import eigh
from scipy.stats import entropy

from metrics.fd import compute_FD_with_stats


def update_mu_cov(mu, cov, feat, nt):
    bs = feat.shape[0]
    for j in range(1, bs + 1):
        m = nt - bs + j
        if m > 1:
            z = feat[j - 1] - mu
            cov = (m - 2) / (m - 1) * cov + 1 / m * np.outer(z, z)
        mu = (m - 1) / m * mu + 1 / m * feat[j - 1]
    return mu, cov


class fd_naive_greedy:
    def __init__(self, G, num_dim, bs, mu_r, sigma_r, burn_in_nsamples, delta=None, M=None, kappa=None,
                 epsilon_greedy=False, epsilon=0.05):
        self.G = G
        self.bs = bs
        self.visitation = np.zeros((G,))
        self.num_dim = num_dim
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.tilde_fid = np.array([-np.inf for _ in range(G)])
        self.hat_fid = None
        self.hat_mu = np.zeros((G, num_dim,))
        self.hat_cov = np.zeros((G, num_dim, num_dim,))
        self.burn_in_nsamples = burn_in_nsamples
        self.burn_in_phase = np.array([True for _ in range(G)])
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon

    def select_arm(self):
        if self.epsilon_greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.G)
        return np.random.choice(np.where(self.tilde_fid == np.min(self.tilde_fid))[0])

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

    def bonus(self, g):
        pass


class fd_naive_ucb:
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
        self.kappa = 1.
        self.burn_in_phase = np.array([True for _ in range(G)])
        self.burn_in_nsamples = burn_in_nsamples

        self.tr_sigmar_r = np.trace(sigma_r)
        self.M = M
        w_sigma_r = eigh(sigma_r, eigvals_only=True)
        self.norm2_sigma_r = w_sigma_r[-1]
        self.tr_root_sigma_r = np.sum(np.sqrt(w_sigma_r))

    def select_arm(self):
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
        mu_g = deepcopy(self.hat_mu[g])
        n_g = int(self.visitation[g] * self.bs) + self.burn_in_nsamples

        norm2_cov_g = 1.
        tr_cov2_g = self.num_dim
        eff_rank_cov_g = self.num_dim

        L = np.log(1. / self.delta)
        L_prime = np.log(1. / self.delta)

        Delta_mu_g = np.sqrt(1. / n_g * (np.sqrt(1. * tr_cov2_g * L) + 1. * norm2_cov_g * L))
        Delta_cov_g = 1. * (self.kappa ** 2) * norm2_cov_g * np.sqrt((1. * eff_rank_cov_g + L_prime) / n_g) + \
                      (Delta_mu_g ** 2)

        term1 = 1. * Delta_mu_g * (Delta_mu_g + np.linalg.norm(mu_g - self.mu_r))
        term2 = self.tr_root_sigma_r * np.sqrt(1. * Delta_cov_g)
        term3 = self.num_dim * np.sqrt(1. / n_g * L)
        term4 = 1. * norm2_cov_g / n_g * L

        return term1 + term2 + term3 + term4


class is_naive_greedy:
    def __init__(self, G, bs, total_nsamples, num_cls=1008, epsilon_greedy=False, epsilon=0.05):
        self.G = G
        self.total_nsamples = total_nsamples
        self.bs = bs
        self.visitation = np.zeros((G,))
        self.num_cls = num_cls
        self.tilde_is = np.array([+np.inf for _ in range(G)])
        self.hat_log_is = None
        self.tilde_cond_ent = np.zeros(G)
        self.tilde_marginal_dist = np.zeros((G, num_cls,))
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = epsilon

    def select_arm(self):
        if self.epsilon_greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.G)
        return np.random.choice(np.where(self.tilde_is == np.max(self.tilde_is))[0])

    def update_stats(self, g, batch_preds):
        self.visitation[g] += 1
        n_sample_g = int(self.visitation[g] * self.bs)

        # Update H(Y_g|X_g)
        prev_accumulative_cond_ent = self.tilde_cond_ent[g] * (n_sample_g - self.bs)
        for i in range(self.bs):
            ent_i = entropy(batch_preds[i])
            prev_accumulative_cond_ent += ent_i
        self.tilde_cond_ent[g] = prev_accumulative_cond_ent / n_sample_g

        # Update marginal cls distribution
        self.tilde_marginal_dist[g] = ((n_sample_g - self.bs) * self.tilde_marginal_dist[g] +
                                       np.sum(batch_preds, axis=0, keepdims=True)) / n_sample_g

        # Update empirical IS
        self.tilde_is[g] = np.exp(entropy(self.tilde_marginal_dist[g]) - self.tilde_cond_ent[g])


class is_naive_ucb:
    def __init__(self, G, bs, total_nsamples, num_cls=1008, delta=0.05, epsilon_greedy=False, epsilon=0.):
        self.G = G
        self.total_nsamples = total_nsamples
        self.bs = bs
        self.visitation = np.zeros((G,))
        self.num_cls = num_cls
        self.delta = delta
        self.tilde_is = np.empty((G,))
        self.hat_log_is = np.array([+np.inf for _ in range(G)])
        self.tilde_cond_ent = np.zeros(G)
        self.tilde_marginal_dist = np.zeros((G, num_cls,))

    def select_arm(self):
        return np.random.choice(np.where(self.hat_log_is == np.max(self.hat_log_is))[0])

    def update_stats(self, g, batch_preds):

        self.visitation[g] += 1
        n_sample_g = int(self.visitation[g] * self.bs)

        # Update H(Y_g|X_g)
        prev_accumulative_cond_ent = self.tilde_cond_ent[g] * (n_sample_g - self.bs)
        for i in range(self.bs):
            ent_i = entropy(batch_preds[i])
            prev_accumulative_cond_ent += ent_i
        self.tilde_cond_ent[g] = prev_accumulative_cond_ent / n_sample_g

        # Update marginal cls distribution
        self.tilde_marginal_dist[g] = ((n_sample_g - self.bs) * self.tilde_marginal_dist[g] +
                                       np.sum(batch_preds, axis=0, keepdims=True)) / n_sample_g

        # Update empirical IS
        self.tilde_is[g] = np.exp(entropy(self.tilde_marginal_dist[g]) - self.tilde_cond_ent[g])

        # Update hat_IS
        L = np.log(4. / self.delta)
        sig2_cls = np.ones(self.num_cls) / 4.
        e0 = np.sqrt(2 * sig2_cls / n_sample_g * L) + 7 * L / (3 * (n_sample_g - 1))
        sig2_cond_ent = np.log(self.num_cls) ** 2 / 4.
        e1 = np.sqrt(2 * sig2_cond_ent / n_sample_g * L) + 7 * np.log(self.num_cls) * L / (3 * (n_sample_g - 1))

        opt_dist = self.tilde_marginal_dist[g]  # optimistic marginal distribution
        for j in range(self.num_cls):
            delta_j = ((1 / np.e) - opt_dist[j]) / np.abs((1 / np.e) - opt_dist[j]) * e0[j]
            if np.abs(delta_j) > np.abs((1 / np.e) - opt_dist[j]):
                opt_dist[j] = 1 / np.e
            else:
                opt_dist[j] += delta_j
        full_ent = - opt_dist.dot(np.log(opt_dist))

        self.hat_log_is[g] = full_ent - (self.tilde_cond_ent[g] - e1)

