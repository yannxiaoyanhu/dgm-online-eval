import numpy as np
from scipy.stats import entropy


class is_ucb:
    def __init__(self, G, bs, total_nsamples, num_cls=1000, delta=0.05, epsilon_greedy=False, epsilon=0.):
        self.G = G
        self.total_nsamples = total_nsamples
        self.bs = bs
        self.visitation = np.zeros((G,))
        self.num_cls = num_cls
        self.delta = delta
        self.tilde_is = np.zeros((G,))
        self.hat_log_is = np.array([+np.inf for _ in range(G)])
        self.tilde_cond_ent = np.zeros(G)
        self.tilde_marginal_dist = np.zeros((G, num_cls,))

        self.dist_list = np.zeros((G, total_nsamples, num_cls,))
        self.cond_ent_list = [[] for _ in range(G)]

    def select_arm(self):
        if not np.all(self.visitation):
            return np.random.choice(np.where(self.visitation == 0)[0])
        return np.random.choice(np.where(self.hat_log_is == np.max(self.hat_log_is))[0])

    def update_stats(self, g, batch_preds):
        self.visitation[g] += 1
        n_sample_g = int(self.visitation[g] * self.bs)
        self.dist_list[g][n_sample_g - self.bs:n_sample_g] = batch_preds

        # Update H(Y_g|X_g)
        prev_accumulative_cond_ent = self.tilde_cond_ent[g] * (n_sample_g - self.bs)
        for i in range(self.bs):
            ent_i = entropy(batch_preds[i])
            prev_accumulative_cond_ent += ent_i
            self.cond_ent_list[g].append(ent_i)
        self.tilde_cond_ent[g] = prev_accumulative_cond_ent / n_sample_g

        # Update marginal cls distribution
        self.tilde_marginal_dist[g] = ((n_sample_g - self.bs) * self.tilde_marginal_dist[g] +
                                       np.sum(batch_preds, axis=0, keepdims=True)) / n_sample_g

        # Update empirical IS
        self.tilde_is[g] = np.exp(entropy(self.tilde_marginal_dist[g]) - self.tilde_cond_ent[g])

        # Update hat_IS
        L = np.log(4. / self.delta)
        sig2_cls = np.var(self.dist_list[g][:n_sample_g], axis=0)
        e0 = np.sqrt(2 * sig2_cls / n_sample_g * L) + 7 * L / (3 * (n_sample_g - 1))
        sig2_cond_ent = np.var(np.array(self.cond_ent_list[g]))
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
