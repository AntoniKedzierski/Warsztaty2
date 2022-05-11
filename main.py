import pandas as pd
import numpy as np
from scipy.optimize import brentq

def frobenius(A, B):
    return np.sqrt(np.trace((A - B) @ (A - B)))

def entropy_loss(A, B):
    return np.trace(np.linalg.inv(A) @ B) - np.log(np.linalg.det(np.linalg.inv(A) @ B)) - A.shape[0]

def cs_matrix(sigma, rho, m):
    return sigma * (np.eye(m) * (1 - rho) +  rho * np.ones((m, m)))

def cov_estim(X):
    n = X.shape[0]
    Q = np.eye(n) - np.ones((n, n)) / n
    return X.transpose() @ Q @ X / n

def generate_diag(n):
    for i in range(-n + 1, n):
        yield i, np.diag(np.repeat(1, n - abs(i)), i)

def diag(n, i):
    return np.diag(np.repeat(1, n - abs(i)), i)



class Frobenius:
    def __init__(self, X):
        self.cov = cov_estim(X)
        self.cov_inv = np.linalg.inv(self.cov)
        self.n = self.cov.shape[0]

    def CS(self):
        alpha = np.trace(self.cov @ (np.ones((self.n, self.n)) - np.eye(self.n)))
        rho = alpha / ((self.n - 1) * (np.trace(self.cov)))
        sigma_2 = (np.trace(self.cov) + rho * alpha) / (self.n + self.n * (self.n - 1) * (rho ** 2))
        return rho, sigma_2

    def T1(self):
        rho = self.n * np.trace(self.cov @ diag(self.n, 1)) / (2 * (self.n - 1) * np.trace(self.cov))
        sigma_2 = np.trace(self.cov) / self.n
        return rho, sigma_2

    def AR1(self):
        def optim_target(x, n, cov):
            return -sum(i * (x ** (i - 1)) * np.trace(cov @ H) if i > 0 else 0 for i, H in generate_diag(n)) \
                    + 2 * sum(x ** i * np.trace(cov @ H) if i >= 0 else 0 for i, H in generate_diag(n)) \
                    * sum((n - i) * i * (x ** (2 * i - 1)) for i in range(1, n)) \
                    / (n + 2 * sum((n - i) * (x ** (2 * i)) for i in range(1, n)))
        rho = brentq(optim_target, a=-1, b=1, args=(self.n, self.cov))
        sigma_2 = sum(rho ** i * np.trace(self.cov @ H) if i >= 0 else 0 for i, H in generate_diag(self.n)) \
                    / (self.n + 2 * sum((self.n - i) * rho ** (2 * i) for i in range(1, self.n)))
        return rho, sigma_2


class Entropy:
    def __init__(self, X):
        self.cov = cov_estim(X)
        self.cov_inv = np.linalg.inv(self.cov)
        self.n = self.cov.shape[0]

    def CS(self):
        alpha = np.trace(self.cov_inv @ (np.ones((self.n, self.n)) - np.eye(self.n)))
        rho = - alpha / ((self.n - 1) * np.trace(self.cov_inv) + (self.n - 2) * alpha)
        sigma_2 = self.n / (np.trace(self.cov_inv) + rho * alpha)
        return rho, sigma_2

    def T1(self):
        def s(j, n):
            return np.cos(np.pi * j / (n + 1))

        def optim_target(x, n, cov_inv):
            return sum(2 * s(i, n) / (1 + 2 * x * s(i, n)) for i in range(1, n + 1)) \
                    - n * np.trace(cov_inv @ diag(n, 1)) / (np.trace(cov_inv) + x * np.trace(cov_inv @ diag(n, 1)))

        rho = brentq(optim_target, a=-s(1, self.n), b=s(1, self.n), args=(self.n, self.cov_inv))
        sigma_2 = sum(2 * s(i, self.n) / (1 + 2 * rho * s(i, self.n)) for i in range(1, self.n + 1)) / np.trace(self.cov_inv @ diag(self.n, 1))
        return rho, sigma_2

    def AR1(self):
        def optim_target(x, n, cov_inv):
            return n * sum(i * x ** (i - 1) * np.trace(cov_inv @ H) if i > 0 else 0 for i, H in generate_diag(n)) \
                    / sum(x ** i * np.trace(cov_inv @ H) if i >= 0 else 0 for i, H in generate_diag(n)) + 2 * (n - 1) * x / (1 - x ** 2)

        rho = brentq(optim_target, a=-0.9999, b=0.9999, args=(self.n, self.cov_inv))
        sigma_2 = self.n / sum(rho ** i * np.trace(self.cov_inv @ H) if i >= 0 else 0 for i, H in generate_diag(self.n))
        return rho, sigma_2

class CovMatrix():
    def __init__(self):
        pass

    def fit(self, X):
        self.X = X
        self.frobenius = [Frobenius(x) for x in X]
        self.entropy = [Entropy(x) for x in X]

    def CS(self, metric=None):
        if metric == 'frobenius':
            return pd.DataFrame([x.CS() for x in self.frobenius]).rename(columns={0: 'rho', 1: 'sigma_2'})
        if metric == 'entropy':
            return pd.DataFrame([x.CS() for x in self.entropy]).rename(columns={0: 'rho', 1: 'sigma_2'})
        return pd.concat([
            pd.DataFrame([x.CS() + ('frobenius',) for x in self.frobenius]).rename(columns={0: 'rho', 1: 'sigma_2', 2: 'metric'}),
            pd.DataFrame([x.CS() + ('entropy',) for x in self.entropy]).rename(columns={0: 'rho', 1: 'sigma_2', 2: 'metric'}),
        ], axis=0).reset_index().rename(columns={'index': 'set'})

    def T1(self, metric=None):
        if metric == 'frobenius':
            return pd.DataFrame([x.T1() for x in self.frobenius]).rename(columns={0: 'rho', 1: 'sigma_2'})
        if metric == 'entropy':
            return pd.DataFrame([x.T1() for x in self.entropy]).rename(columns={0: 'rho', 1: 'sigma_2'})
        return pd.concat([
            pd.DataFrame([x.T1() + ('frobenius',) for x in self.frobenius]).rename(columns={0: 'rho', 1: 'sigma_2', 2: 'metric'}),
            pd.DataFrame([x.T1() + ('entropy',) for x in self.entropy]).rename(columns={0: 'rho', 1: 'sigma_2', 2: 'metric'}),
        ], axis=0).reset_index().rename(columns={'index': 'set'})

    def AR1(self, metric=None):
        if metric == 'frobenius':
            return pd.DataFrame([x.AR1() for x in self.frobenius]).rename(columns={0: 'rho', 1: 'sigma_2'})
        if metric == 'entropy':
            return pd.DataFrame([x.AR1() for x in self.entropy]).rename(columns={0: 'rho', 1: 'sigma_2'})
        return pd.concat([
            pd.DataFrame([x.AR1() + ('frobenius',) for x in self.frobenius]).rename(columns={0: 'rho', 1: 'sigma_2', 2: 'metric'}),
            pd.DataFrame([x.AR1() + ('entropy',) for x in self.entropy]).rename(columns={0: 'rho', 1: 'sigma_2', 2: 'metric'}),
        ], axis=0).reset_index().rename(columns={'index': 'set'})

if __name__ == '__main__':
    df = pd.read_csv('barley.csv')
    df = np.transpose(np.log(df.iloc[:, 2:]).to_numpy())
    X = [df[:, 0:200], df[:, 200:400], df[:, 400:600], df[:, 600:]]

    cov_matrix = CovMatrix()
    cov_matrix.fit(X)
    print(cov_matrix.AR1())

    # frob_rho_1, frob_sigma_1 = estim_from_frobenius(cov1)
    # frob_rho_2, frob_sigma_2 = estim_from_frobenius(cov2)
    # frob_rho_3, frob_sigma_3 = estim_from_frobenius(cov3)
    # frob_rho_4, frob_sigma_4 = estim_from_frobenius(cov4)
    #
    # ent_rho_1, ent_sigma_1 = estim_from_entropy(cov1)
    # ent_rho_2, ent_sigma_2 = estim_from_entropy(cov2)
    # ent_rho_3, ent_sigma_3 = estim_from_entropy(cov3)
    # ent_rho_4, ent_sigma_4 = estim_from_entropy(cov4)
    #
    # rozb_frob_1 = frobenius(cov1, cs_matrix(frob_sigma_1, frob_rho_1, cov1.shape[0]))
    # rozb_frob_2 = frobenius(cov2, cs_matrix(frob_sigma_2, frob_rho_2, cov2.shape[0]))
    # rozb_frob_3 = frobenius(cov3, cs_matrix(frob_sigma_3, frob_rho_3, cov3.shape[0]))
    # rozb_frob_4 = frobenius(cov4, cs_matrix(frob_sigma_4, frob_rho_4, cov4.shape[0]))
    #
    # rozb_ent_1 = entropy_loss(cov1, cs_matrix(ent_sigma_1, ent_rho_1, cov1.shape[0]))
    # rozb_ent_2 = entropy_loss(cov2, cs_matrix(ent_sigma_2, ent_rho_2, cov2.shape[0]))
    # rozb_ent_3 = entropy_loss(cov3, cs_matrix(ent_sigma_3, ent_rho_3, cov3.shape[0]))
    # rozb_ent_4 = entropy_loss(cov4, cs_matrix(ent_sigma_4, ent_rho_4, cov4.shape[0]))
    #
    # results = pd.DataFrame({
    #     'rho_f': [frob_rho_1, frob_rho_2, frob_rho_3, frob_rho_4],
    #     'rho_e': [ent_rho_1, ent_rho_2, ent_rho_3, ent_rho_4],
    #     'sigma_f': [frob_sigma_1, frob_sigma_2, frob_sigma_3, frob_sigma_4],
    #     'sigma_e': [ent_sigma_1, ent_sigma_2, ent_sigma_3, ent_sigma_4],
    #     'rozb_f': [rozb_frob_1, rozb_frob_2, rozb_frob_3, rozb_frob_4],
    #     'rozb_e': [rozb_ent_1, rozb_ent_2, rozb_ent_3, rozb_ent_4]
    # }, index=[1, 2, 3, 4])

    # print(results)

