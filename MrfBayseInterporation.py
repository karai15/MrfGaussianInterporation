import copy

import numpy as np
from src.MrfGaussianInterporation.myfunc import *


class MrfGaussianInterpolarion():

    def update_y_loss(self, y_obs, prior, posterior, mask):
        beta = posterior["beta"]
        mu_a = posterior["mu_a"]
        mu_a_loss = mu_a[np.where(mask == 1)]

        posterior["mu_y_loss"] = mu_a_loss
        posterior["Cov_y_loss"] = 1 / beta * np.eye(mu_a_loss.shape[0])

        return posterior

    def update_a(self, y_obs, prior, posterior, mask):
        # param
        Dy = y_obs.shape[0]  # 出力数
        Dy_loss = np.sum(mask)  # 欠損数
        Dy_obs = Dy - Dy_loss  # 観測数
        id_obs = np.where(mask == 0)[0]
        id_loss = np.where(mask == 1)[0]

        beta = posterior["beta"]
        alpha = posterior["alpha"]
        y_loss = posterior["mu_y_loss"]
        Lambda_a_pre = alpha * prior["Lambda_a"]

        # 事前精度行列の計算
        Lambda_a_pre_oo = maskmat(Lambda_a_pre, id_obs, id_obs)
        Lambda_a_pre_lo = maskmat(Lambda_a_pre, id_loss, id_obs)
        Lambda_a_pre_ll = maskmat(Lambda_a_pre, id_loss, id_loss)

        # 事後精度行列の計算
        Lambda_a_post_oo = Lambda_a_pre_oo + beta * np.eye(Dy_obs)
        Lambda_a_post_lo = Lambda_a_pre_lo
        Lambda_a_post_ll = Lambda_a_pre_ll + beta * np.eye(Dy_loss)

        Lambda_a_post = np.zeros((Dy, Dy))
        Lambda_a_post = remaskmat(Lambda_a_post, Lambda_a_post_oo, id_obs, id_obs)
        Lambda_a_post = remaskmat(Lambda_a_post, Lambda_a_post_lo, id_loss, id_obs)
        Lambda_a_post = remaskmat(Lambda_a_post, Lambda_a_post_lo.T, id_obs, id_loss)
        Lambda_a_post = remaskmat(Lambda_a_post, Lambda_a_post_ll, id_loss, id_loss)

        # 事後分散
        Cov_a_post = np.linalg.inv(Lambda_a_post)
        # 事後平均
        y_post = copy.deepcopy(y_obs)
        y_post[np.where(mask == 1)] = y_loss
        # 事後分布更新
        posterior["mu_a"] = beta * Cov_a_post @ y_post
        posterior["Cov_a"] = Cov_a_post

        return posterior

    def update_alpha(self, prior, posterior):
        # param
        Lambda_a_pre = prior["Lambda_a"]
        Cov_a = posterior["Cov_a"]
        mu_a = posterior["mu_a"]
        Dy = Lambda_a_pre.shape[0]

        # update
        posterior["alpha"] = Dy / (np.trace(Lambda_a_pre @ Cov_a) + (r_vec(mu_a) @ Lambda_a_pre @ c_vec(mu_a))[0,0] )

        return posterior

    def update_beta(self, y_obs, prior, posterior, mask):

        # param
        mu_a = posterior["mu_a"]
        Cov_a = posterior["Cov_a"]
        mu_y_loss = posterior["mu_y_loss"]
        Cov_y_loss = posterior["Cov_y_loss"]
        mu_y = copy.deepcopy(y_obs)
        mu_y[np.where(mask == 1)] = mu_y_loss
        Dy = mu_y.shape[0]

        # update
        posterior["beta"] = Dy / (np.linalg.norm(mu_y, ord=2)**2 + np.linalg.norm(mu_a, ord=2)**2
                                  + np.trace(Cov_y_loss) + np.trace(Cov_a) - 2 * np.dot(mu_y, mu_a))

        return posterior


    def VariationalInference(self, y_obs, prior, posterior, option_beta_update=True, max_iter=10, threshold=1e-4):

        # param
        mask = np.isnan(y_obs)  # 1:欠損, 0:観測
        alpha_pre = posterior["alpha"]
        beta_pre = posterior["beta"]

        for iter in range(max_iter):

            # print
            print("iter=", iter)
            print("alpha", posterior["alpha"])
            print("beta", posterior["beta"])
            print()

            # E-step
            posterior = self.update_y_loss(y_obs, prior, posterior, mask)  # 欠損値の補間
            posterior = self.update_a(y_obs, prior, posterior, mask)  # 潜在変数の推定

            # M-step
            posterior = self.update_alpha(prior, posterior)  # alpha
            if option_beta_update == True:  # betaの更新は任意
                posterior = self.update_beta(y_obs, prior, posterior, mask) # beta

            # 終了条件
            e_alpha = np.abs(alpha_pre - posterior["alpha"])
            e_beta = np.abs(beta_pre - posterior["beta"])
            if alpha_pre < threshold and beta_pre < threshold:
                break
            alpha_pre = posterior["alpha"]
            beta_pre = posterior["beta"]

        y_post = copy.deepcopy(y_obs)
        y_post[np.where(mask == 1)] = posterior["mu_y_loss"]  # 事後平均

        return  posterior, y_post

# MRFの精度行列の作成
def calc_mrf_cov_inv(height, width):
    """
    MRFの事前分布の精度行列の作成
        p(x_all) = Π_ij {f_ij} = N(x_all|, 0, Cov)
        f_ij = exp[ - alpha/2 * (x_i-x_j)^2] (精度alphaは1とする)
        (ノードの番号は横向きに振っていく)
    :param height: 画像の縦サイズ
    :param width: 画像の横サイズ
    :return: Cov_inv_mat 精度行列 (height*width, height*width)
    """
    # 事前分布の精度行列 (共分散行列の逆行列) の作成

    base_diag = np.zeros((width, width), dtype="float64")  # 精度行列計算のためのベースになる行列を作成
    for k in range(width):
        if k == 0:
            base_diag[k, k] = 3
            base_diag[k, k + 1] = -1

        elif k == width - 1:
            base_diag[k, k] = 3
            base_diag[k, k - 1] = -1

        else:
            base_diag[k, k] = 4
            base_diag[k, k + 1] = -1
            base_diag[k, k - 1] = -1

    # 事前分布の精度行列 (共分散行列の逆行列) の作成
    Cov_inv = np.zeros((height * width, height * width), dtype="float64")
    for y in range(height):
        if y == 0:
            Cov_inv[(y * width):((y + 1) * width), (y * width):((y + 1) * width)] = base_diag[:, :] - np.eye(
                width)
            Cov_inv[(y * width):((y + 1) * width), ((y + 1) * width):((y + 2) * width)] = - np.eye(width)
        elif y == height - 1:
            Cov_inv[(y * width):((y + 1) * width), (y * width):((y + 1) * width)] = base_diag[:, :] - np.eye(
                width)
            Cov_inv[(y * width):((y + 1) * width), ((y - 1) * width):(y * width)] = - np.eye(width)
        else:
            Cov_inv[(y * width):((y + 1) * width), (y * width):((y + 1) * width)] = base_diag[:, :]
            Cov_inv[(y * width):((y + 1) * width), ((y + 1) * width):((y + 2) * width)] = - np.eye(width)
            Cov_inv[(y * width):((y + 1) * width), ((y - 1) * width):(y * width)] = - np.eye(width)

    return Cov_inv
