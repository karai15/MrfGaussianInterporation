import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
from src.MrfGaussianInterporation.myfunc import *
from src.MrfGaussianInterporation.MrfBayseInterporation import *

def load_data(miss_rate=0.3):
    # load data
    img = cv2.imread("./data/Mandrill_8bit_32x32.png", 0)  # Lena, Mandrill 256, 64, 32, 16
    height, width = img.shape  # height:y方向, width：x方向
    img = img.astype("float")

    # loss
    mask_2d = np.random.binomial(1, miss_rate, size=img.shape)  # 1:欠損, 0:観測
    img_obs = deepcopy(img)
    # noise
    noise_var = 0.001 * np.average(img_obs) ** 2  # 平均電力のx倍でノイズを追加
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_var), size=img_obs.shape)
    img_obs = img_obs + noise
    img_obs[np.where(mask_2d == 1)] = np.nan

    # reshape
    Lx = img.shape[0]
    Ly = img.shape[1]
    Dy = Lx * Ly
    Dy_loss = np.sum(mask_2d)  # 欠損数
    Dy_obs = Dy - Dy_loss  # 観測数
    y_true = img.reshape(Dy)
    y_obs = img_obs.reshape(Dy)
    mask = mask_2d.reshape(Dy)

    # # plot
    # fig, ax = plt.subplots(1, 3, squeeze=False)
    # ax[0, 0].imshow(img, cmap="jet")
    # ax[0, 1].imshow(img_obs, cmap="jet")
    # ax[0, 2].imshow(noise, cmap="jet")
    # plt.show()

    return y_true, y_obs, mask, Lx, Ly, Dy, Dy_obs, Dy_loss

# 欠損部分を観測部分の平均値で埋める (VIの初期値として利用)
def loss_avg_interpolation(y_obs, mask):
    _y_obs = y_obs[np.where(mask==0)]
    avg_y = np.average(_y_obs)
    y_itpl = copy.deepcopy(y_obs)
    y_itpl[np.where(mask==1)] = avg_y
    return y_itpl

if __name__ == '__main__':

    ###########################################
    # load data
    y_true, y_obs, mask, Lx, Ly, Dy, Dy_obs, Dy_loss = load_data()
    Lambda_a = calc_mrf_cov_inv(Lx, Ly)
    y_avg_itpl = loss_avg_interpolation(y_obs, mask)  # 欠損値を平均値で埋める
    ###########################################

    ###########################################
    # 変分推論
    # param
    option_beta_update = True  # beta の更新の有無
    threshold = 1e-4
    max_iter = 20

    # 事前分布
    prior= {
        "alpha": 0.01,  # alpha, beta は確率変数ではない変数で最尤法で求める
        "beta": 0.01,
        "Lambda_a": Lambda_a,  # 潜在変数の精度行列 (alpha=1の場合)
    }

    # 事後分布の初期値
    posterior ={
        "alpha": prior["alpha"],  # alpha, beta は確率変数ではない変数で最尤法で求める
        "beta": prior["beta"],
        "mu_a": y_avg_itpl,  # 潜在変数の期待値  <-- mu_a, aplha, beta 以外は初期値はなんでもok
        "Cov_a": None,  # 潜在変数の精度行列
        "mu_y_loss": None,  # 欠損値の平均
        "Cov_y_loss": None  # 欠損値の制度行列
    }

    # VI
    mrfGaItpl = MrfGaussianInterpolarion()  # インスタンス化
    posterior = \
        mrfGaItpl.VariationalInference(deepcopy(y_obs), prior, posterior, option_beta_update, max_iter, threshold)
    ###########################################

    ###########################################
    # MSE評価
    Dy = y_true.shape[0]
    MSE_pre = 1 / Dy * np.linalg.norm(y_true - y_avg_itpl, ord=2) ** 2
    MSE_post = 1 / Dy * np.linalg.norm(y_true - posterior["mu_a"], ord=2) ** 2
    print("MSE_pre = ", MSE_pre)
    print("MSE_post = ", MSE_post)
    ###########################################


    ###########################################
    # plot
    # 画像に変換
    img = y_true.reshape(Lx, Ly)
    img_obs = y_obs.reshape(Lx, Ly)
    img_avg_itpl = y_avg_itpl.reshape(Lx, Ly)  # 欠損を初期値で埋めた場合
    img_post = posterior["mu_a"].reshape(Lx, Ly)

    # plot
    fig, ax = plt.subplots(2, 2, squeeze=False)

    ax[0, 0].imshow(img)
    ax[0, 0].set_title("True")

    ax[0, 1].imshow(img_obs)
    ax[0, 1].set_title("Obs")

    ax[1, 0].imshow(img_avg_itpl)
    ax[1, 0].set_title("Pre")

    ax[1, 1].imshow(img_post)
    ax[1, 1].set_title("Post")
    plt.show()
    ###########################################
