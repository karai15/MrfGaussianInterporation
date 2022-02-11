import numpy as np
# import matplotlib.pyplot as plt

# 配列を列ベクトルに変形 (N,1)
def c_vec(array):
    return array.reshape(-1, 1)

# 配列を行ベクトルに変形 (1,N)
def r_vec(array):
    return array.reshape(1, -1)

# 指定された要素で構成される部分行列を抽出
def maskmat(mat, id_line, id_col):
    mat_ = mat[id_line, :]
    submat = mat_[:, id_col]
    return submat

# 部分行列を指定された要素に戻す
def remaskmat(mat, submat, id_line, id_col):

    for i, idl in enumerate(id_line):
        for j, idc in enumerate(id_col):
            mat[idl, idc] = submat[i, j]
    return mat