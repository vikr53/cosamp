import numpy as np
import pandas
from scipy import fft
from tensorflow import math as tf_math
import torch
import ffht


TOP_K_ABS_FUNC = np.array([lambda arr, K: np.argpartition(np.abs(arr), -K)[-K:],
                           lambda arr, K: tf_math.top_k(tf_math.abs(arr), K, sorted=False)[1].numpy(),
                           lambda arr, K: torch.topk(torch.abs(torch.tensor(arr)), K,
                                                     largest=True, sorted=False).indices.numpy()],
                          dtype=object)


def FastIHT_WHT(y, K, Q, d, Sigma, top_k_func=1):
    """
    Fast iterative hard thresholding algorithm with partial Walsh-Hadamard Transform sensing matrices.
    y : numpy.ndarray
        the measurement vector
    K : int
        number of nonzero entries in the recovered signal
    Q : int
        dimension of y
    d : int
        dimension of the recovered signal, must be a power of 2
    Sigma : numpy.ndarray
        a Q-dimensional array consisting of row indices of the partial WHT matrix
    top_k_fun : {0, 1, 2}
        indicates the function used for computing the top K indices of a vector
        0 - numpy.argpartition
        1 - tensorflow.math.top_k
        2 - torch.topk

    Packages required: numpy, pandas, ffht, tensorflow, torch
    """

    if (d & (d-1) != 0) or (d <= 0):
        print("The dimension d must be a power of 2.")
        return

    if not(top_k_func in (0, 1, 2)):
        top_k_func = 0

    eps = 1e-4
    max_iter = 25

    res_norms = np.zeros(max_iter)      # recording norms of y - Φ * w_vec
    wht_factor = 1 / np.sqrt(Q)         # scaling factor for WHT

    g = np.zeros(d)         # g: the current iterate
    g_prev = g              # g_prev: the previous iterate

    res = np.zeros(d)
    # res <- transpose(Φ) * y
    res[Sigma] = y * wht_factor
    ffht.fht(res)

    Omega_prev = np.array([], dtype=np.int32)                # Omega_prev: support of g_prev
    # Omega <- top K indices of transpose(Φ) * y
    Omega = TOP_K_ABS_FUNC[top_k_func](res, K)               # Omega: support of g
    # initialize g as the top K projection of transpose(Φ) * y
    g[Omega] = res[Omega]

    for s in range(max_iter):
        if s == 0:
            w_vec = g
            Omega_w = Omega     # Omega_w: support of w_vec
        else:
            g_wht = np.copy(g)
            ffht.fht(g_wht)

            # g_diff <- g - g_prev
            g_diff = np.copy(g)
            g_diff[Omega_prev] -= g_prev[Omega_prev]
            g_diff_wht = np.copy(g_diff)
            ffht.fht(g_diff_wht)

            g_diff_wht_partial = g_diff_wht[Sigma] * wht_factor
            # tau <- <y-Φg, Φ(g-g_prev)> / ||Φ(g-g_prev)||^2
            tau = np.dot(y - g_wht[Sigma] * wht_factor, g_diff_wht_partial) / np.dot(g_diff_wht_partial, g_diff_wht_partial)

            # w_vec <- g + tau * (g-g_prev)
            # and update the support of w (i.e., Omega_w)
            Omega_w = pandas.unique(np.concatenate((Omega, Omega_prev)))
            w_vec = np.copy(g)
            w_vec[Omega_w] += tau * g_diff[Omega_w]

        w_vec_wht = np.copy(w_vec)
        ffht.fht(w_vec_wht)

        res_w = np.zeros(d)
        res_w[Sigma] = (y - w_vec_wht[Sigma] * wht_factor) * wht_factor

        # stopping criterion:
        # if the norm of y - Φ * w_vec remains nearly the same in the last 4 iterations
        # or if the current norm of y - Φ * w_vec is small enough
        res_norms[s] = np.linalg.norm(res_w[Sigma]) / wht_factor
        if s >= 3 and np.std(res_norms[s-3:s+1]) / np.mean(res_norms[s-3:s+1]) < 1e-2:
            break
        elif res_norms[s] < eps:
            break

        # res_w <- transpose(Φ) * (y - Φ * w_vec)
        ffht.fht(res_w)

        res_w_proj_wht = np.zeros(d)
        res_w_proj_wht[Omega_w] = res_w[Omega_w] * wht_factor
        ffht.fht(res_w_proj_wht)

        # alpha_tilde <- ||Proj(res_w, Omega_w)||^2 / ||Φ * Proj(res_w, Omega_w)||^2
        alpha_tilde = np.dot(res_w[Omega_w], res_w[Omega_w]) / np.dot(res_w_proj_wht[Sigma], res_w_proj_wht[Sigma])

        g_prev = g
        Omega_prev = Omega

        # h_vec <- w_vec + alpha_tilde * res_w
        h_vec = alpha_tilde * res_w
        h_vec[Omega_w] += w_vec[Omega_w]

        # set g as the top K projection of h_vec
        # and update the support of g (i.e., Omega)
        Omega = TOP_K_ABS_FUNC[top_k_func](h_vec, K)
        g = np.zeros(d)
        g[Omega] = h_vec[Omega]

        g_wht = np.copy(g)
        ffht.fht(g_wht)

        # res <- transpose(Φ) * (y - Φ * g)
        res = np.zeros(d)
        res[Sigma] = (y - g_wht[Sigma] * wht_factor) * wht_factor
        ffht.fht(res)

        # res_proj_wht <- Φ * Proj(res, Omega)
        res_proj_wht = np.zeros(d)
        res_proj_wht[Omega] = res[Omega] * wht_factor
        ffht.fht(res_proj_wht)

        # alpha <- ||Proj(res, Omega)||^2 / ||Φ * Proj(res, Omega)||^2
        alpha = np.dot(res[Omega], res[Omega]) / np.dot(res_proj_wht[Sigma], res_proj_wht[Sigma])

        # g <- g + alpha * Proj(res, Omega)
        g[Omega] += alpha * res[Omega]

    return g


def FastIHT_DCT(y, K, Q, d, Sigma):
    """
    Fast iterative hard thresholding algorithm with partial Discrete Cosine Transform sensing matrices.
    y : numpy.ndarray
        the measurement vector
    K : int
        number of nonzero entries in the recovered signal
    Q : int
        dimension of y
    d : int
        dimension of the recovered signal
    Sigma : numpy.ndarray
        a Q-dimensional array consisting of row indices of the partial DCT matrix
    """

    eps = 1e-4
    max_iter = 25

    res_norms = np.zeros(max_iter)
    dct_factor = np.sqrt(d/Q)

    g = np.zeros(d)
    g_prev = g

    res_tmp = np.zeros(d)
    res_tmp[Sigma] = y

    res = fft.idct(res_tmp, norm="ortho") * dct_factor

    Omega = np.argpartition(np.abs(res), -K)[-K:]
    g[Omega] = res[Omega]

    for s in range(max_iter):
        if s == 0:
            w_vec = g
        else:
            g_dct = fft.dct(g, type=2, norm="ortho") * dct_factor
            g_diff_dct = fft.dct(g - g_prev, norm="ortho") * dct_factor

            tau = np.dot(y - g_dct[Sigma], g_diff_dct[Sigma]) / np.dot(g_diff_dct[Sigma], g_diff_dct[Sigma])
            w_vec = g + tau * (g - g_prev)

        w_vec_dct = fft.dct(w_vec, norm="ortho") * dct_factor

        res_tmp[Sigma] = y - w_vec_dct[Sigma]
        res_w = fft.idct(res_tmp, norm="ortho") * dct_factor

        res_norms[s] = np.linalg.norm(res_w)
        if s >= 3 and np.std(res_norms[s-3:s+1]) / np.mean(res_norms[s-3:s+1]) < 1e-2:
            break
        elif res_norms[s] < eps:
            break

        Omega_w = (w_vec != 0)
        res_w_proj_dct = np.zeros(d)
        res_w_proj_dct[Omega_w] = res_w[Omega_w] * dct_factor
        fft.dct(res_w_proj_dct, norm="ortho", overwrite_x=True)

        alpha_tilde = np.dot(res_w[Omega_w], res_w[Omega_w]) / np.dot(res_w_proj_dct[Sigma], res_w_proj_dct[Sigma])

        g_prev = g

        h_vec = w_vec + alpha_tilde * res_w
        Omega = np.argpartition(np.abs(h_vec), -K)[-K:]
        g = np.zeros(d)
        g[Omega] = h_vec[Omega]

        g_dct = fft.dct(g, norm="ortho") * dct_factor

        res_tmp[Sigma] = y - g_dct[Sigma]
        res = fft.idct(res_tmp, norm="ortho") * dct_factor

        res_proj_dct = np.zeros(d)
        res_proj_dct[Omega] = res[Omega] * dct_factor
        fft.dct(res_proj_dct, norm="ortho", overwrite_x=True)

        alpha = np.dot(res[Omega], res[Omega]) / np.dot(res_proj_dct[Sigma], res_proj_dct[Sigma])

        g[Omega] += alpha * res[Omega]

    return g
