import numpy as np
import ffht
import timeit

import FIHT


def main():

    d = 668426
    K = 1000
    Q = K * 12

    d_aug = 2 ** np.ceil(np.log2(d)).astype('int32')

    rng = np.random.default_rng(2)
    Phi_row_idx = rng.choice(d_aug, Q, replace=False)

    time_total = 0
    for ii in range(100):
        rng = np.random.default_rng()
        supp = rng.choice(d, K, replace=False)
        g0 = np.zeros(d)
        g0[supp] = np.random.randn(K)

        g0 += np.random.randn(d) * 0.01

        supp = np.argpartition(np.abs(g0), -K)[-K:]
        g0_sp = np.zeros(d)
        g0_sp[supp] = g0[supp]

        g0_wht = np.concatenate((g0, np.zeros(d_aug-d))) / np.sqrt(Q)
        ffht.fht(g0_wht)
        y = g0_wht[Phi_row_idx]

        t1 = timeit.default_timer()
        g_rec = FIHT.FastIHT_WHT(y, K, Q, d_aug, Phi_row_idx, top_k_func=1)[0:d]
        t2 = timeit.default_timer()

        time_total += t2 - t1

        rel_diff_rec = np.linalg.norm(g_rec - g0) / np.linalg.norm(g0)
        rel_diff_bestK = np.linalg.norm(g0_sp - g0) / np.linalg.norm(g0)

        print("||g_rec - g|| / ||g|| = ", rel_diff_rec)
        if rel_diff_bestK != 0:
            print("||g_bestK - g|| / ||g|| = ", rel_diff_bestK)
            print("amplification factor = ", rel_diff_rec / rel_diff_bestK)

    print(time_total / 100)


if __name__ == "__main__":
    main()

