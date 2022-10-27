"""Hilbert space construction for estimation problems."""

import numpy as np
from scipy.linalg import block_diag


class MultiKernel:
    """Multiple Kernel method for estimation.

    References
    ----------
    [1] Yukawa, Masahiro. "Multikernel adaptive filtering." IEEE Transactions on Signal Processing 60.9 (2012): 4672-4682.
    """

    def __init__(
        self, kernel_type="Gaussian", kernel_params=[0.1, 0.3], num_dicts=33, kinv_norm_factor=0.0, tau=0.95, nonnegative=False, max_iter=5
    ) -> None:
        if kernel_type == "Gaussian":
            self.kernel_func = self.gauss_kern
        elif kernel_type == "Laplacian":
            self.kernel_func = self.laplace_kern
        else:
            raise NotImplementedError
        self.kernel_params = kernel_params
        self.num_kernels = len(kernel_params)
        self.num_dicts = num_dicts
        self.dict_elements = None
        self.tau = tau
        self.kernel_matrix = None
        self.nonnegative = nonnegative  # enforce nonnegative weights
        self.max_iter = max_iter

        self.kinv_norm_factor = kinv_norm_factor

    def gauss_kern(self, x, y, s):
        return np.exp(-np.sum((x - y) ** 2, axis=-1) / (2 * s ** 2))

    def laplace_kern(self, x, y, s):
        return np.exp(-np.sum(np.abs(x - y), axis=-1) / s)

    def gen_dict(self, locs):

        # dict_elements = self.dict_elements
        N = locs.shape[0]

        # if dict_elements is None:
        # dict_elements = np.array([locs[0, :]])
            # self.dict_indexes = [0]

        # dict_indexes = self.dict_indexes

        scales = self.kernel_params

        # K_list, locs_list, loc_indexes = [], [], []
        num_dicts = 0
        max_iters = self.max_iter
        iter_count = 0

        while num_dicts != self.num_dicts and max_iters > iter_count:

            dict_elements = np.array([locs[0, :]])

            for i in range(N):
                K_val = 0
                # if i in dict_indexes:
                #     continue
                for j in range(dict_elements.shape[0]):
                    for s in range(len(scales)):
                        K_val = np.maximum(K_val, self.kernel_func(locs[i], dict_elements[j], scales[s]))

                if K_val <= self.tau:
                    dict_elements = np.append(dict_elements, locs[i, np.newaxis, :], axis=0)

            # self.num_dicts = self.num_dicts if self.num_dicts <= N else N
            # sorted_indexes = np.argsort(K_list)[:self.num_dicts - 1]
            # # pick locs corresponding to first num_dicts smallest Ks
            # dict_selected = np.array(locs_list)[sorted_indexes, :]

            # self.dict_elements = np.append(np.array(dict_elements), dict_selected, axis=0)
            # self.dict_indexes = np.append(self.dict_indexes, np.array(loc_indexes)[sorted_indexes])

            # self.dict_elements = dict_elements
            num_dicts = dict_elements.shape[0]
            if num_dicts > self.num_dicts:
                # reduce tau
                self.tau -= 0.05 * (num_dicts - self.num_dicts)/self.num_dicts
            elif num_dicts < self.num_dicts:
                self.tau += 0.05 * (self.num_dicts - num_dicts)/self.num_dicts
            else:
                break

            iter_count += 1

        self.dict_elements = dict_elements

        self.num_dicts = dict_elements.shape[0]

        self.vector_dim = self.num_dicts * self.num_kernels

    def gen_kernel_matrix(self):
        K = []
        D = self.num_dicts
        for s in self.kernel_params:
            K_tmp = np.zeros((D, D))
            for i in range(D):
                for j in range(D):
                    K_tmp[i, j] = self.kernel_func(self.dict_elements[i], self.dict_elements[j], s)
            K.append(K_tmp)

        K = block_diag(*K)

        self.kernel_matrix = K + self.kinv_norm_factor * np.eye(K.shape[0])

        self.kernel_matrix_inv = np.linalg.inv(self.kernel_matrix)

    def kernel_eval(self, x):
        k = []
        for s in self.kernel_params:
            k.append(self.kernel_func(x, self.dict_elements, s).flatten())

        return np.concatenate(k, axis=0)

    def projection_hyperslab(self, vector, feature, measurement, slab_width):

        # K = self.kernel_matrix
        if self.kernel_matrix is None:
            self.gen_kernel_matrix()
        Kinv = self.kernel_matrix_inv
        y = self.kernel_eval(feature)
        yi = Kinv @ y
        wTy = np.inner(vector, y)


        if wTy > measurement + slab_width:
            return vector - ((wTy - measurement - slab_width) * yi / (np.inner(yi, y)))
        elif wTy < measurement - slab_width:
            return vector - ((wTy - measurement + slab_width) * yi / (np.inner(yi, y)))
        else:
            return vector


