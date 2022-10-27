"""Wireless multiple access communication class"""

from turtle import delay
from bs4 import ResultSet
import numpy as np
import math
from multiprocessing import Pool
from joblib import Parallel, delayed


class NetworkGraph:
    """Graph of wirelessly networked agents based on agents' locations (2D and 3D)."""

    def __init__(
        self, agents_locs, connectivity_radius=0.3, alg_connectivity=0.0, check_max_iter=1000, rng=None
    ) -> None:

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.agents_locs = agents_locs
        self.num_agents = agents_locs.shape[0]
        self.connectivity_radius = connectivity_radius
        self.alg_connectivity = alg_connectivity

        self.max_iter = check_max_iter

        self.timer = 0
        self.timer_end = 1e9

        self.gen_connected_graph()

    def find_connected_agents(self):

        tmp_mat = np.stack([self.agents_locs] * self.num_agents, axis=1)
        tmp_mat = tmp_mat - np.transpose(tmp_mat, [1, 0, 2])
        distance_matrix = np.linalg.norm(tmp_mat, axis=2, ord=2)

        self.distance_matrix = distance_matrix

        self.connected_agents = np.where(
            distance_matrix >= self.connectivity_radius, 0.0, 1.0
        ) - np.eye(self.num_agents)

    def check_if_digraph_connected(self):
        # check if graph is strongly connected
        A = self.connected_agents
        N = self.num_agents

        # First check using matrix power
        if (np.linalg.matrix_power((np.eye(N) + A), N - 1) <= 0).any():
            return False
        else:  # second check via algebraic connectivity
            indegree_vec = np.sum(A, axis=1)  # row sum
            in_laplacian = np.diag(indegree_vec) - A
            s = np.linalg.svd(in_laplacian, compute_uv=False)

            # remove eigval lower than e-10
            s = np.where(s < 1e-10, 0, s).flatten()

            if s[-2] <= self.alg_connectivity:
                return False
            else:
                return True

    def gen_connected_graph(self):
        counter = 0

        while True:
            self.find_connected_agents()

            is_connected = self.check_if_digraph_connected()

            if is_connected:
                break

            else:
                

                if counter >= self.max_iter:
                    self.connectivity_radius += abs(self.rng.normal(0.0, 0.05 * np.max(self.agents_locs)))
                    # raise RuntimeError(
                    #     "Maximum recursion count of `%d` reached before a connected digraph can be found."
                    #     "Consider changing network parameters." % self.max_iter
                    # )
                    counter = 0
                counter += 1


class OTA_Comp:
    """Over-the-air computation (OTA-Comp) protocols"""

    def __init__(
        self,
        agent_locs,
        dom_range=(0.0, 1.0),
        ota_reps=100,
        SNR=0.0,
        connectivity_radius=0.3,
        pow_at_conn_rad=-50,
        alg_connectivity=0,
        check_max_iter=1000,
        gamma_mul=1.0,
        field_size=1.0,
        noiseless=False,
        ignore_channel_powers=False,
        rng=None,
        genie_aided=False,
        multiproc=False,
    ) -> None:

        network_obj = NetworkGraph(
            agent_locs,
            connectivity_radius=connectivity_radius,
            alg_connectivity=alg_connectivity,
            check_max_iter=check_max_iter,
            rng=rng,
        )
        self.connectivity_radius_init = connectivity_radius
        self.connectivity_radius = network_obj.connectivity_radius
        self.network = network_obj
        self.ota_reps = ota_reps
        self.delta_min, self.delta_max = dom_range
        self.SNR = SNR
        self.pow_at_conn_rad = pow_at_conn_rad
        self.noise_scale = np.sqrt(10 ** ((-SNR / 10) + (self.pow_at_conn_rad / 10)))
        self.gamma_mul = gamma_mul
        self.gamma_val = None
        self.noiseless = noiseless
        self.ignore_channel_powers = ignore_channel_powers
        self.multiproc = multiproc
        self.field_size = field_size

        self.nmse_m = []
        self.nmse_c = []

        self.timer = 0
        self.genie_aided = genie_aided

    def __iter__(self):
        self.timer = 0

        self.num_agents = self.network.num_agents
        self.agents_locs = self.network.agents_locs
        self.adjacency_matrix = self.network.connected_agents
        self.distance_matrix = self.network.distance_matrix
        self.prop_constant = (self.connectivity_radius ** 2) * (10 ** (self.pow_at_conn_rad / 10))
        self.rng = self.network.rng
        self.set_gamma()

        self.receive_power_matrix = self.rcv_powers_matrix()

        return self

    def __next__(self):

        # capture any update in agents locations
        self.network.agents_locs = self.agents_locs
        self.network.connectivity_radius = self.connectivity_radius_init

        self.network.gen_connected_graph()

        self.num_agents = self.network.num_agents
        self.adjacency_matrix = self.network.connected_agents
        self.distance_matrix = self.network.distance_matrix
        self.connectivity_radius = self.network.connectivity_radius
        self.prop_constant = (self.connectivity_radius ** 2) * (10 ** (self.pow_at_conn_rad / 10))

        self.receive_power_matrix = self.rcv_powers_matrix()

    def rcv_powers_matrix(self, transmit_power=1):

        D = self.distance_matrix
        masked_D = np.multiply(D, self.adjacency_matrix)  # remove links that are not connected
        dist_sqr_matrix = masked_D ** 2
        # proportional power received matrix
        W = (
            transmit_power
            * self.prop_constant
            * np.divide(1, dist_sqr_matrix, out=np.zeros_like(self.adjacency_matrix), where=self.adjacency_matrix != 0)
        )
        return W
    
    def set_gamma(self):
        self.gamma_val = self.gamma_mul / (self.ota_reps * self.num_agents * np.pi * (self.connectivity_radius/self.field_size)**2 * self.prop_constant / (self.connectivity_radius/10)**2)

    def WMAC_output(self, inputs):

        num_agents = self.num_agents
        M = inputs.shape[1]
        noiseless = self.noiseless
        ignore_channel_powers = self.ignore_channel_powers

        if self.genie_aided:
            self.delta_min, self.delta_max = np.min(inputs) - 0.1, np.max(inputs) + 0.1

        output_data = np.zeros((num_agents, M + 1))

        if ignore_channel_powers:
            receive_power_matrix = self.adjacency_matrix * np.median(self.receive_power_matrix)
        else:
            receive_power_matrix = self.receive_power_matrix
            # self.gamma_val = gamma_mul / (self.ota_reps * np.max(np.sum(receive_power_matrix, axis=0)))

        
        n_indices = np.nonzero(self.adjacency_matrix)

        def parallel_job(i):
            neighbor_indexes = n_indices[1][np.argwhere(n_indices[0] == i)].flatten()
            neighbor_inputs = inputs[neighbor_indexes, :]
            v_m, v_c, v_m_n, v_c_n = self.single_receiver_output(
                neighbor_inputs,  # inputs from the neighbors
                receive_power_matrix[i, neighbor_indexes],  # power received from neighbors
            )

            if noiseless:
                return np.concatenate([v_m_n, [v_c_n]], axis=0)
            else:
                return np.concatenate([v_m, [v_c]], axis=0)
                # return np.concatenate([v_m, [v_c], v_m_n, [v_c_n]], axis=0)


        if self.multiproc:
            results = Parallel(n_jobs=-1)(delayed(parallel_job)(i) for i in range(num_agents))
        else:
            results = []
            for i in range(num_agents):
                results.append(parallel_job(i))
        results = np.stack(results, axis=0)
        output_data = results[:, :M+1]

        if 0:
            noiseless_output = results[:, M+1:]
            self.nmse_m.append(np.mean((noiseless_output[:, :M] - output_data[:, :M])**2 * (np.where((noiseless_output[:, :M])**2 == 0, np.zeros_like(noiseless_output[:, :M]), 1 / (noiseless_output[:, :M])**2))))
            self.nmse_c.append(np.mean((noiseless_output[:, -1] - output_data[:, -1])**2 * (np.where((noiseless_output[:, -1])**2 == 0, np.zeros_like(noiseless_output[:, -1]), 1 / (noiseless_output[:, -1])**2))))

        return self.gamma_val * output_data

    def encoder(self, inputs):

        ota_reps = self.ota_reps
        N, M = inputs.shape

        U_samples = 2*self.rng.integers(0, 2, size=(N, M + 1, ota_reps)) - 1
        U_samples = U_samples.astype(complex)

        m_mod = 1 / (self.delta_max - self.delta_min) * np.repeat(np.expand_dims(inputs - self.delta_min, axis=-1), ota_reps, axis=-1)
        encoded_m = np.multiply(np.sqrt(m_mod), U_samples[:, :-1, :])
        encoded_c = U_samples[:, -1, :]

        return encoded_m, encoded_c

    def WMAC_otac(self, encoded_m, encoded_c, rcv_pow):
        rng = self.rng
        ota_reps = self.ota_reps
        N, M, _ = encoded_m.shape
        noise_scale = self.noise_scale

        rcv_pow_rep = np.repeat(rcv_pow.reshape([-1, 1, 1]), M, axis=1)
        rcv_pow_rep = np.repeat(rcv_pow_rep, ota_reps, axis=-1)

        encoded_m = np.multiply(np.sqrt(rcv_pow_rep), encoded_m)
        encoded_c = np.multiply(np.sqrt(rcv_pow_rep[:, 0, :]), encoded_c)

        channel_samples = rng.standard_normal([N, 1, ota_reps]) + 1j * rng.standard_normal([N, 1, ota_reps])

        channel_samples /= math.sqrt(2)
        channel_samples_m = np.repeat(channel_samples, M, axis=1)
        channel_samples_c = np.squeeze(channel_samples)

        rcv_signal_m = np.sum(np.multiply(channel_samples_m, encoded_m), axis=0)
        rcv_signal_c = np.sum(np.multiply(channel_samples_c, encoded_c), axis=0)

        rcv_signal_m += (
            rng.normal(0, noise_scale, [M, ota_reps]) + 1j * rng.normal(0, noise_scale, [M, ota_reps])
        ) / math.sqrt(2)
        rcv_signal_c += (
            rng.normal(0, noise_scale, [ota_reps]) + 1j * rng.normal(0, noise_scale, [ota_reps])
        ) / math.sqrt(2)

        return rcv_signal_m, rcv_signal_c

    def rcv_postproc(self, rcv_signal_m, rcv_signal_c):

        ota_reps = self.ota_reps
        noise_scale = self.noise_scale

        ota_noise_mean = ota_reps * (noise_scale ** 2)
        y_m_norm = np.sum(np.abs(rcv_signal_m) ** 2, axis=1)
        y_c_norm = np.sum(np.abs(rcv_signal_c) ** 2)
        v_c = y_c_norm - ota_noise_mean
        v_m = (y_m_norm - ota_noise_mean) * (self.delta_max - self.delta_min) + self.delta_min * v_c

        return v_m, v_c


    def single_receiver_output(self, inputs, rcv_pow):

        # Assume same channel for all M array elements

        ota_reps = self.ota_reps
        N, M = inputs.shape
        noiseless = self.noiseless

        encoded_m, encoded_c = self.encoder(inputs)
        
        rcv_signal_m, rcv_signal_c = self.WMAC_otac(encoded_m, encoded_c, rcv_pow)

        v_m, v_c = self.rcv_postproc(rcv_signal_m, rcv_signal_c)

        if noiseless:
            v_m_noiseless = np.sum(
                np.multiply(np.repeat(rcv_pow.reshape([-1, 1]), M, axis=-1), inputs),
                axis=0
            ) * ota_reps
            v_c_noiseless = np.sum(rcv_pow) * ota_reps
        else:
            v_m_noiseless, v_c_noiseless = 0, 0

        # error_m = np.sqrt(np.sum((v_m - v_m_noiseless)**2)) / np.sqrt(np.sum((v_m_noiseless)**2))
        # error_c = np.abs(v_c - v_c_noiseless) / np.abs(v_c_noiseless)

        # print("OTA NMSE in v_m, v_c", error_m, error_c)

        return v_m, v_c, v_m_noiseless, v_c_noiseless


    