#!/usr/bin/python

import numpy as np
import argparse
from data_gen_random_field import Oceanograph_realdata_winter
from hilbert_sp_const import MultiKernel
from network_class import OTA_Comp
import os
import csv
import json

"""
    Simulations for Random field estimation using Distributed subgradient method.
"""


if __name__ == "__main__":

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = "realdata"
    results_dir = os.path.join(curr_dir, "results")
    plots_dir = os.path.join(curr_dir, "plots")

    parser = argparse.ArgumentParser(
        prog="Random Field estimation using distributed optimization over-the-air algorithm.",
        description="Iterative algorithm to estimate the random field via a network of agents connected"
        "wirelessly and exchanging information using OTA-Comp framework. "
        "Program supports multiple algorithms with OTAC: D-CHYPASS, RFFMKAF. ",
    )
    parser.add_argument("--rand-seed", "-S", type=int, default=None, help="Random seed.")
    parser.add_argument("--sim-id", "-I", type=int, default=1, help="Simulation ID to get config from CSV.")
    parser.add_argument("--debug", action="store_true", help="Run in Debug mode.")
    parser.add_argument("--multiproc", action="store_true", help="Multiprocessing.")
    args = parser.parse_args()

    seed = args.rand_seed
    if seed is None:
        inrng = np.random.default_rng()
        seed = inrng.integers(0, 1e8)
    sim_id = args.sim_id    
    debug = args.debug
    multiproc = args.multiproc

    if sim_id is None:
        raise ValueError("Param `sim_id` must be specified.")
    sim_file = curr_dir + "/%s_simulations.csv" % prefix
    with open(sim_file, "r") as f:
        cdata = csv.reader(f, delimiter=",", skipinitialspace=True)
        i = 0
        for crow in cdata:
            if i == 0:
                field_names = crow
                i += 1
                continue
            sid = int(crow[0])
            if sid == sim_id:
                to_print = json.dumps({field_names[k]: crow[k] for k in range(len(crow))}, indent=4)
                (
                    N,
                    Q,
                    D,
                    param_mu,
                    param_e,
                    dmin,
                    dmax,
                    OTA_reps,
                    update_steps,
                    gamma_mul,
                    beta_exp,
                    beta_steps,
                    iter_max,
                    connectivity_radius,
                    pow_at_conn_rad,
                    snr_at_conn_rad,
                    measurement_noise_scale,
                    comm_type,
                    run_name,
                ) = tuple(crow[1:])
                N, Q, D, OTA_reps, update_steps, beta_steps, iter_max = (
                    int(N),
                    int(Q),
                    int(D),
                    int(OTA_reps),
                    int(update_steps),
                    int(beta_steps),
                    int(iter_max),
                )
                (
                    param_mu,
                    param_e,
                    dmin,
                    dmax,
                    gamma_mul,
                    beta_exp,
                    connectivity_radius,
                    pow_at_conn_rad,
                    snr_at_conn_rad,
                    measurement_noise_scale,
                ) = (
                    float(param_mu),
                    float(param_e),
                    float(dmin),
                    float(dmax),
                    float(gamma_mul),
                    float(beta_exp),
                    float(connectivity_radius),
                    float(pow_at_conn_rad),
                    float(snr_at_conn_rad),
                    float(measurement_noise_scale),
                )
                comm_type, run_id = str(comm_type), str(run_name)
                weights_dom_range = [dmin, dmax]
                break

    # Fixed Parameters
    field_size = 1000  # size of field in meters
    alg_connectivity = 0.1
    data_depth_limit = 400

    sum_sid = 0
    sim_id = sim_id + sum_sid

    # check for existing runs in the results
    tmp_f = "res_%s_S%d_R%s_Sd%d_"%(prefix, sim_id, run_id, seed)
    matched_files = []
    mlen = 0
    msel = ""

    for filen in os.listdir(curr_dir + "/results"):
        if tmp_f in filen:
            matched_files.append(filen)
    if len(matched_files) > 1:
        for mfile in matched_files:
            with open(curr_dir + "/results/" + mfile, 'r') as f:
                dd = csv.reader(f)
                data = dd.__next__()
                if mlen < len(data):
                    mlen = len(data)
                    msel = mfile

    elif len(matched_files) == 1:
        msel = matched_files[0]
        with open(curr_dir + "/results/" + msel, 'r') as f:
            dd = csv.reader(f)
            data = dd.__next__()
            mlen = len(data)
    
    if mlen > 0:
        rnd_id = msel[:-4].split("_")[-1]
        run_id = "%s_S%d_R%s_Sd%d_%s" % (prefix, sim_id, run_name, seed, rnd_id)
        # check for existing trained weights
        w_file = curr_dir + "/results/weights_%s.out" % (run_id)
        try:
            w = np.loadtxt(w_file)
        except:
            mlen = 0
            print("ERROR: Could not load the previous weights. Starting training anew from 0.")

    if mlen == 0:
        rnd_id = str("%0004x" % np.random.randint(0, 1e8 + 1))
        run_id = "%s_S%d_R%s_Sd%d_%s" % (prefix, sim_id, run_name, seed, rnd_id)
        w_file = curr_dir + "/results/weights_%s.out" % (run_id)
        iter_start = 0
    else:
        iter_start = mlen

    kinv_norm_factor = 1e-5

    # printer
    def my_print(txt):
        my_file = curr_dir + "/slurm_logs/%s.out" % (run_id)
        # dt = datetime.now()
        with open(my_file, "a") as f:
            # f.write(dt.strftime("%d-%m-%Y %H:%M:%S"))
            # f.write("\t\t")
            f.write(txt)
            f.write("\n")

        if debug:
            # print(dt, txt)
            print(txt)

    def save_nmse_data(data):
        my_results = curr_dir + "/results/res_%s.csv" % (run_id)
        with open(my_results, "a") as f:
            f.write("%.4f,"%data)

    def save_wavg_data(data):
        my_results = curr_dir + "/results/cons_%s.csv" % (run_id)
        with open(my_results, "a") as f:
            f.write("%.4f,"%data)

    dict_file = curr_dir + "/results/dicts_%s.npz" % (run_id)

    for pp in to_print.split("\n"):
        my_print(pp)
    my_print("SEED %d" % seed)
    my_print("SUM_SID %s" % sum_sid)
    my_print("RUN_ID %s" % run_id)
    if iter_start > 0:
        my_print("Restarting run at %d" % iter_start)

    ###################################################
    # data generator
    data_gen = Oceanograph_realdata_winter(
        N, field_size, measurement_noise_scale, depth_limit=data_depth_limit, seed=seed, partition_data=True,
    )

    # generate first data measurements
    agents_locs, measurements = data_gen.next_measurements()

    #####################################################
    # Initialize communication network object - OTA-Comp
    OTAC_obj = OTA_Comp(
        agents_locs,
        dom_range=weights_dom_range,
        ota_reps=OTA_reps,
        SNR=snr_at_conn_rad,
        connectivity_radius=connectivity_radius,
        alg_connectivity=alg_connectivity,
        pow_at_conn_rad=pow_at_conn_rad,
        gamma_mul=gamma_mul,
        field_size=field_size,
        rng=data_gen.rng,
    )

    OTA_Comp_iter = iter(OTAC_obj)

    # Communication receiver noise
    noise_scale = np.sqrt(10 ** ((-snr_at_conn_rad / 10) + (pow_at_conn_rad / 10)))
    my_print("Comm_noise_scale : %.3f" % noise_scale)

    #####################################################

    # Estimation model constructors
    # Initialize Multikernel HS construction object
    # Kernel params
    kernel_scales = [50, 300]
    if D == 25:
        tau_val = 0.715
    elif D == 50:
        tau_val = 0.81
    elif D == 100:
        tau_val = 0.883
    elif D == 200:
        tau_val = 0.93075
    else:
        tau_val = 0.9
    mk_obj = MultiKernel(
        kernel_type="Gaussian",
        kernel_params=kernel_scales,
        num_dicts=D,
        kinv_norm_factor=kinv_norm_factor,
        tau=tau_val,
        max_iter=20,
    )

    if 1:  # use random dict
        data = data_gen.train_data
        locs_for_dicts = data[data_gen.rng.choice(np.arange(data.shape[0]), D * 10), :3]
        mk_obj.gen_dict(locs_for_dicts)
    else:  # dict from user locations
        mk_obj.gen_dict(agents_locs)
    
    if iter_start == 0:
        # save dicts
        np.savez(dict_file, dict_elements=mk_obj.dict_elements)
    else:
        dict_d = np.load(dict_file)
        mk_obj.dict_elements = dict_d["dict_elements"]

    my_print("Num dicts %d" % mk_obj.num_dicts)
    mk_obj.gen_kernel_matrix()

    ## Validation
    # Field grid sampler

    kern_lloc = []
    for i in range(data_gen.test_data.shape[0]):
        lloc = np.array([data_gen.test_data[i, 0], data_gen.test_data[i, 1], data_gen.test_data[i, 2]])
        kern_lloc.append(mk_obj.kernel_eval(lloc))
    kern_lloc = np.squeeze(np.stack(kern_lloc, axis=1))
    M = mk_obj.vector_dim
    
    my_print("Param Dimensionality : %d" % M)

    vector_ones = np.ones((M,))
    tfv = np.repeat(np.expand_dims(data_gen.test_data[:, -1], axis=0), N, axis=0)
    tfv_norm = np.sum((data_gen.test_data[:, -1]) ** 2)

    # NMSE func
    def nmse(param_est):

        num = np.mean(
            np.sum(
                (tfv - param_est @ kern_lloc) ** 2,
                axis=-1,
            )
            / tfv_norm
        )

        return num

    #################
    if iter_start == 0:
        w = np.zeros((N, M))
    save_nmse, save_w_avg, save_num_zeros = [], [], []

    def projection_constraint_set(x):
        x_tmp = np.where(x < weights_dom_range[0], np.ones_like(x) * weights_dom_range[0], x)
        x_tmp = np.where(x > weights_dom_range[1], np.ones_like(x) * weights_dom_range[1], x_tmp)
        return x_tmp


    next_update_at = data_gen.rng.poisson(update_steps)
    update_counter = 1
    
    ## Begin iterations in time
    for i in range(iter_start, iter_max):  # iterate in time

        if update_counter == next_update_at:  # update locs
            agents_locs, measurements = data_gen.next_measurements()
            # update communication model
            OTA_Comp_iter.agents_locs = agents_locs
            next(OTA_Comp_iter)
            next_update_at = data_gen.rng.poisson(update_steps)
            update_counter = 1

        update_counter += 1

        ## local sub-gradient update step

        for j in range(N):  # local optimization step for each agent
            w_tmp = w[j, :].copy()

            proj = mk_obj.projection_hyperslab(w_tmp, agents_locs[j, :], measurements[j], slab_width=param_e)

            # APSM
            w_tmp = w_tmp - param_mu * (w_tmp - proj)

            w[j, :] = w_tmp


        if not OTA_Comp_iter.genie_aided:
            w = projection_constraint_set(w.copy())

        if beta_steps > 0:  # diminishing step-size
            beta_i = 1 / (((i // beta_steps) + 1) ** beta_exp)
        else:  # constant step-size
            beta_i = beta_exp

        gamma_val = 0

        if comm_type == "OTAC":
            if gamma_mul == 0.0:
                gamma_val = 1.0 / (OTA_Comp_iter.ota_reps * np.max(np.sum(OTA_Comp_iter.receive_power_matrix, axis=0)))
                OTA_Comp_iter.gamma_val = gamma_val
            else:
                gamma_val = OTA_Comp_iter.gamma_val
            
            OTAC_outputs = OTA_Comp_iter.WMAC_output(w.copy())

            w_new = np.zeros_like(w)

            for k in range(N):
                w_new[k, :] = (
                    np.multiply((vector_ones - beta_i * OTAC_outputs[k, -1]), w[k, :]) + beta_i * OTAC_outputs[k, :-1]
                )

        elif comm_type == "OSN":  # one-shot noiseless averaging
            w_new = np.repeat(np.mean(w, axis=0, keepdims=True), w.shape[0], axis=0)
        elif comm_type == "LOC":  # Only local information, no sharing
            w_new = w.copy()
        else:
            raise NotImplementedError("`%s` communication type is not recognised." % comm_type)

        # compute errors
        res2 = 10 * np.log10(np.mean(np.sqrt(np.sum((w_new - np.mean(w_new, axis=0, keepdims=True)) ** 2, axis=-1))))
        res3 = 10 * np.log10(nmse(w_new))
        # save_w_conv.append(res1)
        save_w_avg.append(res2)
        save_nmse.append(res3)

        nzeros = w.size - np.count_nonzero(w)

        gamma_adapt = gamma_mul / (OTA_Comp_iter.ota_reps * np.max(np.sum(OTA_Comp_iter.receive_power_matrix, axis=0)))
        if i == 0:
            my_print(
                "(%s): %s, %s, %s, %s, %s, %s, %s, \t %s"
                % ("i", "b_i", "gm_val", "gm_adt", "w_mini", "w_maxi", "nzeros", "cons_e", "nmse")
            )
        my_print(
            "(%d): %.2f, %.2e, %.2e, %.5f, %.5f, %.2f, %.5f, \t %.4f"
            % (i, beta_i, OTA_Comp_iter.gamma_val, gamma_adapt, np.min(w), np.max(w), nzeros/w.size*100, res2, res3)
        )

        w = w_new

        save_nmse_data(res3)
        save_wavg_data(res2)

        # save current weights
        np.savetxt(w_file, w)

# Save data
nmse_data = np.asarray(save_nmse)
w_data = np.asarray(save_w_avg)
nzeros_data = np.asarray(save_num_zeros)
save_data_filename = results_dir + "/%s_results.npz" % (run_id)
os.makedirs(os.path.dirname(save_data_filename), exist_ok=True)
np.savez(save_data_filename, nmse=nmse_data, nzeros=nzeros_data, w_avg=w_data)
