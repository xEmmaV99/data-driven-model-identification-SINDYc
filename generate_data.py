import os

import numpy as np

from source import *
import multiprocessing
from datetime import date


def do_old_simulation(V_applied, motor_path, save_path, t_end=1.0, test_name=""):
    dt = 5e-5
    save_path = (
        save_path + "/IMMEC_history_" + test_name + str(V_applied) + "V_" + str(t_end) + "sec"
    )  # for conventional naming

    create_and_save_immec_data(
        mode="linear",
        timestep=dt,
        t_end=t_end,
        path_to_motor=motor_path,
        save_path=save_path,
        V=V_applied,
        solving_tolerance=1e-5,
    )
    return


def do_simulation(V_applied, motor_path, load=3.7, ecc=np.zeros(2), t_end=1.0, mode="linear"):
    dt = 5e-5  # default

    datalogger = create_immec_data(
        mode=mode,
        timestep=dt,
        t_end=t_end,
        path_to_motor=motor_path,
        load=load,
        ecc=ecc,
        V=V_applied,
        solving_tolerance=1e-5,
    )

    return [
        datalogger.quantities["i_st"],
        datalogger.quantities["omega_rot"],
        datalogger.quantities["T_em"],
        datalogger.quantities["F_em"],
        datalogger.quantities["v_applied"],
        datalogger.quantities["T_l"],
        datalogger.quantities["ecc"],
        datalogger.quantities["time"],
        datalogger.quantities["flux_st_yoke"],
        datalogger.quantities["gamma_rot"],
    ]


if __name__ == "__main__":
    generate_traindata = False
    generate_testdata = True

    motor_path = os.path.join(os.getcwd(), "Cantoni.pkl")

    t_end = 1.6 #debug
    ecc = np.zeros(2)
    eccname = "0"
    numbr_of_simulations = 50  # number of train simulations (of 5sec)
    mode = 'nonlinear'

    if generate_traindata:
        print("Generating training data")
        save_path = os.path.join(os.getcwd(), "train-data/", date.today().strftime("%m-%d"))
        # create directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        V_ranges = np.random.randint(40, 400, numbr_of_simulations)  # randomly generated V values
        load_ranges = (
            1 / 100 * np.random.randint(0.0, 370, numbr_of_simulations) * (V_ranges / 400.0)
        )  # randomly generated (initial) load values, scaled according to V

        save_simulation_data(motor_path, save_path, extra_dict={"V": V_ranges, "load": load_ranges})  # save motor data
        """
        dict_keys(
            ['potentials_st', 'potentials_rot', 'flux_st_tooth', 'flux_st_yoke', 'flux_rot_tooth', 'flux_rot_yoke',
             'i_st', 'i_rot', 'alpha_rot', 'omega_rot', 'gamma_rot', 'T_em', 'F_em', 'v_applied', 'T_l', 'ecc',
             'iterations', 'time'])
        """

        print("Starting simulation")
        p = multiprocessing.Pool(processes=10)
        input_data = [(V, motor_path, load_ranges[i], ecc, t_end, mode) for i, V in enumerate(V_ranges)]
        output_list = p.starmap(do_simulation, input_data)
        p.close()
        p.join()

        # if dataset library not initialised, create it, else append to it
        dataset = None
        for simulation in output_list:
            if dataset is None:
                # initialise dataset dictionary
                dataset = {
                    "i_st": simulation[0],  # shape t, 3, simulations
                    "omega_rot": simulation[1],  # shape t, simulations
                    "T_em": simulation[2],  # shape t, simulations
                    "F_em": simulation[3],  # shape t, 2, simulations
                    "v_applied": simulation[4],  # shape t, 3, simulations
                    "T_l": simulation[5],  # shape t, simulations
                    "ecc": simulation[6],  # shape t, 2, simulations
                    "time": simulation[7],
                    "flux_st_yoke": simulation[8],
                    "gamma_rot": simulation[9],
                }  # shape t, simulations

            else:
                for i, key in enumerate(dataset.keys()):
                    # append to dataset for each simulation, along third dimension
                    dataset[key] = np.dstack((dataset[key], simulation[i]))

        print("Simulation finished")
        title = "\\IMMEC_" + eccname + "ecc_" + str(t_end) + "sec"
        np.savez_compressed(
            save_path + title + ".npz",
            i_st=dataset["i_st"],
            omega_rot=dataset["omega_rot"],
            T_em=dataset["T_em"],
            F_em=dataset["F_em"],
            v_applied=dataset["v_applied"],
            T_l=dataset["T_l"],
            ecc=dataset["ecc"],
            time=dataset["time"],
            flux_st_yoke=dataset["flux_st_yoke"],
            gamma_rot=dataset["gamma_rot"],
        )

    elif generate_testdata:
        print("Generating test data")
        save_path = os.path.join(os.getcwd(), "test-data/", date.today().strftime("%m-%d"))
        # choose V and (initial) load somewhere between 0.0 and 3.7, but scale with V (so 3.7 is for V=400)
        V = np.random.randint(40, 400)

        #load = 1 / 100 * np.random.randint(0.0, 370) * (V / 400.0)
        load = 0.0

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_simulation_data(motor_path, save_path, extra_dict={"V": V, "load": load})  # save motor data

        print("Starting simulation")
        simulation = do_simulation(V, motor_path, t_end=t_end, ecc=ecc, load=load, mode=mode)
        dataset = {
            "i_st": simulation[0],  # shape t, 3, simulations
            "omega_rot": simulation[1],  # shape t, simulations
            "T_em": simulation[2],  # shape t, simulations
            "F_em": simulation[3],  # shape t, 2, simulations
            "v_applied": simulation[4],  # shape t, 3, simulations
            "T_l": simulation[5],  # shape t, simulations
            "ecc": simulation[6],  # shape t, 2, simulations
            "time": simulation[7],
            "flux_st_yoke": simulation[8],
            "gamma_rot": simulation[9],
        }  # shape t, simulations

        print("Simulation finished")
        title = "\\IMMEC_" + eccname + "ecc_" + str(t_end) + "sec"
        np.savez_compressed(
            save_path + title + ".npz",
            i_st=dataset["i_st"],
            omega_rot=dataset["omega_rot"],
            T_em=dataset["T_em"],
            F_em=dataset["F_em"],
            v_applied=dataset["v_applied"],
            T_l=dataset["T_l"],
            ecc=dataset["ecc"],
            time=dataset["time"],
            flux_st_yoke=dataset["flux_st_yoke"],
            gamma_rot=dataset["gamma_rot"],
        )
