import os
import multiprocessing
from datetime import date
import joblib
from generate_data_source import *

if __name__ == "__main__":
    generate_traindata = False
    generate_testdata = True

    motor_path = os.path.join(os.getcwd(), "Cantoni.pkl")  # path to the motor file
    save_name = "dynamic_linear"
    t_end = 5.0

    ecc_value = 0.5
    ecc_dir = np.array([1, 0])  # direction of the eccentricity

    if (
        np.linalg.norm(ecc_dir) > 1e-10 or np.abs(ecc_value) > 1e-10
    ):  # avoid division by zero, if one of those is zero, the ecc is zero
        ecc = ecc_dir / np.linalg.norm(ecc_dir) * ecc_value
    else:
        print("No eccentricity")
        ecc = ecc_dir * ecc_value

    numbr_of_simulations = 50  # number of train simulations
    mode = "linear"

    ecc_random_direction = True  # if True, random direction of the eccentricity is set
    if ecc_random_direction:
        xvalue = np.random.random(numbr_of_simulations) * 2 - 1  # between() -1 and 1
        xvalue = ecc_value * xvalue  # scale with ecc
        yvalue = np.sqrt(ecc_value**2 - xvalue**2)  # ecc^2 = x^2 + y^2
        ecc_list = np.column_stack((xvalue, yvalue))
    else:
        ecc_list = np.repeat(ecc[np.newaxis, :], numbr_of_simulations, axis=0)

    if generate_traindata:
        print("Generating training data")
        save_path = os.path.join(
            os.getcwd(), "train-data/", date.today().strftime("%m-%d")
        )

        # create directory if not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        V_ranges = np.random.randint(
            40, 400, numbr_of_simulations
        )  # randomly generated V values
        load_ranges = 0 * np.random.randint(
            0.0, 370, numbr_of_simulations
        )  # the initial load is zero, but can be changed to random values

        save_simulation_data(
            motor_path, save_path, extra_dict={"V": V_ranges, "load": load_ranges}
        )  # save the motor data

        print("Starting simulation")
        input_data = [
            (V, motor_path, load_ranges[i], ecc_list[i], t_end, mode)
            for i, V in enumerate(V_ranges)
        ]

        n_jobs = 6  # number of cores to be used
        print("Using", n_jobs, "cores for simulation")
        # use joblib to parallelize the simulation, forces max_num_threads to 1
        with joblib.parallel_config(
            n_jobs=n_jobs, backend="loky", inner_max_num_threads=1
        ):
            # unpack tuples and run simulations
            output_list = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(do_simulation)(*data) for data in input_data
            )

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
                    "time": simulation[7],  # shape t, 1, simulations
                    "flux_st_yoke": simulation[8],
                    "gamma_rot": simulation[9],
                    "wcoe": simulation[10],
                }

            else:
                for i, key in enumerate(dataset.keys()):
                    # append to dataset for each simulation, along third dimension
                    dataset[key] = np.dstack((dataset[key], simulation[i]))
        print("Simulation finished")

        # save the dataset as .npz file
        title = "\\IMMEC_" + save_name + "_" + str(t_end) + "sec"
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
            wcoe=dataset["wcoe"],
        )

    elif generate_testdata:
        print("Generating test data")
        save_path = os.path.join(
            os.getcwd(), "test-data/", date.today().strftime("%m-%d")
        )
        # V = np.random.randint(40, 400)
        V = 300
        load = 0.0

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_simulation_data(
            motor_path, save_path, extra_dict={"V": V, "load": load}
        )  # save motor data

        print("Starting simulation")
        simulation = do_simulation(
            V, motor_path, t_end=t_end, ecc=ecc_list[0], load=load, mode=mode
        )
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
            "wcoe": simulation[10],
        }  # shape t, simulations

        print("Simulation finished")
        title = "\\IMMEC_" + save_name + "_" + str(t_end) + "sec"
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
            wcoe=dataset["wcoe"],
        )
