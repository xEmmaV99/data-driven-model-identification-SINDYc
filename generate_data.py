import os
import multiprocessing
from datetime import date
from generate_data_source import *

if __name__ == "__main__":
    generate_traindata = False
    generate_testdata = True

    motor_path = os.path.join(os.getcwd(), "Cantoni.pkl")

    t_end = 5.0  # debug
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
        load_ranges = (0 * np.random.randint(0.0, 370, numbr_of_simulations))
        # INITIAL LOAD IS ZERO

        save_simulation_data(motor_path, save_path, extra_dict={"V": V_ranges, "load": load_ranges})  # save motor data

        print("Starting simulation")
        p = multiprocessing.Pool(processes=6)
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

        # load = 1 / 100 * np.random.randint(0.0, 370) * (V / 400.0)
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