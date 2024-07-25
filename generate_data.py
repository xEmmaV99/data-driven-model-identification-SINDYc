import os

import numpy as np

from source import *
import multiprocessing
from datetime import date


def do_simulation(V_applied, motor_path, save_path, t_end=1.0, test_name=""):
    dt = 5e-5
    save_path = save_path + '/IMMEC_history_' + test_name + str(V_applied) + 'V_' + str(
        t_end) + 'sec'  # for conventional naming

    create_and_save_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, save_path=save_path,
                               V=V_applied, solving_tolerance=1e-5)
    return


def do_merged_simulation(V_applied, motor_path, save_path, load=3.7, ecc=0.0, t_end=1.0):
    dt = 5e-5  # default

    datalogger = create_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, load=load, ecc=ecc,
                                   V=V_applied, solving_tolerance=1e-5)

    return [datalogger.quantities['i_st'], datalogger.quantities['omega_rot'], datalogger.quantities['T_em'],
            datalogger.quantities['F_em'], datalogger.quantities['v_applied'], datalogger.quantities['T_l'],
            datalogger.quantities['ecc']]


if __name__ == "__main__":
    generate_traindata = True
    generate_testdata = False

    save_path = os.path.join(os.getcwd(), 'train-data/', date.today().strftime("%m-%d"))
    motor_path = os.path.join(os.getcwd(), 'Cantoni.pkl')

    t_end = 1.0
    ecc = np.zeros(2)
    eccname = '0'
    numbr_of_simulations = 5

    # create directory if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if generate_traindata:
        # to do: save V-ranges and load-ranges in SIMULATION_DATA.pkl as a dictionary.
        V_ranges = np.random.randint(40, 400, numbr_of_simulations)  # randomly generated V values
        load_ranges = 1 / 100 * np.random.randint(0.0, 370, numbr_of_simulations) * (
                    V_ranges / 400.0)  # randomly generated load values, scaled according to V

        save_motor_data(motor_path, save_path, extra_dict={"V": V_ranges,"load": load_ranges})  # save motor data
        '''
        dict_keys(
            ['potentials_st', 'potentials_rot', 'flux_st_tooth', 'flux_st_yoke', 'flux_rot_tooth', 'flux_rot_yoke',
             'i_st', 'i_rot', 'alpha_rot', 'omega_rot', 'gamma_rot', 'T_em', 'F_em', 'v_applied', 'T_l', 'ecc',
             'iterations', 'time'])
        '''
        print("Starting simulation")

        p = multiprocessing.Pool(processes=5)
        input_data = [(V, motor_path, save_path, load_ranges[i], ecc, t_end) for i, V in enumerate(V_ranges)]
        output_list = p.starmap(do_merged_simulation, input_data)
        p.close()
        p.join()
        print(len(output_list))

        # if dataset library not initialised, create it, else append to it
        for simulation in output_list:
            if not 'dataset' in locals():
                # initialise dataset dictionnary
                dataset = {'i_st': simulation[0],  # shape t, 3, simulations
                           'omega_rot': simulation[1],  # shape t, simulations
                           'T_em': simulation[2],  # shape t, simulations
                           'F_em': simulation[3],  # shape t, 2, simulations
                           'v_applied': simulation[4],  # shape t, 3, simulations
                           'T_l': simulation[5],  # shape t, simulations
                           'ecc': simulation[6]}  # shape t, 2, simulations
            else:
                for i, k in enumerate(dataset.keys()):
                    # append to dataset for each simulation, perhaps hstack ?
                    dataset[k] = np.stack((dataset[k], output_list[i]), axis=-1)

        print("Simulation finished")
        title = "\\IMMEC_" + eccname + 'ecc_' + str(t_end) + 'sec'
        np.savez_compressed(save_path + title + '.npz', i_st=dataset['i_st'],
                            omega_rot=dataset['omega_rot'], T_em=dataset['T_em'], F_em=dataset['F_em'],
                            v_applied=dataset['v_applied'], T_l=dataset['T_l'], ecc=dataset['ecc'])

    elif generate_testdata:
        # choose V and load somewhere between 0.0 and 3.7, but scale with V (so 3.7 is for V=400)
        V = np.random.randint(40, 400, 1)
        load = 1 / 100 * np.random.randint(0.0, 370, 1) * (V / 400.0)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # do_simulation(V, motor_path, save_path, t_end=1.0)
        do_merged_simulation(V, motor_path, save_path, t_end=1.0)
