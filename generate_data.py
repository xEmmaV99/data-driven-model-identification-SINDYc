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


def do_merged_simulation(V_applied, motor_path, save_path, t_end=2e-4):
    dt = 5e-5
    save_path = save_path + '/IMMEC_history_' + str(V_applied) + 'V_' + str(t_end) + 'sec'  # for conventional naming
    datalogger = create_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path,
                                   save_path=save_path,
                                   V=V_applied, solving_tolerance=1e-5)

    return [datalogger.quantities['i_st'], datalogger.quantities['omega_rot'], datalogger.quantities['T_em'],
            datalogger.quantities['F_em'], \
            datalogger.quantities['v_applied'], datalogger.quantities['T_l'], datalogger.quantities['ecc']]


if __name__ == "__main__":
    generate_traindata = True
    generate_testdata = False

    save_path = os.path.join(os.getcwd(), 'data/', date.today().strftime("%m-%d"))
    motor_path = os.path.join(os.getcwd(), 'Cantoni.pkl')

    # create directory if not exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if generate_traindata:
        V_ranges = np.random.randint(40, 400, 1)  # randomly generated V values
        V_dict = {"V": V_ranges}
        save_motor_data(motor_path, save_path, extra_dict=V_dict)  # save motor data
        '''
        dict_keys(
            ['potentials_st', 'potentials_rot', 'flux_st_tooth', 'flux_st_yoke', 'flux_rot_tooth', 'flux_rot_yoke',
             'i_st', 'i_rot', 'alpha_rot', 'omega_rot', 'gamma_rot', 'T_em', 'F_em', 'v_applied', 'T_l', 'ecc',
             'iterations', 'time'])
        '''

        print("Starting simulation")
        debug = True
        if not debug:
            p = multiprocessing.Pool(processes=2)
            input_data = [(V, motor_path, save_path) for V in V_ranges]
            output_list = p.starmap(do_merged_simulation, input_data)
        else:
            output_list = []
            V = V_ranges[0]
            output_list = do_merged_simulation(V, motor_path, save_path)

        # if dataset library not initialised, create it, else append to it
        if not 'dataset' in locals():
            # initialise dataset dictionnary
            dataset = {'i_st': output_list[0],  # shape t, 3, simulations
                       'omega_rot': output_list[1],  # shape t, simulations
                       'T_em': output_list[2],  # shape t, simulations
                       'F_em': output_list[3],  # shape t, 2, simulations
                       'v_applied': output_list[4],  # shape t, 3, simulations
                       'T_l': output_list[5],  # shape t, simulations
                       'ecc': output_list[6]}  # shape t, 2, simulations
        else:
            for i, k in enumerate(dataset.keys()):
                # append to dataset for each simulation, perhaps hstack ?
                dataset[k] = np.stack((dataset[k], output_list[i]), axis=-1)

        print("Simulation finished")

        np.savez_compressed(save_path + '.npz', i_st=dataset['i_st'],
                            omega_rot=dataset['omega_rot'], T_em=dataset['T_em'], F_em=dataset['F_em'],
                            v_applied=dataset['v_applied'], T_l=dataset['T_l'], ecc=dataset['ecc'])

    elif generate_testdata:
        # choose V and load somewhere between 0.0 and 3.7, but scale with V (so 3.7 is for V=400)
        V = np.random(40, 400, 1)
        load = np.random(0.0, 3.7, 1) * (V / 400.0)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # do_simulation(V, motor_path, save_path, t_end=1.0)
        do_merged_simulation(V, motor_path, save_path, t_end=1.0)
