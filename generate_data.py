import os
from source import *
import multiprocessing
from datetime import date


def do_simulation(V_applied, motor_path, save_path, t_end=1.0, test_name=None):
    dt = 1e-4

    save_path = save_path + '/IMMEC_history_'+ test_name + str(V_applied) + 'V_' + str(
        t_end) + 'sec'  # for conventional naming

    create_and_save_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, save_path=save_path,
                               V=V_applied, solving_tolerance=1e-5)
    return

if __name__ == "__main__":
    generate_multiple = False

    if generate_multiple:
        # first, save motor data
        cwd = os.getcwd()
        date = date.today().strftime("%m-%d")
        motor_path = os.path.join(cwd, 'Cantoni.pkl')
        save_path = os.path.join(cwd, 'data/', date)

        V_ranges = np.random.randint(40, 400, 20)  # randomly generated V values
        V_dict = {"V": V_ranges}

        # create directory if not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_motor_data(motor_path, save_path, extra_dict=V_dict)  # save motor data

        print("Starting simulation")
        p = multiprocessing.Pool(processes=10)
        input_data = [(V, motor_path, save_path) for V in V_ranges]
        p.starmap(do_simulation, input_data)
        print("Simulation finished")

    else: #if only one file should be generated
        cwd = os.getcwd()
        date = date.today().strftime("%m-%d")
        motor_path = os.path.join(cwd, 'Cantoni.pkl')
        save_path = os.path.join(cwd, 'data/', date)
        V = -200
        do_simulation(V, motor_path, save_path, t_end=1.0, test_name="torque") #test simulation with 2 seconds

