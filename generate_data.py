import os.path
from source import *
import multiprocessing
#todo: create data from V = 40 to V = 400 equidistant, check if outermost data is good and generate (maybe by multithreading)


def do_simulation(motor_path,save_path, V_applied, t_end= 1.0):
    dt = 1e-4


    save_path = save_path + '/IMMEC_history_' + str(V_applied) + 'V_' + str(
        t_end) + 'sec'  # for conventional naming

    create_and_save_immec_data(mode='linear', timestep=dt, t_end=t_end, path_to_motor=motor_path, save_path=save_path,
                               V=V_applied)
    load_path = save_path + '.pkl'
    return

if __name__ == "__main__":
    # first, save motor data
    motor_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/Cantoni.pkl'
    save_path = 'C:/Users/emmav/PycharmProjects/SINDY_project/data'
    if not os.path.exists(save_path+'/MOTORDATA.pkl'):
        save_motor_data(motor_path, save_path)

    do_simulation(motor_path, save_path, V_applied=400 ,t_end=1e-3)
    ''' # multiprocessing
    V_ranges = np.linspace(40,400,10)
    p = multiprocessing.Pool(processes=10)

    p.map(do_simulation, V_ranges)
    print("done")
    '''