from tqdm import tqdm
from immec import *
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps

def reference_abc_to_dq0(coord_array):
    # coord_array is a Nx3 array with N the number of samples

    #align a to d axis and project b and c axis on q
    T = 3/2*np.array([[1,-0.5,-0.5],
                     [0,np.sqrt(3)/2,-np.sqrt(3)/2],
                     [0.5,0.5,0.5]])

    return np.dot(T,coord_array.T).T
    # ugly selection but otherwise it has size (N,) which is not desired for hstack


'''
def generate_MCC_data(mode, timestep, t_end, path_to_motor, save_path=""):
    motordict = read_motordict(path_to_motor)
    stator_connection = 'directPhase'

    motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton')
    tuner = RelaxationTuner()
    data_logger = HistoryDataLogger(motor_model)

    steps_total = int(t_end // timestep)  # Total number of steps to simulate

    # data_logger.pre_allocate(steps_total)

    for n in tqdm(range(steps_total)):
        # I. Generate the input

        # I.A Load torque
        # The IMMEC function smooth_runup is called.
        # Here, it runs up to 3.7 Nm between 1.5 seconds and 1.7 seconds
        T_l = smooth_runup(3.7, n * timestep, 1.5, 1.7)

        # I.B Applied voltage
        # 400 V_RMS symmetrical line voltages are used
        v_a = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep)
        v_b = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 2 * np.pi / 3)
        v_c = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 4 * np.pi / 3)
        v_abc = np.array([v_a, v_b, v_c])
        print("no smooth runup")
        #v_abc = smooth_runup(v_abc, n * timestep, 0.0, 1.5)

        # I.C Rotor eccentricity
        # In this demo, the rotor is placed in a centric position
        ecc = np.zeros(2)

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_abc, [T_l], ecc])

        # II. Log the motor model values, time, and inputs

        data_logger.log(n * timestep, inputs)

        # III. Step the motor model
        if mode == 'linear':
            motor_model.step(inputs)
        else:
            # A step is initialised as unsolved
            tuner.solved = False

            while not tuner.solved:
                try:
                    # Apply the relaxation factor of the tuner to the motor model
                    motor_model.relaxation_factor = tuner.relaxation
                    # Attempt to step the motor model
                    motor_model.step(inputs)
                    # When succesful, increase the tuner relaxation factor
                    tuner.step()
                # When unsuccesful, decrease the tuner relaxation factor
                except NoConvergenceException:
                    tuner.jump()

        if save_path != "":
            data_logger.save_history(save_path)
    return data_logger
'''


def get_immec_training_data(check_training_data=False, path_to_data_logger="IMMEC_history_unnamed", timestep = 1e-4):
    with open(path_to_data_logger, 'rb') as file:
        data_logger = pkl.load(file)

    # check if data_logger is fine
    if check_training_data:
        # data_logger.postprocess()
        plt.plot(data_logger['i_st'])
        plt.show()

    # get x train data

    x_train = reference_abc_to_dq0(data_logger['i_st'])

    # get u data: potentials_st, i_st, omega_rot, gamma_rot, and the intergals.
    I = np.cumsum(x_train,
                  0) * timestep  # those are caluculated using forward euler. Note that x_train*timestep instead of cumsum * timestep yield same results
    V = np.cumsum(data_logger['v_applied'], 0) * timestep

    u_data = np.hstack((data_logger['v_applied'], I, V, data_logger['gamma_rot']%(2*np.pi),
                        data_logger['omega_rot']))
    return x_train, u_data


def create_and_save_immec_data(timestep, t_end, path_to_motor, save_path, mode = 'linear'):
    motordict = read_motordict(path_to_motor)
    stator_connection = 'directPhase'

    motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton')
    tuner = RelaxationTuner()
    data_logger = HistoryDataLogger(motor_model)

    steps_total = int(t_end // timestep)  # Total number of steps to simulate

    # data_logger.pre_allocate(steps_total)

    for n in tqdm(range(steps_total)):
        # I. Generate the input

        # I.A Load torque
        # The IMMEC function smooth_runup is called.
        # Here, it runs up to 3.7 Nm between 1.5 seconds and 1.7 seconds
        T_l = smooth_runup(3.7, n * timestep, 1.5, 1.7)

        # I.B Applied voltage
        # 400 V_RMS symmetrical line voltages are used
        v_a = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep)
        v_b = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 2 * np.pi / 3)
        v_c = 400 / np.sqrt(3) * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 4 * np.pi / 3)
        v_abc = np.array([v_a, v_b, v_c])

        v_abc = smooth_runup(v_abc, n * timestep, 0.0, 1.5)

        # I.C Rotor eccentricity
        # In this demo, the rotor is placed in a centric position
        ecc = np.zeros(2)

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_abc, [T_l], ecc])

        # II. Log the motor model values, time, and inputs

        data_logger.log(n * timestep, inputs)

        # III. Step the motor model
        if mode == 'linear':
            motor_model.step(inputs)
        else:
            # A step is initialised as unsolved
            tuner.solved = False

            while not tuner.solved:
                try:
                    # Apply the relaxation factor of the tuner to the motor model
                    motor_model.relaxation_factor = tuner.relaxation
                    # Attempt to step the motor model
                    motor_model.step(inputs)
                    # When succesful, increase the tuner relaxation factor
                    tuner.step()
                # When unsuccesful, decrease the tuner relaxation factor
                except NoConvergenceException:
                    tuner.jump()

        data_logger.save_history(save_path)
    return





if __name__ == '__main__' :
    def f(t,offset): return np.sin(2 * np.pi * t + offset)

    # test reference_abc_to_dq0 : for a balanced system
    array = np.array([[f(0,0),f(0,2/3*np.pi),f(0,4/3*np.pi)],
                      [f(0.2,0),f(0.2,2/3*np.pi),f(0.2,4/3*np.pi)]])
    print(reference_abc_to_dq0(array))