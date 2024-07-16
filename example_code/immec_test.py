from tqdm import tqdm
from immec import *

motordict = read_motordict('../Cantoni.pkl')

timestep = 1e-4
stator_connection = 'wye'

motor_model = MotorModel(motordict, timestep, stator_connection, solver='newton')
tuner = RelaxationTuner()
data_logger = HistoryDataLogger(motor_model)

t_end = 3.0  # Total simulation time
steps_total = int(t_end // timestep)  # Total number of steps to simulate

#data_logger.pre_allocate(steps_total)

for n in tqdm(range(steps_total)):
    # I. Generate the input

    # I.A Load torque
    # The IMMEC function smooth_runup is called.
    # Here, it runs up to 3.7 Nm between 1.5 seconds and 1.7 seconds
    T_l = smooth_runup(3.7, n * timestep, 1.5, 1.7)

    # I.B Applied voltage
    # 400 V_RMS symmetrical line voltages are used
    v_uw = 400 * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep)
    v_vu = 400 * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 2 * np.pi / 3)
    v_wv = 400 * np.sqrt(2) * np.sin(2 * np.pi * 50 * n * timestep - 4 * np.pi / 3)
    v_line = np.array([v_uw, v_vu, v_wv])
    v_line = smooth_runup(v_line, n * timestep, 0.0, 1.5)

    # I.C Rotor eccentricity
    # In this demo, the rotor is placed in a centric position
    ecc = np.zeros(2)

    # I.D The inputs are concatenated to a single vector
    inputs = np.concatenate([v_line, [T_l], ecc])

    # II. Log the motor model values, time, and inputs

    data_logger.log(n * timestep, inputs)

    # III. Step the motor model
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

data_logger.postprocess()

data_logger.plot('all')
