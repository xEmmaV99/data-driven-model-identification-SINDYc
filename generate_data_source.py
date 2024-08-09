import numpy as np
import pickle as pkl
import immec
from tqdm import tqdm


def do_simulation(V_applied, motor_path, load=3.7, ecc=np.zeros(2), t_end=1.0, mode="linear"):
    dt = 1e-4  # default

    datalogger, wcoe = create_immec_data(
        mode=mode,
        timestep=dt,
        t_end=t_end,
        path_to_motor=motor_path,
        load=load,
        initial_ecc=ecc,
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
        wcoe[:,np.newaxis]
    ]


def save_simulation_data(motor_path: str, save_path: str, extra_dict: dict = None):
    """
    Saves useful simulation data in a dictionary to a .pkl file
    :param motor_path: path to motor data file
    :param save_path: path to save the file
    :param extra_dict: dict with extra things to save
    :return:
    """
    motordict = immec.read_motordict(motor_path)
    motor_model = immec.MotorModel(motordict, 5e-5, "wye", solver="newton")

    dictionary = {
        "stator_leakage_inductance": motor_model.stator_leakage_inductance,
        "N_abc_T": motor_model.N_abc_T,
        "R_st": motor_model.R_st,
    }

    for key in extra_dict.keys():  # add extra things
        dictionary[key] = extra_dict[key]

    with open(save_path + "/SIMULATION_DATA.pkl", "wb") as file:
        pkl.dump(dictionary, file)
    return


def create_immec_data(
        timestep: float,
        t_end: float,
        path_to_motor: str,
        V: float = 400,
        mode: str = "linear",
        solving_tolerance: float = 1e-4,
        load: float = 3.7,
        initial_ecc: np.array = np.zeros(2),
):
    """
    Creates data from the IMMEC model
    :param timestep: timestep of the simulation
    :param t_end: end of the simulation
    :param path_to_motor: path to the motor data file
    :param V: Maximum voltage applied to the motor
    :param mode: 'linear' for no nonlinear effects when simulating
    :param solving_tolerance:  tolerance used for the approximation of non linearities, only used when mode is not "linear"
    :param load: Load applied to the motor in Nm, at time 1.5s to 1.7s
    :param ecc: eccentricity of the motor, in percentage of the airgap
    :return: data_logger object (HistoryDataLogger) containing the data
    """
    # V should always be below 400, minimal V is 40 (means 5hz f)
    motordict = immec.read_motordict(path_to_motor)
    stator_connection = "wye"

    if mode == "linear":
        motor_model = immec.MotorModel(motordict, timestep, stator_connection)
    else:
        motor_model = immec.MotorModel(
            motordict, timestep, stator_connection, solver="newton", solving_tolerance=solving_tolerance
        )

    tuner = immec.RelaxationTuner()
    data_logger = immec.HistoryDataLogger(motor_model)

    steps_total = int(t_end // timestep)  # Total number of steps to simulate

    Vf_ratio = 400 / 50

    # data_logger.pre_allocate(steps_total)

    # initial params
    dynamic_ecc = False
    ecc_value = np.linalg.norm(initial_ecc)
    ecc_phi = np.arctan2(initial_ecc[1], initial_ecc[0])
    start_load = 0.0
    end_load = load
    start_time = 0.0
    close_to_steady_state = False
    dt_load = 0.2  # first applied load for 1 second
    Vfmode = "chirp"
    print('Mode: ', Vfmode)

    Wmagcoen = np.zeros(steps_total)
    for n in tqdm(range(steps_total)):
        # I. Generate the input

        # I.A Load torque

        # The IMMEC function smooth_runup is called.
        # Here, it runs up to 3.7 Nm (load) between 1.5 seconds and 1.7 seconds
        # T_l = smooth_runup(load, n * timestep, 1.5, 1.7)  # n*timestep is current time
        # Here, no torque is applied
        # T_l = 0
        if close_to_steady_state:
            # start_load = end_load #continuous
            start_load = change_load(start_load, end_load, n * timestep, start_time, start_time + dt_load)
            end_load = int(np.random.randint(0, 370) * (V / 400.0)) / 100  # choose new load
            print("New applied load: ", end_load, "Nm")
            start_time = n * timestep  # apply now
            close_to_steady_state = False  # change back
            dt_load = .2  # apply faster

        T_l = change_load(start_load, end_load, n * timestep, start_time, start_time + dt_load)

        # I.B Applied voltage
        # 400 V_RMS symmetrical line voltages are used
        if Vfmode == "constant_freq":
            V_amp = V
            f_amp = V / Vf_ratio * n * timestep
        elif Vfmode == "chirp_linear":
            V_amp = linear_runup(V, n * timestep, 1.5)
            f_amp = linear_runup_freq(V / Vf_ratio, n * timestep, 1.5)
        elif Vfmode == "chirp":
            V_amp = immec.smooth_runup(V, n * timestep, 0.0, 1.5)
            f_amp = chirp_freq(V / Vf_ratio, n * timestep, 1.5)

        v_u = V_amp * np.sqrt(2) * np.sin(2 * np.pi * f_amp)
        v_v = V_amp * np.sqrt(2) * np.sin(2 * np.pi * f_amp - 2 * np.pi / 3)
        v_w = V_amp * np.sqrt(2) * np.sin(2 * np.pi * f_amp - 4 * np.pi / 3)
        v_uvw = np.array([v_u, v_v, v_w])

        if Vfmode == "constant_freq":
            v_uvw = immec.smooth_runup(v_uvw, n * timestep, 0.0, 1.5)  # change amplitude of voltage

        # I.C Rotor eccentricity
        if not dynamic_ecc:
            ecc = initial_ecc * motordict["d_air"] # static eccentricity
        else:
            omega = motordict['omega_rot'][-1]
            ex = ecc_value * np.cos(omega * n*timestep + ecc_phi)
            ey = ecc_value * np.sin(omega * n*timestep + ecc_phi)
            ecc = np.array(ex, ey)

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_uvw, [T_l], ecc])

        # II. Log the motor model values, time, and inputs
        data_logger.log(n * timestep, inputs)

        # DEBUG: MAGNETIC COENERGY, after datalogger : might consider to skip first step

        if n != 0:
            psi_st = data_logger.quantities['potentials_st'][-1,:]
            psi_rot = data_logger.quantities['potentials_rot'][-1,:]

            Psi = np.broadcast_to(psi_st[:, np.newaxis], (psi_st.shape[0], motor_model.N_rot))\
                  - np.ones((motor_model.N_st, 1)) * psi_rot[np.newaxis, :]

            W = 0.5 * (Psi ** 2 * motor_model.P_air_hl_noskew()).sum(axis=1).sum()  # Total magnetic co energy
            Wmagcoen[n] = W

        # III. Step the motor model
        if mode == "linear":
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
                except immec.NoConvergenceException:
                    tuner.jump()

        # check if the steady state is reached, every .1 seconds
        if n % int(0.1 / timestep) == 0 and n*timestep > 0.5: #only check after .5 seconds
            close_to_steady_state = check_steady_state(
                    T_em=data_logger.quantities["T_em"],
                    speed=data_logger.quantities["omega_rot"],
                    nmbr_of_steps_per_chunk=int(0.03 / timestep),
                )

    return data_logger, Wmagcoen


""" # testing code #
x = np.arange(0,2,0.0001)
y=np.array([0,0,0])
Vf_ratio = 400/5 # should be 400/50 but this is more visible
for x_val in x:
    V_amp = smooth_runup(400, x_val, 0.0, 1.5)
    v_u = V_amp * np.sqrt(2) * np.sin(2 * np.pi * V_amp / Vf_ratio * x_val)
    v_v = V_amp * np.sqrt(2) * np.sin(2 * np.pi * V_amp / Vf_ratio * x_val - 2 * np.pi / 3)
    v_w = V_amp * np.sqrt(2) * np.sin(2 * np.pi * V_amp / Vf_ratio * x_val - 4 * np.pi / 3)
    v_uvw = np.array([v_u, v_v, v_w])
    y = np.vstack((y,v_uvw))

plt.plot(y)
plt.show()
"""


def linear_runup_freq(values, time: float, end_time: float, start_time: float = 0.0):
    """
    Increase the frequency, chirp wave, for a linear runup
    :param values: Maximum frequency
    :param time: Current timestep (often n*dt)
    :param end_time: Time at which `values` is reached
    :param start_time: Time to start the increase
    :return: The current value
    """
    f_0 = 0  # start from 0 frequency
    c = (values - f_0) / (end_time - start_time)
    if time < start_time:
        return values * time
    elif start_time <= time < end_time:
        # integrate the linear function from 0 to t
        return 0.5 * c * (time) ** 2 + f_0 * time
    else:
        phi_add = 0.5 * c * (end_time) ** 2 + f_0 * end_time
        # calculate the additional phaseshift
        return values * (time - end_time) + phi_add


def linear_runup(values, time: float, end_time: float, start_time: float = 0.0):
    """
    The linear runup for the voltages
    :param values: Maximum voltage to be reached
    :param time: current time, often n*dt
    :param end_time: end time of the runup
    :param start_time: start time of the runup
    :return: the current value of the voltage
    """
    if time < start_time:
        return np.zeros_like(values)
    elif start_time <= time < end_time:
        return values * (time - start_time) / (end_time - start_time)
    else:
        return values


def chirp_freq(values, time: float, end_time: float, start_time: float = 0.0):
    """
    Function to find the correct value of the frequency (chirpwave) for 1-cos runup
    :param values: Maximum frequency
    :param time:  current time, often n*dt
    :param end_time: end time of the runup
    :param start_time: start time of the runup
    :return: the current value of the frequency
    """
    duration = end_time - start_time
    if time < start_time:
        return values * time
    elif start_time <= time < end_time:
        # this is the integral of the 1-cos function
        return 1 / 2 * (time - np.sin(np.pi * time / duration) * duration / np.pi) * values

    else:
        # additional phaseshift
        phi_add = 1 / 2 * (end_time - np.sin(np.pi * end_time / duration) * duration / np.pi) * values
        return values * (time - end_time) + phi_add


def check_steady_state(T_em, speed, nmbr_of_steps_per_chunk, mode="linear"):
    """
    Checks whether steady state is close to reached, when the last 4 chunks (mean) are whitin 5% of the mean of these chunks

    :param T_em: Electromagnetic torque
    :param speed: omega, rotation speed
    :param nmbr_of_steps: number of steps taken into account for the convergence
    :return: boolean
    """
    # steady state  is when T_em and speed is constant
    meansT = np.array([])
    meansS = np.array([])
    for i in range(4):
        meansT = np.append(meansT, np.mean(T_em[-(i+1)*nmbr_of_steps_per_chunk: None if i==0 else -i*nmbr_of_steps_per_chunk]))
        meansS = np.append(meansS, np.mean(speed[-(i+1)*nmbr_of_steps_per_chunk: None if i==0 else -i*nmbr_of_steps_per_chunk]))
    meanT = np.mean(meansT)
    meanS = np.mean(meansS)
    # all points should be within 5% of the mean
    if np.all(np.abs(meansT - meanT) < 0.05 * meanT) and np.all(np.abs(meansS - meanS) < 0.05 * meanS):
        return True
    return False


def change_load(start_load, end_load, time: float, start_time: float, end_time: float):
    """
    Function used to change the load continuously with 1 - cos(t) type of run-up and run-down.
    :param start_load: The value of the initial load
    :param end_load: The value of the applied load
    :param time: The continuous (current) time value
    :param start_time: The time at which run-up starts
    :param end_time: The time at which run-up is completed
    :return: The values subjected to run-up
    """
    # Return start_load before the starting time
    if time < start_time:
        return start_load

    if start_load < end_load:
        # Use 1 - cos(t) run-up between the start time and the end time
        if time < end_time:
            duration = end_time - start_time
            return start_load + (end_load - start_load) * 0.5 * (1 - np.cos(np.pi / duration * (time - start_time)))
    elif start_load > end_load:
        # Use cos(t) run-down between the start time and the end time
        if time < end_time:
            duration = end_time - start_time
            return end_load + (start_load - end_load) * 0.5 * (1 + np.cos(np.pi / duration * (time - start_time)))
    # Return the value(s) after the end time
    return end_load


"""
x  = np.arange(0,10,0.01)
y = []
for x_val in x:
    y.append(change_load(0,10,x_val,0,10))
x2 = np.arange(10,20,0.01)
for x_val in x2:
    y.append(change_load(10,5,x_val,13,15))

plt.plot(y)
plt.show()
"""
