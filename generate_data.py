import os
import numpy as np
import pickle as pkl
import immec
import multiprocessing
from datetime import date
import tqdm


def do_simulation(V_applied, motor_path, load=3.7, ecc=np.zeros(2), t_end=1.0, mode="linear"):
    dt = 5e-5  # default

    datalogger = create_immec_data(
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
    start_load = 0.0
    end_load = load
    start_time = 0.0
    close_to_steady_state = False
    dt_load = 0.2  # first applied load for 1 second
    Vfmode = "chirp"
    print('Mode: ', Vfmode)

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
        # print("I think the problem is the .2 seconds, the motor cannot deal with the load, applied to fast")

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
        ecc = initial_ecc * motordict["d_air"]

        # I.D The inputs are concatenated to a single vector
        inputs = np.concatenate([v_uvw, [T_l], ecc])

        # II. Log the motor model values, time, and inputs
        data_logger.log(n * timestep, inputs)

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
        if n % int(0.1 / timestep) == 0:
            if mode == "nonlinear" and n * timestep < .5:
                # for the nonlinear case, somehow a load got applied, therefore i am forcing it to
                # only apply after 0.5 second
                close_to_steady_state = False
            else:
                close_to_steady_state = check_steady_state(
                    T_em=data_logger.quantities["T_em"],
                    speed=data_logger.quantities["omega_rot"],
                    nmbr_of_steps=int(0.15 / timestep),
                    mode=mode
                )

    return data_logger


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


def check_steady_state(T_em, speed, nmbr_of_steps, mode="linear"):
    """
    Checks whether steady state is close to reached, when T_em *and* omega are nearly constant
    True is returned when the last values of T_em and omega are whitin 5% marge of the mean of the last values

    if the mode is nonlinear, only check omega
    :param T_em: Electromagnetic torque
    :param speed: omega, rotation speed
    :param nmbr_of_steps: number of steps taken into account for the convergence
    :return: boolean
    """
    # steady state  is when T_em and speed is constant
    T_em = T_em[-nmbr_of_steps:]
    speed = speed[-nmbr_of_steps:]

    meanT = np.mean(T_em)
    meanS = np.mean(speed)
    if mode == "nonlinear":
        # only check the speed
        return np.all(np.abs(speed - meanS) < 0.05 * meanS)

    # all points should be within 5% of the mean
    if np.all(np.abs(T_em - meanT) < 0.05 * meanT) and np.all(np.abs(speed - meanS) < 0.05 * meanS):
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
