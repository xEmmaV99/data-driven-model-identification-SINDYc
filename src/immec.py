import numba as nb
import numpy as np
from numba.experimental import jitclass
import pickle as pkl
import matplotlib.pyplot as plt


class NoConvergenceException(Exception):
    """
    Custom exception which indicates that the nonlinear solver has not reached convergence
    """
    pass


def smooth_runup(values, time: float, start_time: float, end_time: float):
    """
    Function used to run up an applied quantity using a 1 - cos(t) type of run-up.
    :param values: The values subjected to run-up
    :param time: The continuous time value
    :param start_time: The time at which run-up starts
    :param end_time: The time at which run-up is completed
    :return: The values subjected to run-up
    """
    # Return zero(s) before the starting time
    if time < start_time:
        return np.zeros_like(values)

    # Use 1 - cos(t) run-up between the start time and the end time
    elif start_time <= time <= end_time:
        duration = end_time - start_time
        return values * 0.5 * (1 - np.cos(np.pi / duration * (time - start_time)))

    # Return the value(s) after the end time
    else:
        return values


def read_motordict(path_to_dict: str):
    """
    Read and convert a pickled motor dictionary to a typed Numba dictionary. String keys and float64 values are assumed.
    """
    # Read the motor dictionary pickle file
    with open(path_to_dict, 'rb') as file:
        motordict = pkl.load(file)

    # Create an empty typed dictionary
    typed_dict = nb.typed.Dict.empty(key_type=nb.types.string, value_type=nb.types.float64)

    # Populate the typed dictionary
    for key, value in motordict.items():
        unicode_key = str(key)
        typed_dict[unicode_key] = value

    return typed_dict


@nb.njit(nb.float64[:, :](nb.int_, nb.int_), fastmath=True, cache=True)
def z(x: nb.int_, y: nb.int_) -> nb.float64[:, :]:
    """
    Easier creation of zero-matrices
    """
    return np.zeros((x, y))


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int_, nb.int_), fastmath=True, cache=True)
def roll_matrix(arr: nb.float64[:, :], roll: nb.int_, axis: nb.int_) -> nb.float64[:, :]:
    """
    Function that rolls the elements of the array by 'roll' along the specified axis.
    This replaces the np.roll function, which is not supported by Numba.
    """
    # Limit the array roll to the dimension of the rolled axis
    roll = roll % arr.shape[axis]

    # Copy an array which is not rolled
    if roll == 0:
        return arr.copy()

    # Perform either a row or a column roll
    if axis == 0:
        # Row roll
        return np.concatenate((arr[-roll:], arr[:-roll]), axis=0)
    elif axis == 1:
        # Column roll
        return np.concatenate((arr[:, -roll:], arr[:, :-roll]), axis=1)
    else:
        raise ValueError("Axis should be 0 or 1")


@nb.njit(nb.float64[:, :](nb.int_, nb.int_), fastmath=True, cache=True)
def rolled_identity(size: nb.int_, roll: nb.int_) -> nb.float64[:, :]:
    """
    Easier creation of rolled identity matrix as a substitute for Kronecker delta matrices.
    np.roll is not supported by Numba; a custom function is used instead.
    :return: A (size x size) matrix which is rolled by 'roll' along its first dimension.
    """
    return roll_matrix(np.eye(size), roll, 0)


@nb.njit(fastmath=True, cache=True)
def clarke(quantity: nb.float64[:, :], mode=0):
    """
    Clarke transformation using power-invariant approach. Transforms quantities in the abc frame to the alpha-beta frame
    :param quantity: Size three 1D array in abc frame
    :param mode: Single or series mode transformation
    :return: Size two 1D array in alpha-beta frame
    """
    # Clarke transformation matrix
    K_C = np.array([[1, -0.5, -0.5],
                    [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]])
    K_C = np.sqrt(2 / 3) * K_C

    # single-value transformation
    if mode == 0:
        return np.ascontiguousarray(K_C) @ np.ascontiguousarray(quantity)
    # series of values transformation
    elif mode == 1:
        return np.ascontiguousarray(quantity) @ np.ascontiguousarray(K_C.transpose())


def voltage_source_inverter(switching_state: int, V_dc: float):
    """
    Simple model of a voltage source inverter (VSI) with the top branch potential connected to V_dc/2 and the bottom
    branch potential connected to -V_dc/2. The switching states correspond to:
    {1 : [0 0 0], 2 : [0 0 1], 3 : [0 1 0], 4 : [0 1 1], 5 : [1 0 0], 6 : [1 0 1], 7 : [1 1 0], 8 : [1 1 1]}, where 1
    represents a closed upper switch and an opened lower switch.
    :param switching_state: Integer-valued switching state, ranged 1 - 8
    :param V_dc: Total DC source voltage of VSI
    :return: the line voltages defined as [v_u - v_w, v_v - v_u, v_w - v_v]
    """
    # Map the switching state to the UVW terminal potentials
    if switching_state == 1:
        v_uvw = np.array([-V_dc / 2, -V_dc / 2, -V_dc / 2])
    elif switching_state == 2:
        v_uvw = np.array([-V_dc / 2, -V_dc / 2, V_dc / 2])
    elif switching_state == 3:
        v_uvw = np.array([-V_dc / 2, V_dc / 2, -V_dc / 2])
    elif switching_state == 4:
        v_uvw = np.array([-V_dc / 2, V_dc / 2, V_dc / 2])
    elif switching_state == 5:
        v_uvw = np.array([V_dc / 2, -V_dc / 2, -V_dc / 2])
    elif switching_state == 6:
        v_uvw = np.array([V_dc / 2, -V_dc / 2, V_dc / 2])
    elif switching_state == 7:
        v_uvw = np.array([V_dc / 2, V_dc / 2, -V_dc / 2])
    elif switching_state == 8:
        v_uvw = np.array([V_dc / 2, V_dc / 2, V_dc / 2])
    else:
        raise ValueError('Invalid switching state provided to voltage_source_inverter')

    # Calculate the line voltages
    v_line = np.array([v_uvw[0] - v_uvw[2], v_uvw[1] - v_uvw[0], v_uvw[2] - v_uvw[1]])
    return v_line


# Numba type specifications
spec = [
    ('timestep', nb.float64),
    ('connectionType', nb.types.string),
    ('phase_resistance_factor', nb.int64[:]),
    ('solving_tolerance', nb.float64),
    ('solver', nb.types.string),
    ('linearsolver', nb.types.string),
    ('relaxation_factor', nb.float64),
    ('N_rot', nb.int64),
    ('N_st', nb.int64),
    ('r_st_out', nb.float64),
    ('r_st_in', nb.float64),
    ('l_z', nb.float64),
    ('l_st_tooth', nb.float64),
    ('l_st_head', nb.float64),
    ('d_air', nb.float64),
    ('r_rot_in', nb.float64),
    ('l_rot_head', nb.float64),
    ('l_rot_tooth', nb.float64),
    ('w_rot_er', nb.float64),
    ('alpha_sk', nb.float64),
    ('w_st_tooth', nb.float64),
    ('w_st_head', nb.float64),
    ('w_rot_tooth', nb.float64),
    ('w_rot_head', nb.float64),
    ('er_resistanceCalculationMethod', nb.float64),
    ('er_innerRadius', nb.float64),
    ('er_outerRadius', nb.float64),
    ('rho_rot', nb.float64),
    ('mu_fe', nb.float64),
    ('p', nb.int64),
    ('D', nb.float64),
    ('J', nb.float64),
    ('N_w', nb.float64),
    ('R_st_i', nb.float64),
    ('CoreMaterial', nb.float64),
    ('r_st_tooth_in', nb.float64),
    ('r_st_tooth_out', nb.float64),
    ('S_st_head', nb.float64),
    ('l_st_tooth_toNode', nb.float64),
    ('r_rot_out', nb.float64),
    ('r_rot_tooth_out', nb.float64),
    ('r_rot_tooth_in', nb.float64),
    ('S_rot_head', nb.float64),
    ('r_bdn', nb.float64),
    ('Kdelta1', nb.float64[:, :]),
    ('Kdelta2', nb.float64[:, :]),
    ('N_abc', nb.float64[:, :]),
    ('R_st', nb.float64[:, :]),
    ('SubB_Nrot', nb.float64[:, :]),
    ('R_rot', nb.float64[:, :]),
    ('SubF_Nst', nb.float64[:, :]),
    ('I_Nst', nb.float64[:, :]),
    ('SubB_Nst', nb.float64[:, :]),
    ('SubF_Nrot', nb.float64[:, :]),
    ('I_Nrot', nb.float64[:, :]),
    ('Set_1Nst', nb.float64[:, :]),
    ('R_st_t', nb.float64[:, :]),
    ('R_st_y', nb.float64[:, :]),
    ('R_rot_t', nb.float64[:, :]),
    ('R_rot_y', nb.float64[:, :]),
    ('Kdelta7', nb.float64[:, :]),
    ('N_abc_T', nb.float64[:, :]),
    ('system_matrix_static', nb.float64[:, :]),
    ('gamma_hl', nb.float64[:, :]),
    ('gamma_hl0', nb.float64[:, :]),
    ('Kdelta3', nb.float64[:, :]),
    ('Kdelta4', nb.float64[:, :]),
    ('Kdelta5', nb.float64[:, :]),
    ('Kdelta6', nb.float64[:, :]),
    ('e_h', nb.float64[:, :]),
    ('mu_0', nb.float64),
    ('r_rot_out', nb.float64),
    ('theta_st', nb.float64),
    ('theta_rot', nb.float64),
    ('K_0', nb.float64),
    ('K_1', nb.float64),
    ('A_st_y', nb.float64),
    ('A_st_t', nb.float64),
    ('A_rot_y', nb.float64),
    ('A_rot_t', nb.float64),
    ('a_1', nb.float64),
    ('state', nb.float64[:]),
    ('skew_method', nb.types.string),
    ('torque_method', nb.types.string),
    ('d_ecc', nb.float64[:]),
    ('r_ecc', nb.float64[:]),
    ('r_ecc_m', nb.float64[:, :]),
    ('K_mu1', nb.float64),
    ('K_mu2', nb.float64),
    ('K_mu3', nb.float64),
    ('theta_st_y', nb.float64),
    ('phi_st_t', nb.float64[:]),
    ('state_rotational', nb.float64[:]),
    ('iterations', nb.int64),
    ('iterations_threshold', nb.int64),
    ('alpha_sk_acc', nb.float64),
    ('c_perm_shape', nb.float64),
    ('c_perm_sat', nb.float64),
    ('nism_slices', nb.int64),
    ('B_data', nb.float64[:]),
    ('mu_data', nb.float64[:]),
    ('mu_data_deriv', nb.float64[:]),
    ('core_fromdata', nb.bool_),
    ('l_rot_y', nb.float64),
    ('Kdelta8', nb.float64[:, :]),
    ('stator_leakage_inductance', nb.float64)]


@jitclass(spec) # dit moet aanliggen oeps DEBUG
class MotorModel:
    """
    Main induction motor model class object
    """
    def __init__(self, motordict: nb.typed.typeddict, _timestep: float, _connectionType: str,
                 broken_rotor_bars: tuple = (0, 0.0), phase_resistance_factor: np.ndarray = None,
                 solving_tolerance: float = 1e-4, solver: str = 'linear', skew_method: str = 'AISM',
                 linearsolver: str = 'algebraic', torque_method: str = 'nodal', relaxation_factor: float = 0.5,
                 iterations_threshold: int = 500, nism_slices=10, core_material: np.ndarray = None):
        """
        :param motordict: The motor parameters in Numba typed dictionary format
        :param _timestep: The time stepping value
        :param _connectionType: The stator electrical connection type
        :param broken_rotor_bars: A tuple of the number of broken rotor bars and their resistance multiplication factor
        :param phase_resistance_factor: The stator resistance multiplication factors as a size-three Numpy array
        :param solving_tolerance: The nonlinear solving tolerance
        :param solver: The nonlinear or linear solver
        :param skew_method: The rotor skew calculation method
        :param linearsolver: The linear solver
        :param torque_method: The torque calculation method
        :param relaxation_factor: The nonlinear solution relaxation factor
        """
        # Attributes which should not be changed over time
        self.timestep = _timestep
        self.connectionType = _connectionType

        # Attributes which may be modified over time
        self.solving_tolerance = solving_tolerance
        self.solver = solver
        self.skew_method = skew_method
        self.linearsolver = linearsolver
        self.torque_method = torque_method
        self.relaxation_factor = relaxation_factor
        self.iterations_threshold = iterations_threshold
        self.nism_slices = nism_slices

        # Initialise the nonlinear iterations per step
        self.iterations = 0

        # Generate static elements
        if phase_resistance_factor is None:
            self.phase_resistance_factor = np.array([1, 1, 1])

        """Read and store the motor dictionary"""

        self.N_rot = int(motordict['N_rot'])  # Number of rotor teeth
        self.N_st = int(motordict['N_st'])  # Number of stator teeth
        self.r_st_out = motordict['r_st_out']  # Stator outer radius
        self.r_st_in = motordict['r_st_in']  # Stator inner radius
        self.l_z = motordict['l_z']  # Motor depth (= in axial direction)
        self.l_st_tooth = motordict['l_st_tooth']  # Stator tooth base length
        self.l_st_head = motordict['l_st_head']  # Stator tooth head length
        self.d_air = motordict['d_air']  # Nominal air gap radial length
        self.r_rot_in = motordict['r_rot_in']  # Rotor inner radius
        self.l_rot_head = motordict['l_rot_head']  # Rotor tooth head length
        self.l_rot_tooth = motordict['l_rot_tooth']  # Rotor tooth base length
        self.w_rot_er = motordict['w_rot_er']  # End ring thickness
        self.alpha_sk = motordict['alpha_sk']  # Rotor bar skew angle
        self.w_st_tooth = motordict['w_st_tooth']  # Stator tooth base width
        self.w_st_head = motordict['w_st_head']  # Stator tooth head width
        self.w_rot_tooth = motordict['w_rot_tooth']  # Rotor tooth base width
        self.w_rot_head = motordict['w_rot_head']  # Rotor tooth head width
        self.er_resistanceCalculationMethod = motordict['er_resistanceCalculationMethod']
        if self.er_resistanceCalculationMethod == 0.0:
            self.er_innerRadius = motordict['er_innerRadius']  # End ring inner radius
            self.er_outerRadius = motordict['er_outerRadius']  # End ring outer radius
        self.rho_rot = motordict['rho_rot']  # Rotor bar electrical resistivity
        self.mu_fe = motordict['mu_fe']  # Linearised permeability
        self.p = int(motordict['p'])  # Pole pair number
        self.D = motordict['D']
        self.J = motordict['J']
        self.N_w = motordict['N_w']  # Windings per slot
        self.R_st_i = motordict['R_st_i']  # Stator phase resistance
        self.CoreMaterial = motordict['CoreMaterial']  # Core material
        self.c_perm_shape = motordict['c_perm_shape']
        self.c_perm_sat = motordict['c_perm_sat']
        self.stator_leakage_inductance = motordict['stator_leakage_inductance']

        if core_material is None:
            self.core_fromdata = False
            # Pure iron
            if self.CoreMaterial == 0.0:
                self.K_mu1 = 0.007940206909425818
                self.K_mu2 = 4.381031839692676
                self.K_mu3 = -6.087231108067038
            # M19 electrical steel
            elif self.CoreMaterial == 1.0:
                self.K_mu1 = 0.005152188423960256
                self.K_mu2 = 4.317060423770096
                self.K_mu3 = -5.631690293817687
            elif self.CoreMaterial == 2.0:
                self.K_mu1 = motordict['K_mu1']
                self.K_mu2 = motordict['K_mu2']
                self.K_mu3 = motordict['K_mu3']
            else:
                raise ValueError('Invalid core material given in motor dictionary')
        else:
            self.core_fromdata = True
            if solver != 'linear':
                self.B_data = core_material[1]
                self.mu_data = core_material[1] / core_material[0]
                self.mu_data_deriv = (self.mu_data[1:] - self.mu_data[:-1]) / (self.B_data[1:] - self.B_data[:-1])

        """Initialise the model state"""

        # Reset the model
        self.reset()

        """Derive additional dimensions"""

        # Basic derived dimensions
        self.r_st_tooth_in = self.r_st_in + self.l_st_head  # Stator tooth inner radius
        self.r_st_tooth_out = self.r_st_tooth_in + self.l_st_tooth  # Stator tooth outer radius
        self.S_st_head = self.w_st_head * self.l_z  # Stator tooth head cross-section
        self.l_st_tooth_toNode = self.l_st_tooth + (
                self.r_st_out - self.r_st_tooth_out) / 2  # Length of tooth base segment to node in yoke
        self.r_rot_out = self.r_st_in - self.d_air  # Rotor outer radius
        self.r_rot_tooth_out = self.r_rot_out - self.l_rot_head  # Rotor tooth outer radius
        self.r_rot_tooth_in = self.r_rot_tooth_out - self.l_rot_tooth  # Rotor tooth inner radius
        self.S_rot_head = self.w_rot_head * self.l_z  # Rotor tooth head cross-section

        # Depth of node in rotor yoke
        self.r_bdn = 0.5 * np.sqrt((np.pi * (
                2 * self.r_rot_out - 2 * self.l_rot_head - 2 * self.l_rot_tooth) * self.w_rot_tooth) / self.N_rot)

        # Cross-sections for flux density calculation
        self.A_st_y = (self.r_st_out - self.r_st_tooth_out) * self.l_z  # Stator yoke cross-section
        self.A_st_t = self.w_st_tooth * self.l_z  # Stator tooth base cross-section
        self.A_rot_y = (self.r_rot_tooth_in - self.r_rot_in) * self.l_z  # Rotor yoke cross-section
        self.A_rot_t = self.w_rot_tooth * self.l_z  # Rotor tooth base cross-section

        # Absolute permeability of a vacuum, used for air
        self.mu_0 = 4 * np.pi * 10 ** (-7)

        # Variables required for the calculation of the air gap permeance
        self.r_rot_out = self.r_st_in - self.d_air
        self.theta_st = self.w_st_head / self.r_st_in
        self.theta_rot = self.w_rot_head / self.r_rot_out

        # Mean air gap permeance neglecting fringing
        self.a_1 = (self.mu_0 * (self.theta_st + self.theta_rot) * self.l_z) / (
                2 * np.log(self.r_st_in / self.r_rot_out))

        # Rotor skew angle in lateral plane
        self.alpha_sk_acc = np.arctan(self.r_rot_out * self.alpha_sk / self.l_z)

        """Construct system matrix submatrices"""

        # Kronecker delta matrices
        self.Kdelta1 = -np.eye(self.N_st) + roll_matrix(np.eye(self.N_st), -1, 0)
        self.Kdelta2 = np.eye(self.N_rot) - roll_matrix(np.eye(self.N_rot), -1, 0)

        # N_abc
        # The N_abc-matrix contains the winding distribution along the stator slots
        N = 6 * self.p * int(self.N_st / (6 * self.p))  # Array rows
        N_abc = np.zeros((N, 3), dtype=np.float64)  # Array size containing zeroes
        # Define the winding pattern
        a_sequence = np.array([0, 1, 0, 0, -1, 0])
        b_sequence = np.array([-1, 0, 0, 1, 0, 0])
        c_sequence = np.array([0, 0, -1, 0, 0, 1])
        idx = 0
        for p_iter in range(self.p):
            for i in range(6):
                for rep in range(int(self.N_st / (6 * self.p))):
                    N_abc[idx, 0] = a_sequence[i]
                    N_abc[idx, 1] = b_sequence[i]
                    N_abc[idx, 2] = c_sequence[i]
                    idx += 1
        self.N_abc = N_abc * self.N_w  # Define the number of turns per slot

        # R_st
        # R_st is a diagonal matrix that contains the three stator phase resistances
        self.R_st = self.R_st_i * np.diag(np.asarray(self.phase_resistance_factor))

        # SubB_Nrot
        # SubB_Nrot is a subtracting matrix
        self.SubB_Nrot = np.eye(self.N_rot) - roll_matrix(np.eye(self.N_rot), 1, 0)

        # R_rot 
        # Rotor end-ring segment resistance
        if self.er_resistanceCalculationMethod == 1.0:
            R_rot_erl = ((np.pi * (2 * self.r_rot_out - 2 * self.l_rot_head - self.l_rot_tooth) * self.rho_rot)
                         / (self.N_rot * self.l_rot_tooth * self.w_rot_er)
                         * np.ones(self.N_rot))
        elif self.er_resistanceCalculationMethod == 0.0:
            alpha_erl = 2 * np.pi / self.N_rot
            A_erl = alpha_erl / 2 * (self.er_outerRadius ** 2 - self.er_innerRadius ** 2)
            l_erl = alpha_erl * np.average([self.er_innerRadius, self.er_outerRadius])
            R_rot_erl = self.rho_rot * l_erl / A_erl * np.ones(self.N_rot)
        else:
            raise Exception

        # Rotor bar resistance
        r_b_o = self.r_rot_out - self.l_rot_head
        r_b_i = r_b_o - self.l_rot_tooth
        alpha_rot_tooth = self.w_rot_tooth / np.average([r_b_o, r_b_i])
        alpha_b = 2 * np.pi / self.N_rot - alpha_rot_tooth
        A_rot_bl = (np.pi * r_b_o ** 2 - np.pi * r_b_i ** 2) * alpha_b / (2 * np.pi)
        r_rot_b = self.r_rot_out - self.l_rot_head - self.l_rot_tooth / 2
        l_rot_bl = np.sqrt(self.l_z ** 2 + (r_rot_b * self.alpha_sk) ** 2)
        R_rot_bl = self.rho_rot * l_rot_bl / A_rot_bl * np.ones(self.N_rot)
        if broken_rotor_bars[0] != 0:
            assert broken_rotor_bars[0] <= self.N_rot
            R_rot_bl[:broken_rotor_bars[0]] = broken_rotor_bars[1] * np.ones(broken_rotor_bars[0])
        # Assumption made of equal equal R_rot_erl
        self.R_rot = (self.nmatmul(rolled_identity(self.N_rot, 2), np.diag(R_rot_bl))
                      - 2 * self.nmatmul(rolled_identity(self.N_rot, 1), np.diag(R_rot_bl + R_rot_erl))
                      + np.diag(R_rot_bl))

        # SubF_Nst
        # SubF_Nst is a subtracting matrix
        self.SubF_Nst = np.eye(self.N_st) - roll_matrix(np.eye(self.N_st), -1, 0)

        # I_Nst
        # I_Nst is an identity matrix
        self.I_Nst = np.eye(self.N_st)

        # SubB_Nst
        # SubB_Nst is a subtracting matrix
        self.SubB_Nst = np.eye(self.N_st) - roll_matrix(np.eye(self.N_st), 1, 0)

        # SubF_Nrot
        # SubF_Nrot is a subtracting matrix
        self.SubF_Nrot = np.eye(self.N_rot) - roll_matrix(np.eye(self.N_rot), -1, 0)

        # I_Nrot
        # I_Nrot is an identity matrix
        self.I_Nrot = np.eye(self.N_rot)

        # Set_1Nst
        # Set_1Nst is a reference setting matrix
        self.Set_1Nst = np.zeros((1, self.N_st))
        self.Set_1Nst[0, 0] = 1

        # R_st_t
        # R_st_t contains the stator tooth reluctances
        # Stator tooth reluctance of base without variable permeability
        R_st_baseh = self.l_st_tooth_toNode / (self.mu_fe * np.ones(self.N_st) * self.A_st_t)
        R_st_headh = self.l_st_head / (self.mu_fe * self.S_st_head)  # Stator tooth head reluctance
        R_st_th = R_st_headh + R_st_baseh  # Stator tooth node to node reluctance
        self.R_st_t = self.nmatmul(self.Kdelta1, np.diag(np.asarray(R_st_th)))

        # R_st_y
        # R_st_y is a diagonal matrix of the stator yoke reluctances
        theta_st_y = 2 * np.pi / self.N_st  # Stator yoke angle
        self.theta_st_y = theta_st_y
        R_st_yh = theta_st_y / (self.mu_fe * np.ones(self.N_st) * self.l_z * np.log(
            self.r_st_out / self.r_st_tooth_out))  # Stator yoke reluctance without variable permeability
        self.R_st_y = np.diag(np.asarray(R_st_yh))

        # R_rot_t
        # R_rot_t contains the rotor tooth reluctances
        R_rot_headl = self.l_rot_head / (2 * self.mu_fe * self.S_rot_head)  # Rotor tooth reluctance of half tooth head
        # Rotor tooth reluctance of base without variable permeability
        R_rot_basel = (self.l_rot_tooth + self.r_bdn) / (self.mu_fe * np.ones(self.N_rot) * self.A_rot_t)
        R_rot_tl = R_rot_headl + R_rot_basel  # Rotor tooth node to node reluctance
        self.R_rot_t = self.nmatmul(self.Kdelta2, np.diag(np.asarray(R_rot_tl)))

        # R_rot_y
        # R_rot_y is a diagonal matrix of the rotor yoke segment reluctances
        theta_rot_y = 2 * np.pi / self.N_rot  # Rotor yoke segment angle spanned
        radius_rot_y = self.r_rot_tooth_in - self.r_bdn
        self.l_rot_y = theta_rot_y * radius_rot_y
        R_rot_yl = self.l_rot_y / (self.mu_fe * np.ones(self.N_rot) * self.A_rot_y) # Rotor yoke reluctance without variable permeability
        self.R_rot_y = np.diag(np.asarray(R_rot_yl))

        # Create the remaining Kdelta matrices
        self.Kdelta3 = roll_matrix(np.eye(self.N_st), 1, 0)
        self.Kdelta4 = roll_matrix(np.eye(self.N_st), -1, 0)
        self.Kdelta5 = roll_matrix(np.eye(self.N_rot), -1, 1)
        self.Kdelta6 = roll_matrix(np.eye(self.N_rot), 1, 1)

        # gamma_hl0
        # Rotor to stator tooth separation angle for non-rotated rotor at depth L/2
        gamma_h = (np.arange(0, self.N_st, 1) * theta_st_y)[:, np.newaxis] * np.ones((self.N_st, self.N_rot))
        gamma_l = (np.arange(0, self.N_rot, 1) * theta_rot_y)[np.newaxis, :] * np.ones((self.N_st, self.N_rot))
        self.gamma_hl = gamma_l - gamma_h
        # Normalise the obtained angular differences
        self.gamma_hl0 = (self.gamma_hl + np.pi) % (2 * np.pi) - np.pi

        # Generate e_h
        gamma_h = gamma_h[:, 0]  # gamma_h is a vector of the stator teeth angles
        self.e_h = np.concatenate((np.cos(gamma_h)[:, np.newaxis], np.sin(gamma_h)[:, np.newaxis]), axis=1)

        """Construct a static version of the system matrix"""

        # Construct the first three rows of A
        self.Kdelta7 = (np.eye(3) - roll_matrix(np.eye(3), -1, 1))[:2, :]
        self.Kdelta8 = -roll_matrix(np.eye(3), 1, 0)
        self.N_abc_T = np.transpose(self.N_abc)

        if self.connectionType == 'directPhase':
            row1 = np.concatenate(
                (z(3, self.N_st), z(3, self.N_rot), z(3, self.N_st), self.N_abc_T, z(3, self.N_rot), z(3, self.N_rot),
                 self.stator_leakage_inductance * np.eye(3), z(3, self.N_rot)), axis=1)
        elif self.connectionType == 'wye':
            row1_voltage = np.concatenate(
                (z(2, self.N_st), z(2, self.N_rot), z(2, self.N_st),
                 self.nmatmul(self.Kdelta7, self.N_abc_T), z(2, self.N_rot),
                 z(2, self.N_rot),
                 self.stator_leakage_inductance * self.Kdelta7, z(2, self.N_rot)), axis=1)
            row1_current = np.concatenate((z(1, 3 * self.N_st + 3 * self.N_rot), np.ones((1, 3)), z(1, self.N_rot)),
                                          axis=1)
            row1 = np.concatenate((row1_voltage, row1_current), axis=0)
        elif self.connectionType == 'delta':
            row1 = np.concatenate(
                (z(3, self.N_st), z(3, self.N_rot), z(3, self.N_st), self.nmatmul(self.Kdelta8, self.N_abc_T), z(3, self.N_rot),
                 z(3, self.N_rot), self.stator_leakage_inductance * self.Kdelta8, z(3, self.N_rot)), axis=1)
        else:
            raise ValueError('Improper value for argument connectionType')

        # Generate the static part of the system matrix
        row2 = np.concatenate((z(self.N_rot, self.N_st), z(self.N_rot, self.N_rot), z(self.N_rot, self.N_st),
                               z(self.N_rot, self.N_st), self.SubB_Nrot, z(self.N_rot, self.N_rot),
                               z(self.N_rot, 3), z(self.N_rot, self.N_rot)), axis=1)
        row3 = np.concatenate(
            (self.SubF_Nst, z(self.N_st, self.N_rot), self.R_st_t, -self.R_st_y, z(self.N_st, self.N_rot),
             z(self.N_st, self.N_rot), self.N_abc, z(self.N_st, self.N_rot)), axis=1)
        row4 = np.concatenate(
            (z(self.N_st, self.N_st), z(self.N_st, self.N_rot), -self.I_Nst, self.SubB_Nst, z(self.N_st, self.N_rot),
             z(self.N_st, self.N_rot), z(self.N_st, 3), z(self.N_st, self.N_rot)),
            axis=1)
        row5 = np.concatenate(
            (z(self.N_rot, self.N_st), self.SubF_Nrot, z(self.N_rot, self.N_st), z(self.N_rot, self.N_st), self.R_rot_t,
             -self.R_rot_y, z(self.N_rot, 3), self.I_Nrot), axis=1)
        row6 = np.concatenate((z(self.N_rot, self.N_st), z(self.N_rot, self.N_rot), z(self.N_rot, self.N_st),
                               z(self.N_rot, self.N_st), self.I_Nrot, self.SubB_Nrot,
                               z(self.N_rot, 3), z(self.N_rot, self.N_rot)), axis=1)
        row7 = np.concatenate((z(self.N_st, self.N_st), z(self.N_st, self.N_rot), self.I_Nst, z(self.N_st, self.N_st),
                               z(self.N_st, self.N_rot), z(self.N_st, self.N_rot),
                               z(self.N_st, 3), z(self.N_st, self.N_rot)), axis=1)
        row8 = np.concatenate((z(self.N_rot, self.N_st), z(self.N_rot, self.N_rot), z(self.N_rot, self.N_st),
                               z(self.N_rot, self.N_st), -self.I_Nrot, z(self.N_rot, self.N_rot),
                               z(self.N_rot, 3), z(self.N_rot, self.N_rot)), axis=1)
        row9 = np.concatenate((self.Set_1Nst, z(1, 4 * self.N_rot + 2 * self.N_st + 3)), axis=1)
        row10 = np.concatenate((z(1, 3 * self.N_st + 3 * self.N_rot + 3), np.ones((1, self.N_rot))), axis=1)

        system_matrix_static = np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8, row9, row10), axis=0)

        # Delete two redundant equations
        rows_to_keep = np.delete(np.arange(len(system_matrix_static)), [3, 3 + self.N_rot + self.N_st])
        system_matrix_static = system_matrix_static[rows_to_keep]
        self.system_matrix_static = system_matrix_static

    @staticmethod
    def nmatmul(arr1: nb.float64[:, :], arr2: nb.float64[:, :]):
        """
        Shorthand for conversion to contiguous arrays and matrix product. Required for fast matrix-products using Numba
        """
        return np.ascontiguousarray(arr1) @ np.ascontiguousarray(arr2)

    def reset(self):
        """
        Resets the state to zero-values
        """
        self.state = np.zeros(3 + int(self.N_st) * 3 + int(self.N_rot) * 4 + 6)

    def mu_phi(self, phi, area):
        """
        Generate the magnetic permeability as a function of the magnetic flux and the flux tube cross-section.
        A hyperbolic tangent approximation is used for this nonlinear relation. Hysteresis is neglected.
        :param phi: magnetic flux
        :param area: flux tube cross-sectional area
        :return: magnetic permeability
        """
        # Calculate the magnetic flux density
        B = phi / area

        if self.core_fromdata == False:
            mu = -self.K_mu1 * np.tanh(self.K_mu2 * np.abs(B) + self.K_mu3) + self.K_mu1 + self.mu_0
        else:
            mu = np.interp(B, self.B_data, self.mu_data)

        return mu

    def dmu_dphi(self, phi, area):
        """
        Calculate the derivative of the magnetic permeability to the magnetic flux
        :param phi: magnetic flux
        :param area: flux tube cross-sectional area
        :return: derivative of the magnetic permeability to the magnetic flux
        """
        B = phi / area

        if self.core_fromdata == False:
            mu_deriv = np.sign(phi) * (
                    -(self.K_mu1 * self.K_mu2 / area) * np.cosh(self.K_mu2 * np.abs(phi) / area + self.K_mu3) ** -2)

        else:
            mu_deriv = np.zeros(B.shape)

            for id, B_i in enumerate(B):
                index = 0

                for B_id, value in enumerate(self.B_data):
                    if value < B_i and B_id > index:
                        index = B_id

                if index > (len(self.mu_data_deriv) - 1):
                    mu_deriv[id] = self.mu_data_deriv[-1]
                else:
                    mu_deriv[id] = self.mu_data_deriv[index]

        return mu_deriv

    def gen_system_matrix(self, state_em, saturation=False):
        """
        Construct the system matrix
        :param state_em: Electromagnetic state
        :param saturation: Whether ferromagnetic saturation is modelled
        :return: The system matrix
        """

        # Read the static elements of the system matrix
        system_matrix = self.system_matrix_static.copy()

        # Read the rotor mechanical angle
        gamma_rot = self.state_rotational[2]

        """The reluctances are recomputed in the loop as a function of the magnetic flux when ferromagnetic saturation
        is modelled"""

        if saturation:
            # Read the magnetic flux
            phi_st_t = state_em[self.N_st + self.N_rot: 2 * self.N_st + self.N_rot]  # Stator tooth flux
            phi_st_y = state_em[2 * self.N_st + self.N_rot: 3 * self.N_st + self.N_rot]  # Stator yoke flux
            phi_rot_t = state_em[3 * self.N_st + self.N_rot: 3 * self.N_st + 2 * self.N_rot]  # Rotor tooth flux
            phi_rot_y = state_em[3 * self.N_st + 2 * self.N_rot: 3 * self.N_st + 3 * self.N_rot]  # Rotor yoke flux

            # Calculate nonlinear reluctances
            # R_st_y
            # R_st_y is a diagonal matrix of the stator yoke reluctances
            R_st_yh = self.theta_st_y / (self.mu_phi(phi_st_y, self.A_st_y) * self.l_z * np.log(
                self.r_st_out / self.r_st_tooth_out))  # Stator yoke reluctance with variable permeability
            R_st_y = np.diag(R_st_yh)
            system_matrix[2 + self.N_rot:2 + self.N_rot + self.N_st,
            2 * self.N_st + self.N_rot:3 * self.N_st + self.N_rot] = -R_st_y

            # R_st_t
            # R_st_t contains the stator tooth reluctances
            # Stator tooth base
            R_st_baseh = self.l_st_tooth_toNode / (self.mu_phi(phi_st_t, self.A_st_t) * self.A_st_t)
            R_st_headh = self.l_st_head / (self.mu_fe * self.S_st_head)  # Stator tooth head
            R_st_th = R_st_headh + R_st_baseh  # Stator tooth node to node reluctance
            R_st_th_diag = np.diag(R_st_th)
            R_st_t = self.nmatmul(self.Kdelta1, R_st_th_diag)
            system_matrix[2 + self.N_rot: 2 + self.N_rot + self.N_st,
            self.N_st + self.N_rot: 2 * self.N_st + self.N_rot] = R_st_t

            # R_rot_t
            # R_rot_t contains the rotor tooth reluctances
            R_rot_headl = self.l_rot_head / (
                    2 * self.mu_fe * self.S_rot_head)  # Tooth head
            R_rot_basel = (self.l_rot_tooth + self.r_bdn) / (self.mu_phi(phi_rot_t,
                                                                         self.A_rot_t) * self.A_rot_t)  # Tooth base
            R_rot_tl = R_rot_headl + R_rot_basel  # Rotor tooth node to node reluctance
            R_rot_tl_diag = np.diag(R_rot_tl)
            R_rot_t = self.nmatmul(self.Kdelta2, R_rot_tl_diag)
            system_matrix[1 + self.N_rot + 2 * self.N_st: 1 + 2 * self.N_rot + 2 * self.N_st,
            3 * self.N_st + self.N_rot: 3 * self.N_st + 2 * self.N_rot] = R_rot_t

            # R_rot_y
            # R_rot_y is a diagonal matrix of the rotor yoke segment reluctances
            theta_rot_y = 2 * np.pi / self.N_rot  # Rotor yoke segment angle spanned
            radius_rot_y = self.r_rot_tooth_in - self.r_bdn
            R_rot_yl = self.l_rot_y / (self.mu_phi(phi_rot_y, self.A_rot_y) * self.A_rot_y)  # Rotor yoke reluctance without variable permeability
            R_rot_y = np.diag(R_rot_yl)
            system_matrix[1 + self.N_rot + 2 * self.N_st: 1 + 2 * self.N_rot + 2 * self.N_st,
            3 * self.N_st + 2 * self.N_rot: 3 * self.N_st + 3 * self.N_rot] = -R_rot_y

        """Air gap permeances"""

        # P_air
        # P_air is a matrix that consists of the permeances linking all rotor teeth to all stator teeth
        # Apply rotor rotation and renormalise
        self.gamma_hl = (self.gamma_hl0 + gamma_rot + np.pi) % (2 * np.pi) - np.pi

        # Air gap permeance calculation
        if self.alpha_sk == 0.0:  # No skew implemented
            P_air = self.P_air_hl_noskew()
        elif self.skew_method == 'NISM':  # Numerical method air gap permeance
            P_air = self.P_air_hl_NISM(self.gamma_hl)
        elif self.skew_method == 'AISM':  # Numerical method air gap permeance
            P_air = self.P_air_hl_AISM()
        elif self.skew_method == 'ostovic':
            P_air = self.P_air_hl_ostovic()
        else:
            raise ValueError('Invalid argument value for skew_method')

        system_matrix[1 + 2 * self.N_st + 3 * self.N_rot: 1 + 3 * self.N_st + 3 * self.N_rot,
        self.N_st: self.N_st + self.N_rot] = -P_air
        system_matrix[1 + 3 * self.N_st + 3 * self.N_rot: 1 + 3 * self.N_st + 4 * self.N_rot,
        : self.N_st] = -np.transpose(P_air)

        """Leakage permeances"""

        # P_sigma_h
        # P_sigma_h is the leakage air permeance between adjacent stator teeth
        r_sigmain_h = self.r_st_in  # Stator leakage permeance inner radius
        r_sigmaout_h = r_sigmain_h + self.l_st_head  # Stator leakage permeance outer radius
        theta_sigma_h = 2 * np.pi / self.N_st - self.theta_st  # Angle between stator teeth edges
        P_sigma_h = self.mu_0 * self.l_z * np.log(
            r_sigmaout_h / r_sigmain_h) / theta_sigma_h  # Stator leakage permeance

        # P_sigma_l
        # P_sigma_l is the leakage air permeance between adjacent rotor teeth
        r_sigmaout_l = self.r_st_in - self.d_air  # Rotor leakage permeance outer radius
        r_sigmain_l = r_sigmaout_l - self.l_rot_head  # Rotor leakage permeance inner radius
        theta_sigma_l = 2 * np.pi / self.N_rot - self.theta_rot  # Angle between rotor teeth edges
        P_sigma_l = self.mu_0 * self.l_z * np.log(r_sigmaout_l / r_sigmain_l) / theta_sigma_l  # Rotor leakage permeance

        # P_st_rot
        # P_st_rot is a permeance matrix which consists of leakage and air gap permeances
        P_st_rot = -self.Kdelta3 * P_sigma_h + np.identity(self.N_st) * 2 * P_sigma_h + np.diag(
            np.sum(P_air, axis=1)) - self.Kdelta4 * P_sigma_h
        system_matrix[1 + 2 * self.N_st + 3 * self.N_rot: 1 + 3 * self.N_st + 3 * self.N_rot, : self.N_st] = P_st_rot

        # P_rot_st
        # P_rot_st is a permeance matrix which consists of leakage and air gap permeances
        P_rot_st = -self.Kdelta5 * P_sigma_l + np.identity(self.N_rot) * 2 * P_sigma_l + np.diag(
            np.sum(P_air, axis=0)) - self.Kdelta6 * P_sigma_l
        system_matrix[1 + 3 * self.N_st + 3 * self.N_rot: 1 + 3 * self.N_st + 4 * self.N_rot,
        self.N_st: self.N_st + self.N_rot] = P_rot_st

        """ OPTIONAL Check whether the system matrix is full rank
        # Calculating the matrix rank takes a significant amount of calculation time
        if system_matrix.shape[0] != np.linalg.matrix_rank(system_matrix):
            warnings.warn('The system is not uniquely determined, check the system matrix')
        """

        return system_matrix

    def gen_rhs(self, v_input: np.ndarray, state_em: np.ndarray):
        """
        Generate the system right-hand side vector
        :param v_input: input voltages
        :param state_em: electromagnetic state
        :return: System right-hand side vector
        """

        # Read the required state quantities
        flux_st_yoke = state_em[2 * self.N_st + self.N_rot: 3 * self.N_st + self.N_rot]
        flux_rot_tooth = state_em[3 * self.N_st + self.N_rot: 3 * self.N_st + 2 * self.N_rot]
        i_rot = state_em[3 * self.N_st + 3 * self.N_rot + 3: 3 * self.N_st + 4 * self.N_rot + 3]
        i_st = state_em[3 * self.N_st + 3 * self.N_rot: 3 * self.N_st + 3 * self.N_rot + 3]

        # Construct the RHS first element
        if self.connectionType == 'directPhase':
            row1_rhs = (self.timestep * v_input
                        + self.nmatmul(self.N_abc_T, flux_st_yoke)
                        + self.nmatmul((self.stator_leakage_inductance * np.eye(3) - self.timestep * self.R_st)
                        , i_st))
        elif self.connectionType == 'wye':
            row1_voltage = (self.timestep * v_input[:2]
                            + self.nmatmul(self.nmatmul(self.Kdelta7, self.N_abc_T), flux_st_yoke)
                            + self.nmatmul((self.stator_leakage_inductance * self.Kdelta7 - self.timestep * self.nmatmul(self.Kdelta7, self.R_st))
                            , i_st))
            row1_rhs = np.concatenate((row1_voltage, np.array([0])))
        elif self.connectionType == 'delta':
            row1_rhs = (self.timestep * v_input
                        + self.nmatmul(self.nmatmul(self.Kdelta8, self.N_abc_T), flux_st_yoke)
                        + self.nmatmul((self.stator_leakage_inductance * self.Kdelta8 - self.timestep * self.Kdelta8)
                        , i_st))
        else:
            raise ValueError('Improper value for argument connectionType.')

        # Construct the RHS second element
        row2_rhs = self.nmatmul(self.SubB_Nrot, flux_rot_tooth) + self.timestep * self.nmatmul(self.R_rot, i_rot)

        # Construct the RHS vector
        rhs = np.concatenate((row1_rhs, row2_rhs, np.zeros(3 * self.N_st + 3 * self.N_rot + 1)))

        # Remove elements of redundant equations
        rhs = np.concatenate((rhs[:3], rhs[4:]))

        return rhs

    def gen_Jac(self, system_matrix: np.ndarray, state: np.ndarray):
        """
        Calculate the Jacobian of the system matrix
        :param system_matrix: motor model system matrix
        :param state: motor model state
        :return: Jacobian with respect to state variables
        """
        # Flux extraction and area calculation
        phi_st_t = state[self.N_st + self.N_rot:2 * self.N_st + self.N_rot].flatten()
        phi_st_y = state[2 * self.N_st + self.N_rot:3 * self.N_st + self.N_rot].flatten()
        phi_rot_t = state[3 * self.N_st + self.N_rot:3 * self.N_st + 2 * self.N_rot].flatten()
        phi_rot_y = state[3 * self.N_st + 2 * self.N_rot:3 * self.N_st + 3 * self.N_rot].flatten()

        # derivative of Rstt
        dRstt_vector = -(self.l_st_tooth_toNode / self.A_st_t) * (
                1 / self.mu_phi(phi_st_t, self.A_st_t) ** 2) * self.dmu_dphi(phi_st_t, self.A_st_t)
        dRstt_matrix = -np.diag(dRstt_vector * phi_st_t) + roll_matrix(np.diag(dRstt_vector * phi_st_t), 1, 1)

        # derivative of Rsty
        dRsty_vector = -self.theta_st / (self.l_z * np.log(self.r_st_out / self.r_st_tooth_out)) * (
                1 / self.mu_phi(phi_st_y, self.A_st_y) ** 2) * self.dmu_dphi(phi_st_y, self.A_st_y)
        dRsty_matrix = np.diag(dRsty_vector * phi_st_y)

        # derivative of Rrott
        dRrott_vector = -(self.l_rot_tooth + self.r_bdn) / self.A_rot_t * (
                1 / self.mu_phi(phi_rot_t, self.A_rot_t) ** 2) * self.dmu_dphi(phi_rot_t, self.A_rot_t)
        dRrott_matrix = np.diag(dRrott_vector * phi_rot_t) - roll_matrix(np.diag(dRrott_vector * phi_rot_t), 1, 1)

        # derivative of Rroty
        dRroty_vector = -(self.l_rot_y) / self.A_rot_y * (
                1 / self.mu_phi(phi_rot_y, self.A_rot_y) ** 2) * self.dmu_dphi(phi_rot_y, self.A_rot_y)
        dRroty_matrix = np.diag(dRroty_vector * phi_rot_y)

        # derivative of A to x
        dAdx_x = np.zeros(system_matrix.shape)
        dAdx_x[3 + self.N_rot - 1:3 + self.N_st + self.N_rot - 1,
        self.N_st + self.N_rot:2 * self.N_st + self.N_rot] = dRstt_matrix
        dAdx_x[3 + self.N_rot - 1:3 + self.N_st + self.N_rot - 1,
        2 * self.N_st + self.N_rot:3 * self.N_st + self.N_rot] = -dRsty_matrix
        dAdx_x[3 + 2 * self.N_st + self.N_rot - 2:3 + 2 * self.N_st + 2 * self.N_rot - 2,
        3 * self.N_st + self.N_rot:3 * self.N_st + 2 * self.N_rot] = dRrott_matrix
        dAdx_x[3 + 2 * self.N_st + self.N_rot - 2:3 + 2 * self.N_st + 2 * self.N_rot - 2,
        3 * self.N_st + 2 * self.N_rot:3 * self.N_st + 3 * self.N_rot] = -dRroty_matrix

        # Assemble Jacobian
        Jac = dAdx_x + system_matrix
        return Jac

    def dP_decc_noskew(self):
        K1 = self.a_1 * self.d_air / (self.d_air - self.r_ecc_m) ** 2
        dP_decc = K1 * np.exp(-(self.gamma_hl / self.theta_st) ** 2) * np.exp(-(self.gamma_hl / self.theta_rot) ** 2)
        return dP_decc

    def dP_decc_NISM(self):
        dz = self.l_z / (self.nism_slices - 1)
        K_0 = (self.mu_0 * self.c_perm_shape * (self.theta_st + self.theta_rot)
               / (2 * np.log(self.r_st_in / self.r_rot_out)))
        K_1 = self.c_perm_sat * (1 / self.theta_st ** 2 + 1 / self.theta_rot ** 2)
        integrand = np.zeros((self.N_st, self.N_rot, self.nism_slices))
        for slice_z in range(self.nism_slices - 1):
            gamma_hl1 = self.gamma_hl + (np.tan(self.alpha_sk_acc) / self.r_rot_out) * (slice_z * dz - self.l_z / 2)
            gamma_hl2 = (self.gamma_hl + (np.tan(self.alpha_sk_acc) / self.r_rot_out)
                         * ((slice_z + 1) * dz - self.l_z / 2))
            integrand[:, :, slice_z] = (1 / 2) * (np.exp(-K_1 * gamma_hl1 ** 2) + np.exp(-K_1 * gamma_hl2 ** 2)) * dz
        dP_decc = -K_0 * (self.d_air / (self.d_air - self.r_ecc_m) ** 2) * np.sum(integrand, axis=2)
        return dP_decc

    def dP_decc_AISM(self):
        K_0 = (self.mu_0 * self.c_perm_shape * (self.theta_st + self.theta_rot)
               / (2 * np.log(self.r_st_in / self.r_rot_out)))
        K_1 = self.c_perm_sat * (1 / self.theta_st ** 2 + 1 / self.theta_rot ** 2)
        K_a_2 = 4 * np.sqrt(K_1) / np.sqrt(np.pi)
        K_a_3 = np.cosh(2 * np.sqrt(K_1) * self.l_z * np.tan(self.alpha_sk_acc) / (np.sqrt(np.pi) * self.r_rot_out))
        K_a_4 = np.sqrt(np.pi) * self.r_rot_out / (np.tan(self.alpha_sk_acc) * np.sqrt(K_1)) * np.sinh(
            2 * np.sqrt(K_1) * self.l_z * np.tan(self.alpha_sk_acc) / (np.sqrt(np.pi) * self.r_rot_out))
        dP_decc = K_0 * self.d_air / (self.d_air - self.r_ecc_m) ** 2 * K_a_4 / (np.cosh(K_a_2 * self.gamma_hl) + K_a_3)
        return dP_decc

    def dP_decc_ostovic(self):
        w_w = max(self.theta_st * (self.r_rot_out + self.d_air), self.theta_rot * self.r_rot_out)
        w_n = min(self.theta_st * (self.r_rot_out + self.d_air), self.theta_rot * self.r_rot_out)
        D_ag = 2 * self.r_rot_out + self.d_air
        O_SS = ((2 * np.pi / self.N_st) - self.theta_st) * (self.r_rot_out + self.d_air)
        O_SR = ((2 * np.pi / self.N_rot) - self.theta_rot) * self.r_rot_out
        w_st = self.theta_st * (self.r_rot_out + self.d_air)
        w_rt = self.theta_rot * self.r_rot_out
        gamma_t = (w_st + w_rt + O_SS + O_SR + self.l_z * np.tan(self.alpha_sk_acc)) / D_ag

        if 0 <= np.tan(self.alpha_sk_acc) <= (w_w - w_n) / self.l_z:
            A_max = self.l_z * w_n
            gamma_acc_t = (w_w - w_n - self.l_z * np.tan(self.alpha_sk_acc)) / D_ag
        elif (w_w - w_n) / self.l_z <= np.tan(self.alpha_sk_acc) <= (w_w + w_n) / self.l_z:
            A_max = (self.l_z * (w_w + w_n) / 2 - (self.l_z ** 2 * np.tan(self.alpha_sk_acc))
                     / 4 - ((w_w - w_n) ** 2) / (4 * np.tan(self.alpha_sk_acc)))
            gamma_acc_t = 0
        elif np.tan(self.alpha_sk_acc) >= (w_w + w_n) / self.l_z:
            A_max = (w_w * w_n) / np.tan(self.alpha_sk_acc)
            gamma_acc_t = (self.l_z * np.tan(self.alpha_sk_acc) - w_w - w_n) / D_ag
        else:
            raise Exception

        g_e = self.d_air - self.r_ecc

        dP_decc = np.zeros((self.N_st, self.N_rot))
        for h in range(self.N_st):
            for l in range(self.N_rot):
                if self.gamma_hl[h, l] <= -gamma_t or gamma_t <= self.gamma_hl[h, l]:
                    dP_decc[h, l] = 0.0
                elif -gamma_t <= self.gamma_hl[h, l] <= -gamma_acc_t:
                    dP_decc[h, l] = -self.mu_0 * A_max / (2 * g_e[h] ** 2) * (
                            1 + np.cos(np.pi * (self.gamma_hl[h, l] + gamma_acc_t) / (gamma_t - gamma_acc_t)))
                elif -gamma_acc_t <= self.gamma_hl[h, l] <= gamma_acc_t:
                    dP_decc[h, l] = -self.mu_0 * A_max / g_e[h] ** 2
                elif gamma_acc_t <= self.gamma_hl[h, l] <= gamma_t:
                    dP_decc[h, l] = -self.mu_0 * A_max / (2 * g_e[h] ** 2) * (
                            1 + np.cos(np.pi * (self.gamma_hl[h, l] - gamma_acc_t) / (gamma_t - gamma_acc_t)))
        return dP_decc

    def gen_dP_decc(self):
        """
        Calculate the derivative of the air gap permeances w.r.t. the eccentricity
        """
        # Unskewed rotor
        if self.alpha_sk == 0.0:
            dP_decc = self.dP_decc_noskew()
        # Numerical method for skewed rotor
        elif self.skew_method == 'NISM':
            dP_decc = self.dP_decc_NISM()
        # Analytical method for skewed rotor
        elif self.skew_method == 'AISM':
            dP_decc = self.dP_decc_AISM()
        elif self.skew_method == 'ostovic':
            dP_decc = self.dP_decc_ostovic()
        else:
            raise ValueError('Incorrect value for skew_method provided')

        return dP_decc

    def dP_dgamma_noskew(self):
        """
        Calculation of partial derivative of air gap permeance to rotor angle: unskewed rotor
        """
        # Constant shape factor
        K1 = (-2 * self.a_1 * self.d_air * self.gamma_hl / (self.d_air - self.r_ecc_m)
              * (self.theta_st ** 2 + self.theta_rot ** 2) / (self.theta_st ** 2 * self.theta_rot ** 2))

        # Analytical calculation of partial derivative
        dP_dgamma = K1 * np.exp(-(self.gamma_hl / self.theta_st) ** 2) * np.exp(-(self.gamma_hl / self.theta_rot) ** 2)

        return dP_dgamma

    def dP_dgamma_NISM(self):
        """
        Calculation of partial derivative of air gap permeance to rotor angle: skewed rotor NISM method
        """
        # Rotor finite angular difference for derivative estimation
        Deltagamma = 1e-3

        # Evaluate NISM twice to approximate the partial derivative
        dP_dgamma = (self.P_air_hl_NISM(self.gamma_hl + Deltagamma / 2)
                     - self.P_air_hl_NISM(self.gamma_hl - Deltagamma / 2)) / Deltagamma

        return dP_dgamma

    def dP_dgamma_AISM(self):
        """
        Calculation of partial derivative of air gap permeance to rotor angle: skewed rotor AISM method
        """
        # Constant shape factors
        K_0 = (self.mu_0 * self.c_perm_shape * (self.theta_st + self.theta_rot)
               / (2 * np.log(self.r_st_in / self.r_rot_out)))
        K_1 = self.c_perm_sat * (1 / self.theta_st ** 2 + 1 / self.theta_rot ** 2)
        K_a_2 = 4 * np.sqrt(K_1) / np.sqrt(np.pi)
        K_a_3 = np.cosh(2 * np.sqrt(K_1) * self.l_z * np.tan(self.alpha_sk_acc) / (np.sqrt(np.pi) * self.r_rot_out))
        K_a_4 = np.sqrt(np.pi) * self.r_rot_out / (np.tan(self.alpha_sk_acc) * np.sqrt(K_1)) * np.sinh(
            2 * np.sqrt(K_1) * self.l_z * np.tan(self.alpha_sk_acc) / (np.sqrt(np.pi) * self.r_rot_out))

        # Analytical partial derivative calculation
        dP_dgamma = -K_0 * self.d_air / (self.d_air - self.r_ecc_m) * K_a_4 * K_a_2 * np.sinh(K_a_2 * self.gamma_hl) / (
                np.cosh(K_a_2 * self.gamma_hl) + K_a_3) ** 2

        return dP_dgamma

    def dP_dgamma_ostovic(self):
        """
        Calculation of partial derivative of air gap permeance to rotor angle: skewed rotor Ostovic (1989) method
        """
        # Largest tooth head width
        w_w = max(self.theta_st * (self.r_rot_out + self.d_air), self.theta_rot * self.r_rot_out)

        # Smallest tooth head width
        w_n = min(self.theta_st * (self.r_rot_out + self.d_air), self.theta_rot * self.r_rot_out)

        # Average air gap diameter
        D_ag = 2 * self.r_rot_out + self.d_air

        # Slot opening widths
        O_SS = ((2 * np.pi / self.N_st) - self.theta_st) * (self.r_rot_out + self.d_air)
        O_SR = ((2 * np.pi / self.N_rot) - self.theta_rot) * (self.r_rot_out)

        # Stator and rotor tooth head widths
        w_st = self.theta_st * (self.r_rot_out + self.d_air)
        w_rt = self.theta_rot * self.r_rot_out

        # Zero-permeance angle
        gamma_t = (w_st + w_rt + O_SS + O_SR + self.l_z * np.tan(self.alpha_sk_acc)) / D_ag

        # Specific angle
        if 0 <= np.tan(self.alpha_sk_acc) <= (w_w - w_n) / self.l_z:
            A_max = self.l_z * w_n
            gamma_acc_t = (w_w - w_n - self.l_z * np.tan(self.alpha_sk_acc)) / D_ag
        elif (w_w - w_n) / self.l_z <= np.tan(self.alpha_sk_acc) <= (w_w + w_n) / self.l_z:
            A_max = self.l_z * (w_w + w_n) / 2 - (self.l_z ** 2 * np.tan(self.alpha_sk_acc)) / 4 - (
                    (w_w - w_n) ** 2) / (
                            4 * np.tan(self.alpha_sk_acc))
            gamma_acc_t = 0
        elif np.tan(self.alpha_sk_acc) >= (w_w + w_n) / self.l_z:
            A_max = (w_w * w_n) / np.tan(self.alpha_sk_acc)
            gamma_acc_t = (self.l_z * np.tan(self.alpha_sk_acc) - w_w - w_n) / D_ag
        else:
            raise Exception

        # Rotor eccentricity
        g_e = self.d_air - self.r_ecc

        # Air gap permeance partial derivative
        dP_dgamma = np.zeros((self.N_st, self.N_rot))
        for h in range(self.N_st):
            for l in range(self.N_rot):
                if self.gamma_hl[h, l] <= -gamma_t or self.gamma_hl[h, l] >= gamma_t:
                    dP_dgamma[h, l] = 0.0
                elif -gamma_t <= self.gamma_hl[h, l] <= -gamma_acc_t:
                    dP_dgamma[h, l] = -self.mu_0 * A_max / (2 * g_e[h]) * np.pi / (gamma_t - gamma_acc_t) * np.sin(
                        np.pi * (self.gamma_hl[h, l] + gamma_acc_t) / (gamma_t - gamma_acc_t))
                elif -gamma_acc_t <= self.gamma_hl[h, l] <= gamma_acc_t:
                    dP_dgamma[h, l] = 0.0
                elif gamma_acc_t <= self.gamma_hl[h, l] <= gamma_t:
                    dP_dgamma[h, l] = -self.mu_0 * A_max / (2 * g_e[h]) * np.pi / (gamma_t - gamma_acc_t) * np.sin(
                        np.pi * (self.gamma_hl[h, l] - gamma_acc_t) / (gamma_t - gamma_acc_t))
                else:
                    raise Exception

        return dP_dgamma

    def P_air_hl_noskew(self):
        """
        Calculation of air gap permeance: unskewed rotor (Lannoo 2020)
        """
        # Analytical calculation
        P_air_hl = ((self.a_1 * self.d_air) / (self.d_air - self.r_ecc_m)
                    * (np.exp(-(self.gamma_hl / self.theta_st) ** 2)
                       * np.exp(-(self.gamma_hl / self.theta_rot) ** 2)))

        return P_air_hl

    def P_air_hl_NISM(self, gamma_hl: np.ndarray):
        """
        Calculation of air gap permeance: skewed rotor NISM method (Desenfans 2024)
        :param gamma_hl: Rotor to stator tooth separation angles
        """
        # Slice axial length
        dz = self.l_z / (self.nism_slices - 1)

        # Constant shape factors
        K_0 = (self.mu_0 * self.c_perm_shape * (self.theta_st + self.theta_rot)
               / (2 * np.log(self.r_st_in / self.r_rot_out)))
        K_1 = self.c_perm_sat * (1 / self.theta_st ** 2 + 1 / self.theta_rot ** 2)

        # Initialise integrand
        integrand = np.zeros((self.N_st, self.N_rot, self.nism_slices))

        # Numerically evaluate slices using trapezoidal integration
        for slice_z in range(self.nism_slices - 1):
            gamma_hl1 = gamma_hl + (np.tan(self.alpha_sk_acc) / self.r_rot_out) * (slice_z * dz - self.l_z / 2)
            gamma_hl2 = gamma_hl + (np.tan(self.alpha_sk_acc) / self.r_rot_out) * ((slice_z + 1) * dz - self.l_z / 2)
            integrand[:, :, slice_z] = (1 / 2) * (np.exp(-K_1 * gamma_hl1 ** 2) + np.exp(-K_1 * gamma_hl2 ** 2)) * dz

        # Calculate air gap permeances
        P_air_hl = K_0 * (self.d_air / (self.d_air - self.r_ecc_m)) * np.sum(integrand, axis=2)

        return P_air_hl

    def P_air_hl_AISM(self):
        """
        Calculation of air gap permeance: skewed rotor AISM method (Desenfans 2024)
        """
        # Constant shape factors
        K_0 = (self.mu_0 * self.c_perm_shape * (self.theta_st + self.theta_rot)
               / (2 * np.log(self.r_st_in / self.r_rot_out)))
        K_1 = self.c_perm_sat * (1 / self.theta_st ** 2 + 1 / self.theta_rot ** 2)
        K_a_2 = 4 * np.sqrt(K_1) / np.sqrt(np.pi)
        K_a_3 = np.cosh(2 * np.sqrt(K_1) * self.l_z * np.tan(self.alpha_sk_acc) / (np.sqrt(np.pi) * self.r_rot_out))
        K_a_4 = np.sqrt(np.pi) * self.r_rot_out / (np.tan(self.alpha_sk_acc) * np.sqrt(K_1)) * np.sinh(
            2 * np.sqrt(K_1) * self.l_z * np.tan(self.alpha_sk_acc) / (np.sqrt(np.pi) * self.r_rot_out))

        # Analytical air gap permeance calculation
        P_air_hl = K_0 * (self.d_air / (self.d_air - self.r_ecc_m)) * K_a_4 / (np.cosh(K_a_2 * self.gamma_hl) + K_a_3)

        return P_air_hl

    def P_air_hl_ostovic(self):
        """
        Calculation of air gap permeance: skewed rotor Ostovic method (OStovic 1989)
        """
        # Largest tooth head width
        w_w = max(self.theta_st * (self.r_rot_out + self.d_air), self.theta_rot * self.r_rot_out)

        # Smallest tooth head width
        w_n = min(self.theta_st * (self.r_rot_out + self.d_air), self.theta_rot * self.r_rot_out)

        # Average air gap diameter
        D_ag = 2 * self.r_rot_out + self.d_air

        # Stator and rotor slot gap width
        O_SS = ((2 * np.pi / self.N_st) - self.theta_st) * (self.r_rot_out + self.d_air)
        O_SR = ((2 * np.pi / self.N_rot) - self.theta_rot) * self.r_rot_out

        # Stator and rotor tooth head widths
        w_st = self.theta_st * (self.r_rot_out + self.d_air)
        w_rt = self.theta_rot * self.r_rot_out

        # Zero-permeance angle
        gamma_t = (w_st + w_rt + O_SS + O_SR + self.l_z * np.tan(self.alpha_sk_acc)) / D_ag

        # Specific angle
        if 0 <= np.tan(self.alpha_sk_acc) <= (w_w - w_n) / self.l_z:
            A_max = self.l_z * w_n
            gamma_acc_t = (w_w - w_n - self.l_z * np.tan(self.alpha_sk_acc)) / D_ag
        elif (w_w - w_n) / self.l_z <= np.tan(self.alpha_sk_acc) <= (w_w + w_n) / self.l_z:
            A_max = (self.l_z * (w_w + w_n) / 2 - (self.l_z ** 2 * np.tan(self.alpha_sk_acc))
                     / 4 - ((w_w - w_n) ** 2) / (4 * np.tan(self.alpha_sk_acc)))
            gamma_acc_t = 0
        elif np.tan(self.alpha_sk_acc) >= (w_w + w_n) / self.l_z:
            A_max = (w_w * w_n) / np.tan(self.alpha_sk_acc)
            gamma_acc_t = (self.l_z * np.tan(self.alpha_sk_acc) - w_w - w_n) / D_ag
        else:
            raise Exception

        # Rotor eccentricity
        g_e = self.d_air - self.r_ecc  # Outwards stator e_h reference

        # Air gap permeance calculation
        P_air_hl = np.zeros((self.N_st, self.N_rot))
        for h in range(self.N_st):
            for l in range(self.N_rot):
                if self.gamma_hl[h, l] <= -gamma_t or gamma_t <= self.gamma_hl[h, l]:
                    P_air_hl[h, l] = 0.0
                elif -gamma_t <= self.gamma_hl[h, l] <= -gamma_acc_t:
                    P_air_hl[h, l] = self.mu_0 * A_max / (2 * g_e[h]) * (
                            1 + np.cos(np.pi * (self.gamma_hl[h, l] + gamma_acc_t) / (gamma_t - gamma_acc_t)))
                elif -gamma_acc_t <= self.gamma_hl[h, l] <= gamma_acc_t:
                    P_air_hl[h, l] = self.mu_0 * A_max / g_e[h]
                elif gamma_acc_t <= self.gamma_hl[h, l] <= gamma_t:
                    P_air_hl[h, l] = self.mu_0 * A_max / (2 * g_e[h]) * (
                            1 + np.cos(np.pi * (self.gamma_hl[h, l] - gamma_acc_t) / (gamma_t - gamma_acc_t)))

        return P_air_hl

    def motor_mechanical_model_rotational(self, T_l: float, state_em_new: np.ndarray):
        """
        Model the rotational dynamics of the induction motor
        :param T_l: load torque
        :return: load torque, magnetic scalar potentials
        """
        """Advance the rotor rotational state"""

        # Rotor mechanical speed
        omega_m_new = ((1 - self.timestep * self.D / self.J) * self.state_rotational[1]
                       + self.timestep / self.J * (self.state[-3] - T_l))

        # Rotor mechanical angle
        gamma_m_new = self.state_rotational[2] + self.timestep * self.state_rotational[1]

        # Rotor mechanical acceleration
        alpha_m_new = (omega_m_new - self.state_rotational[1]) / self.timestep

        # Construct the new rotational state
        self.state_rotational = np.array([alpha_m_new, omega_m_new, gamma_m_new])

        """Electromagnetic torque calculation"""
        # d(P_air)/d(gamma_rot)
        # Derivative of the air gap permeance w.r.t. the rotor angle is needed to calculate the electromagnetic torque
        if self.alpha_sk == 0.0:  # No skew implemented
            dP_dgamma = self.dP_dgamma_noskew()
        elif self.skew_method == 'NISM':  # Numerical method
            dP_dgamma = self.dP_dgamma_NISM()
        elif self.skew_method == 'AISM':  # Analytical method
            dP_dgamma = self.dP_dgamma_AISM()
        elif self.skew_method == 'ostovic':
            dP_dgamma = self.dP_dgamma_ostovic()
        else:
            raise ValueError('Invalid argument for skew_method')

        # Tooth-to-tooth magnetic scalar potential differences
        psi_st = (state_em_new[0:self.N_st]).flatten()
        psi_rot = (state_em_new[self.N_st:self.N_st + self.N_rot]).flatten()
        Psi = (np.broadcast_to(psi_st[:, np.newaxis], (psi_st.shape[0], self.N_rot))
               - np.ones((self.N_st, 1)) * psi_rot[np.newaxis, :])

        # Using all teeth potentials
        if self.torque_method == 'nodal':
            T_em_t = 0.5 * (Psi ** 2 * dP_dgamma).sum(axis=1)  # Torque produced per tooth
            T_em = T_em_t.sum()  # Total rotor electromagnetic torque

        # Using Clarke-transformed quantities
        elif self.torque_method == 'clarke':
            i_st_clarke = clarke(state_em_new[3 * self.N_st + 3 * self.N_rot: 3 * self.N_st + 3 * self.N_rot + 3])
            flux_st_tooth = state_em_new[2 * self.N_st + self.N_rot: 3 * self.N_st + self.N_rot]
            lambda_st_clarke = clarke(self.nmatmul(self.N_abc_T, flux_st_tooth))
            T_em = self.p * (lambda_st_clarke[0] * i_st_clarke[1] - lambda_st_clarke[1] * i_st_clarke[0])
        else:
            raise ValueError('Invalid value for torque_method')

        return T_em, Psi

    def virtual_work(self, T_l: float, state_em_new: np.ndarray):
        """
        Virtual work model for rotor force computational and rotational dynamics
        :param T_l: load torque
        :return: electromagnetic torque, unbalanced magnetic pull
        """
        # Calculate the electromagnetic torque and magnetic scalar potential differences
        T_em, Psi = self.motor_mechanical_model_rotational(T_l, state_em_new)

        # Calculate the derivative of the air gap permeances to the rotor eccentricity
        dP_air_dr_ecc = self.gen_dP_decc()

        # Calculate the unbalanced magnetic pull per stator tooth
        F_emx_t = 0.5 * (Psi ** 2 * dP_air_dr_ecc).sum(axis=1) * self.e_h[:, 0]
        F_emy_t = 0.5 * (Psi ** 2 * dP_air_dr_ecc).sum(axis=1) * self.e_h[:, 1]

        # Calculate the total unbalanced magnetic pull
        F_emx, F_emy = F_emx_t.sum(), F_emy_t.sum()

        return (T_em, F_emx, F_emy)

    def step(self, inputs: np.ndarray):
        """
        Main IMMEC method which advances the model state by one time step
        :param inputs: Concatenated array of the three voltage inputs, the load torque, and the rotor eccentricity
        """

        # Extract the applied load torque
        T_l = inputs[3]

        # Calculate the rotor eccentricity matrix
        self.d_ecc = inputs[4:6]
        self.r_ecc = self.nmatmul(self.e_h, self.d_ecc)
        r_ecc_m = np.broadcast_to(self.r_ecc, (self.N_rot, len(self.r_ecc)))
        self.r_ecc_m = np.transpose(r_ecc_m)

        # Raise an exception if the eccentricity exceeds the air gap length
        if np.linalg.norm(self.d_ecc) > self.d_air:
            raise ValueError('The rotor eccentricity input exceeds the air gap length')

        # Extract the electromagnetic state
        state_em = self.state[:3 + 3 * self.N_st + 4 * self.N_rot].copy()

        # Extract the rotational state
        self.state_rotational = self.state[-6:-3]

        # Calculate the system right-hand side
        rhs = self.gen_rhs(inputs[:3], state_em)  # Generate the RHS vector

        # Linear solving
        if self.solver == 'linear':
            system_matrix = self.gen_system_matrix(state_em)  # Generate the system matrix

            # Solve the system
            if self.linearsolver == 'algebraic':
                state_em_new = np.linalg.solve(system_matrix, rhs)
            elif self.linearsolver == 'least-squares':
                state_em_new = np.linalg.lstsq(system_matrix, rhs, rcond=1e-99)[0]
            else:
                raise Exception("Invalid value given for linearsolver. Try algebraic or least-squares")

        # Nonlinear solving using successive substitution
        elif self.solver == 'successive':
            error = 1e9  # Nonlinear solving error
            state_em_iter = state_em.copy()  # Copy the electromagnetic state
            self.iterations = 0  # Initialise the number of iterations

            while error > self.solving_tolerance:
                # Check whether the maximally allowed number of iterations is exceeded
                if self.iterations > self.iterations_threshold:
                    raise NoConvergenceException('Convergence not obtained by nonlinear solver')

                # System assembly
                system_matrix = self.gen_system_matrix(state_em_iter, saturation=True)  # Calculate the system matrix

                # Root finding
                root = np.linalg.solve(system_matrix, rhs)

                # Estimation update
                state_em_iter += (root - state_em_iter) * self.relaxation_factor

                # Increment the iteration number
                self.iterations += 1

                # Calculate the error
                error = np.linalg.norm(self.nmatmul(system_matrix, state_em_iter) - rhs, ord=np.inf)

            # Update the electromagnetic state
            state_em_new = state_em_iter

        # Nonlinear solving using Newton's method
        elif self.solver == 'newton':
            error = 1e9  # Nonlinear solving error
            state_em_iter = state_em.copy()  # Copy the electromagnetic state
            self.iterations = 0  # Initialise the number of iterations

            # Check whether the maximally allowed number of iterations is exceeded
            while error > self.solving_tolerance:
                if self.iterations > self.iterations_threshold:
                    raise NoConvergenceException('Convergence not obtained by nonlinear solver')

                # System assembly
                system_matrix = self.gen_system_matrix(state_em_iter, saturation=True)

                # Jacobian calculation
                Jac = self.gen_Jac(system_matrix, state_em_iter)

                # Estimation update
                state_em_iter += np.linalg.solve(Jac, -self.nmatmul(system_matrix, state_em_iter) + rhs) * self.relaxation_factor

                # Increment the iteration number
                self.iterations += 1

                # Calculate the error
                error = np.linalg.norm(self.nmatmul(system_matrix, state_em_iter) - rhs, ord=np.inf)

            # Update the electromagnetic state
            state_em_new = state_em_iter

        # Nonlinear solving using inverse Broyden's method
        elif self.solver == 'broyden':
            error = 1e9  # Nonlinear solving error
            state_em_iter = state_em.copy()  # Copy the electromagnetic state
            self.iterations = 0  # Initialise the number of iterations

            # Run the Newton's method the first iteration
            state_em_iter_prev = state_em_iter.copy()

            # System assembly
            system_matrix = self.gen_system_matrix(state_em_iter, saturation=True)

            # Jacobian calculation
            Jac = self.gen_Jac(system_matrix, state_em_iter)

            # A first inverse Jacobian is calculated, to be updated using Broyden's method
            Jac_inv = np.linalg.inv(Jac)

            # Root finding
            root = state_em_iter - self.nmatmul(Jac_inv, self.nmatmul(system_matrix, state_em_iter) - rhs)

            # Estimation update
            state_em_iter = state_em_iter + (root - state_em_iter) * self.relaxation_factor

            # Commence Broyden's method
            while error > self.solving_tolerance:
                # Check whether the maximally allowed number of iterations is exceeded
                if self.iterations > self.iterations_threshold:
                    raise NoConvergenceException('Convergence not obtained by nonlinear solver')

                if self.iterations != 0:
                    # Estimate inverse Jacobian directly
                    system_matrix_prev = system_matrix.copy()
                    # System assembly
                    system_matrix = self.gen_system_matrix(state_em_iter, saturation=True)
                    # Calculate the difference in x over iterations
                    DeltaX = state_em_iter - state_em_iter_prev
                    # Transform the 1D array to a 2D array which can be transposed
                    DeltaX = DeltaX[:, np.newaxis]
                    # Calculate the difference in F over iterations
                    DeltaF = self.nmatmul(system_matrix, state_em_iter) - self.nmatmul(system_matrix_prev, state_em_iter_prev)
                    # Transform the 1D array to a 2D array which can be transposed
                    DeltaF = DeltaF[:, np.newaxis]
                    term1 = (DeltaX - self.nmatmul(Jac_inv, DeltaF)) / (self.nmatmul(np.transpose(DeltaX), self.nmatmul(Jac_inv, DeltaF)))
                    term2 = self.nmatmul(np.transpose(DeltaX), Jac_inv)
                    # Update the inverse Jacobian directly
                    Jac_inv += self.nmatmul(term1, term2)

                    # Log previous values of x
                    state_em_iter_prev = state_em_iter.copy()

                    # Root finding
                    root = state_em_iter - self.nmatmul(Jac_inv, self.nmatmul(system_matrix, state_em_iter) - rhs)

                    # Estimate update
                    state_em_iter += (root - state_em_iter) * self.relaxation_factor

                # Increment the iteration number
                self.iterations += 1

                # Calculate the error
                error = np.linalg.norm(self.nmatmul(system_matrix, state_em_iter) - rhs, ord=np.inf)
            # Update the electromagnetic state
            state_em_new = state_em_iter

        else:
            raise ValueError('Invalid solver name provided')

        # Solve the rotational mechanics submodel
        T_em, F_emx, F_emy = self.virtual_work(T_l, state_em_new)

        # Construct the new system state
        self.state = np.concatenate((
            state_em_new,
            self.state_rotational,
            np.array([T_em]),
            np.array([F_emx]),
            np.array([F_emy])
        ))


class RelaxationTuner:
    """
    Automated tuner for the nonlinear solving relaxation factor (rF) of IMMEC
    """

    def __init__(self, jumpsize: float = 0.1, stepsize: float = 1e-4, initial: float = 0.5, verbose: bool = False):
        """
        :param jumpsize: value to decrease the RF when convergence is not attained
        :param stepsize: value to increase the RF when the system is solved
        :param initial: starting value of the RF
        """
        self.jumpsize = jumpsize
        self.stepsize = stepsize
        self.relaxation = initial
        self.verbose = verbose

        # Initialise the model solved state
        self.solved = False

        # Initialise the history of the RF values
        self.history = []

        # Print the initial RF
        if self.verbose:
            print(f'RelaxationTuner: Starting with {self.relaxation}')

    def step(self):
        """
        Bound, increment the RF and log to history
        """

        if self.relaxation > 1.0:
            self.relaxation = 1.0
            if self.verbose:
                print('RelaxationTuner warning: RF upper bound reached')
        elif self.relaxation < 1.5 * self.jumpsize:
            self.relaxation = 1.5 * self.jumpsize
            if self.verbose:
                print('RelaxationTuner warning: RF lower bound reached')

        self.relaxation += self.stepsize
        self.history.append(self.relaxation)

        self.solved = True

        if self.verbose:
            print(f'RelaxationTuner: Updated to {self.relaxation}')

    def jump(self):
        """
        Decrease the RF when convergence is not reached
        """
        self.relaxation -= self.jumpsize

    def plot(self):
        """
        Plot the history of the RF tuner
        """
        plt.plot(self.history)
        plt.xlabel('Time step')
        plt.ylabel('Relaxation factor')
        plt.title('Relaxation tuner history')
        plt.show()


class HistoryDataLogger:
    """
    Object to log the values of IMMEC over time and postprocess the results
    """

    def __init__(self, model):
        """
        :param model: The IMMEC object to track
        """
        self.model = model

        # Key, indexing range, name, and unit of state quantities
        self.state_information = {'potentials_st': [0,
                                                    model.N_st,
                                                    'Stator tooth magnetic scalar potentials',
                                                    'Ampère'],
                                  'potentials_rot': [model.N_st,
                                                     model.N_rot,
                                                     'Rotor tooth magnetic scalar potentials',
                                                     'Ampère'],
                                  'flux_st_tooth': [model.N_st + model.N_rot,
                                                    model.N_st,
                                                    'Stator tooth magnetic fluxes',
                                                    'Weber'],
                                  'flux_st_yoke': [2 * model.N_st + model.N_rot,
                                                   model.N_st,
                                                   'Stator yoke segment magnetic fluxes',
                                                   'Weber'],
                                  'flux_rot_tooth': [3 * model.N_st + model.N_rot,
                                                     model.N_rot,
                                                     'Rotor tooth magnetic fluxes',
                                                     'Weber'],
                                  'flux_rot_yoke': [3 * model.N_st + 2 * model.N_rot,
                                                    model.N_rot,
                                                    'Rotor tooth magnetic fluxes',
                                                    'Weber'],
                                  'i_st': [3 * model.N_st + 3 * model.N_rot,
                                           3,
                                           'Stator phase currents',
                                           'Ampère'],
                                  'i_rot': [3 * model.N_st + 3 * model.N_rot + 3,
                                            model.N_rot,
                                            'Rotor bar currents',
                                            'Ampère'],
                                  'alpha_rot': [3 * model.N_st + 4 * model.N_rot + 3,
                                                1,
                                                'Rotor rotational acceleration',
                                                'Radians per second squared'],
                                  'omega_rot': [3 * model.N_st + 4 * model.N_rot + 4,
                                                1,
                                                'Rotor rotational speed',
                                                'Radians per second'],
                                  'gamma_rot': [3 * model.N_st + 4 * model.N_rot + 5,
                                                1,
                                                'Rotor rotational position',
                                                'Radians'],
                                  'T_em': [3 * model.N_st + 4 * model.N_rot + 6,
                                           1,
                                           'Electromagnetic torque',
                                           'Newton-meters'],
                                  'F_em': [3 * model.N_st + 4 * model.N_rot + 7,
                                           2,
                                           'Electromagnetic radial force',
                                           'Newtons']}

        # Key, indexing range, name, and unit of input quantities
        self.input_information = {'v_applied': [0,
                                                3,
                                                'Supply voltages',
                                                'Volt'],
                                  'T_l': [3,
                                          1,
                                          'Load torque',
                                          'Newton-meter'],
                                  'ecc': [4,
                                          2,
                                          'Rotor eccentricity',
                                          'Meter']}

        # Key, name, and unit of post-processed quantities
        self.misc_information = {'B_st_tooth': ['Stator tooth magnetic flux densities',
                                                'Tesla'],
                                 'B_st_yoke': ['Stator yoke segment magnetic flux densities',
                                               'Tesla'],
                                 'B_rot_tooth': ['Rotor tooth magnetic flux densities',
                                                 'Tesla'],
                                 'B_rot_yoke': ['Rotor yoke segment magnetic flux densities',
                                                'Tesla'],
                                 'iterations': ['Nonlinear solving iterations',
                                                'Number of iterations']}

        # Define all keys which can be logged
        self.keys = list(self.state_information.keys()) + list(self.input_information.keys()) \
                        + ['time', 'iterations']

        # Initialise an empty dictionary for the physical quantities
        self.quantities = {}

        # Initialise pre-allocation flags
        self.pre_allocated = {key: False for key in self.keys}
        self.pre_allocation_id = 0

    def pre_allocate(self, size: int, quantities_to_preallocate):
        assert type(size) == int

        if type(quantities_to_preallocate) == str:
            quantities_to_preallocate = [quantities_to_preallocate]

        for quantity in quantities_to_preallocate:
            if quantity in self.state_information.keys():
                length = self.state_information[quantity][1]
            elif quantity in self.input_information.keys():
                length = self.input_information[quantity][1]
            elif quantity == 'time':
                length = 1
            else:
                raise ValueError(f'Pre-allocation unsupported for {quantity}')

            self.quantities[quantity] = np.empty((size, length))
            self.pre_allocated[quantity] = True

    def read(self, quantities, inputs: np.ndarray=None):
        """
        Read the current quantities in the motor model state or input
        :param quantities:
        :param inputs:
        :return:
        """
        results = []

        # Turn a single string into a list
        if type(quantities) == str:
            quantities = [quantities]

        for quantity in quantities:
            assert quantity in self.keys

            if quantity in self.state_information.keys():
                value = self.model.state[self.state_information[quantity][0]:
                                         self.state_information[quantity][0] + self.state_information[quantity][1]]
            elif quantity in self.input_information.keys():
                assert inputs is not None, 'For input quantities, the inputs must be provided as an argument'
                value = inputs[self.input_information[quantity][0]:
                               self.input_information[quantity][0] + self.input_information[quantity][1]]
            elif quantity == 'iterations':
                value = self.model.iterations
            else:
                raise ValueError(f'{quantity} not supported for reading')

            results.append(value)

        if len(results) == 1:
            results = results[0]

        return results


    def log(self, time: float, inputs: np.ndarray=None, quantities_to_log='all'):
        """
        Log the simulation data and time
        :param time: continuous time value at log
        :param inputs: IMMEC inputs as [v_input, T_l, e_rot]
        """
        # Populate quantities_to_log when all keys are requested
        if quantities_to_log == 'all':
            quantities_to_log = list(self.state_information.keys()) + list(self.input_information.keys()) \
                                + ['iterations']

        # Turn a single string into a list
        elif type(quantities_to_log) == str:
            quantities_to_log = [quantities_to_log]

        # Ensure that time values are logged
        quantities_to_log.append('time')

        # Record the data
        for quantity in quantities_to_log:
            assert quantity in self.keys

            if quantity in self.state_information.keys():
                value = self.model.state[self.state_information[quantity][0]:
                                         self.state_information[quantity][0] + self.state_information[quantity][1]]
            elif quantity in self.input_information.keys():
                value = inputs[self.input_information[quantity][0]:
                               self.input_information[quantity][0] + self.input_information[quantity][1]]
            elif quantity == 'iterations':
                value = self.model.iterations
            elif quantity == 'time':
                value = time
            else:
                raise ValueError(f'{quantity} not supported for logging')

            if quantity in self.quantities:
                if self.pre_allocated[quantity] == True:
                    self.quantities[quantity][self.pre_allocation_id] = value
                else:
                    self.quantities[quantity] = np.row_stack((self.quantities[quantity], value))
            else:
                self.quantities[quantity] = np.array([value])

        self.pre_allocation_id += 1

    @staticmethod
    def arr_to_list(arr):
        """
        Transform an array to a list of (array) values
        """
        list = []
        for element in arr:
            list.append(element)
        return list

    def postprocess(self):
        """
        Process the data to obtain additional physical quantities
        """
        # Compute the magnetic flux densitites
        if 'flux_st_tooth' in self.quantities.keys():
            B_st_tooth = np.array(self.quantities['flux_st_tooth']) / self.model.A_st_t
            self.quantities.update({'B_st_tooth': B_st_tooth})

        if 'flux_st_yoke' in self.quantities.keys():
            B_st_yoke = np.array(self.quantities['flux_st_yoke']) / self.model.A_st_y
            self.quantities.update({'B_st_yoke': B_st_yoke})

        if 'flux_rot_tooth' in self.quantities.keys():
            B_rot_tooth = np.array(self.quantities['flux_rot_tooth']) / self.model.A_rot_t
            self.quantities.update({'B_rot_tooth': B_rot_tooth})

        if 'flux_rot_yoke' in self.quantities.keys():
            B_rot_yoke = np.array(self.quantities['flux_rot_yoke']) / self.model.A_rot_y
            self.quantities.update({'B_rot_yoke': B_rot_yoke})

    def plot(self, quantities_to_plot):
        """
        Plot one or more quantities
        :param quantities_to_plot:
        string input for single quantity plot corresponding to the self.quantities key.
        list of strings for multiple plots
        'all' to plot all stored quantities
        """
        # Turn a single string into a list
        if type(quantities_to_plot) == str and quantities_to_plot != 'all':
            quantities_to_plot = [quantities_to_plot]

        # Populate the list when all quantities are requested
        if quantities_to_plot == 'all':
            quantities_to_plot = list(self.quantities.keys())
            quantities_to_plot.remove('time')

        quantities_to_plot_copy = quantities_to_plot.copy()
        for quantity in quantities_to_plot_copy:
            if len(self.quantities[quantity]) == 0:
                quantities_to_plot.remove(quantity)

        if len(quantities_to_plot) == 0:
            print('HistoryDataLogger warning: No requested quantities to plot contain logged data')

        # Plot the requested quantities
        for quantity in quantities_to_plot:
            plt.plot(self.quantities['time'], self.quantities[quantity])
            plt.xlabel('Time (s)')
            if quantity in self.state_information.keys():
                plt.ylabel(self.state_information[quantity][3])
                plt.title(self.state_information[quantity][2])
            elif quantity in self.input_information.keys():
                plt.ylabel(self.input_information[quantity][3])
                plt.title(self.input_information[quantity][2])
            elif quantity in self.misc_information.keys():
                plt.ylabel(self.misc_information[quantity][1])
                plt.title(self.misc_information[quantity][0])
            plt.show()

    def save_history(self, path: str = 'IMMEC_history_unnamed'):
        """
        Save the state and input history data locally
        :param path: Local path to save to
        """
        with open(path + '.pkl', 'wb') as file:
            pkl.dump(self.quantities, file)

    def load_history(self, path: str = 'IMMEC_history_unnamed'):
        """
        Load the state and input history data
        :param path: local path
        """
        with open(path + '.pkl', 'rb') as f:
            self.quantities = pkl.load(f)

    def save_state(self, path: str = 'IMMEC_state_unnamed'):
        """
        Save a motor model state
        :param path: local path
        """
        np.save(path, self.model.state)

    def load_state(self, path: str = 'IMMEC_state_unnamed'):
        """
        Load a motor model state
        :param path: local path
        :return:
        """
        self.model.state = np.load(path + '.npy')
