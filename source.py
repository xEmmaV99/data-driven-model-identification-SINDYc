import copy
from datetime import datetime
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pysindy as ps

try:
    import seaborn as sns
except ImportError:
    pass

from libs import get_custom_library_funcs
from prepare_data import *
from scipy.integrate import solve_ivp


def show_data_keys(path_to_data_logger: str):
    """
    Shows the keys of the data logger from a .pkl file
    :param path_to_data_logger: str with the path to the .pkl file
    :return:
    """
    with open(path_to_data_logger, "rb") as file:
        data_logger = pkl.load(file)
    print(data_logger.keys())
    return


def check_training_data(path_to_data_logger: str, keys_to_plot_list: list = None):
    """
    Plots the data from the data logger, optional list of keys to plot
    :param path_to_data_logger: str with the path to the .pkl file
    :param keys_to_plot_list: list of str with the keys to plot
    :return:
    """
    if keys_to_plot_list is None:
        keys_to_plot_list = ["i_st"]
    with open(path_to_data_logger, "rb") as file:
        data_logger = pkl.load(file)
    for key in keys_to_plot_list:
        plt.figure()
        plt.plot(data_logger[key])
    plt.show()
    return


def save_plot_data(
        save_name: str,
        xydata: list,
        title: str,
        xlab: str,
        ylab,
        legend: list = None,
        plot_now: bool = False,
        specs=None,
):
    """
    Saves the plot data in a .pkl file such that it can be plotted later
    :param save_name: str used to save the data with
    :param xydata: list of np.array, containing the data to plot. If only one x, y then [np.array([x,y])].
    Here, x is assumed to be one dimensional, x.shape = (n,1) and y.shape = (n, k) with k the number of lines to be plotted.
    If multiple axis should be plotted, then [np.array([x1,y1]), np.array([x2,y2]), ...]
    :param title: Title of the plot
    :param xlab: x label
    :param ylab: y label, can be a list of two y lables if multiple axis
    :param legend: list of str containing the legend entries
    :param plot_now: True if plt.show is called
    :param specs: list of str containing the specifications, for example ["k--", "b", "r"] for the color and linestypes.
    The number of elements in this list should be equal to 'k' from y.shape = (n,k)
    :return:
    """
    # create the dictionary to save
    pltdata = {
        "title": title,
        "xlab": xlab,
        "ylab": ylab,
        "legend": legend,
        "plots": {},
        "specs": specs,
    }
    for i, xy_array in enumerate(xydata):
        pltdata["plots"][str(i)] = xy_array
    cwd = os.getcwd()
    save_path = os.path.join(cwd, "plot_data\\", save_name + get_date() + ".pkl")

    with open(save_path, "wb") as file:
        pkl.dump(pltdata, file)
    if plot_now:
        plot_data(save_path)
    return save_path


def plot_data(path="plotdata.pkl", show=False, limits=None):
    """
    Plots the data from a .pkl file
    :param path: str or list of str containing the path(s) to the plot data file(s)
    :param show: True if plt.show is called
    :param limits: list of list of floats, to manually set the limits, can be np.array too.
    Example: [[0, 5],[-10,10]] to set ylim [0,5] and ylim [-10,10], for a plot with multiple axis.
    :return:
    """
    suppres_title = False
    if type(path) == str:  # only one path
        paths = [path]
    else:  # multiple paths, suppress title otherwise multiple titles are shown in the same plot
        paths = path
        suppres_title = True

    linetypes = ["-", "--", ":"]
    for j, path in enumerate(paths):
        with open(path, "rb") as file:
            data = pkl.load(file)

        # multiple axis, !!! This was mainly used for the MSE/SPAR plot before the pareto plots were added
        # This is probably not used anymore
        if type(data["ylab"]) != str:
            print("Multiple axis plot detected.")
            print("loglog ax1 and semilogx ax2.")
            # if subplot exist, dont' create a new subplot
            figure = plt.fignum_exists(1)  # check if figure exists
            if not figure:
                fig, ax1 = plt.subplots()

            ax1.set_xlabel(data["xlab"])

            ax1.set_ylabel(data["ylab"][0], color="r")
            ax1.loglog(
                data["plots"]["0"][:, 0], data["plots"]["0"][:, 1:], "r" + linetypes[j]
            )

            ax2 = ax1.twinx()
            ax2.set_ylabel(data["ylab"][1], color="b")
            ax2.semilogx(
                data["plots"]["1"][:, 0], data["plots"]["1"][:, 1:], "b" + linetypes[j]
            )

            if not suppres_title:
                plt.title(data["title"])
            if limits == None:
                fig.tight_layout()
            else:
                ax1.set_ylim(limits[0])
                ax2.set_ylim(limits[1])

        else:  # only one axis
            plt.figure()
            plt.xlabel(data["xlab"]), plt.ylabel(data["ylab"])
            specs = data["specs"]
            # shape should be (2, t), where t is the number of time points and 2 number of solution
            # for ump, it is saved as [x, y, reference]
            # or for torque [x, ref, simplified model]
            # currents [di, ref]
            # wcoe [x,ref]

            # Gather y-values for the fourier plot later.
            # yid is used to select 1,2 or 3 elements (for example, currents have 3 (dq0) and ump has 2 (xy))
            if data["plots"]["0"].shape[-1] == 4:  # currents
                yvalues = np.hstack(
                    (data["plots"]["0"][:, 1:], data["plots"]["1"][:, 1:])
                )
                yid = 3
            elif len(data["plots"].keys()) == 2:  # this is currents or wcoe
                yvalues = np.hstack(
                    (data["plots"]["0"][:, 1:], data["plots"]["1"][:, 1:])
                )
                yid = 1
            elif data["plots"]["2"].shape[-1] == 2:  # torque
                yvalues = np.hstack(
                    (data["plots"]["0"][:, 1:], data["plots"]["1"][:, 1:])
                )
                yid = 1
            else:
                yvalues = np.hstack(
                    (
                        np.hstack(
                            (data["plots"]["0"][:, 1:], data["plots"]["1"][:, 1:])
                        ),
                        data["plots"]["2"][:, 1:],
                    )
                )
                yid = 2

            # plot the data
            for idx in data["plots"]:
                if specs is not None and specs[int(idx)] is not None:
                    plt.plot(
                        data["plots"][idx][:, 0],
                        data["plots"][idx][:, 1:],
                        specs[int(idx)],
                    )
                else:
                    plt.plot(data["plots"][idx][:, 0], data["plots"][idx][:, 1:])

            plt.legend(data["legend"])
            plt.title(data["title"])
            if limits is not None:
                plt.ylim(limits[0])

            # plot the fourier data, reference, predicted
            plot_fourier(
                yvalues[:, yid:],
                yvalues[:, :yid],
                dt=1e-4,
                tmax=data["plots"]["0"][-1, 0],
                show=False,
            )
    if show:
        plt.show()
    return


def plot_coefs(model):
    """
    Creates a heatplot of the coefficients of the model, see
    https://pysindy.readthedocs.io/en/latest/examples/7_plasma_examples/example.html

    :param model: a model instance
    :return:
    """
    coefs = model.coefficients()
    featurenames = model.feature_names()
    input_features = [rf"$\dot x_{k}$" for k in range(coefs.shape[0])]
    if featurenames == None:
        input_names = [rf"$x_{k}$" for k in range(coefs.shape[1])]
    else:
        input_names = featurenames

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        fig, ax = plt.subplots(1, 1)
        max_magnitude = np.max(np.abs(coefs))
        heatmap_args = {
            "xticklabels": input_features,
            "yticklabels": input_names,
            "center": 0.0,
            # "cmap": "RdBu_r",
            "cmap": sns.color_palette("vlag", n_colors=20),
            "ax": ax,
            "linewidths": 0.1,
            "linecolor": "whitesmoke",
        }

        sns.heatmap(coefs[:, : len(input_names)].T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)
    return


def plot_coefs2(model, show=False, log=False, type='currents'):
    """
    Plot the coefficients of a model, but on an axis, based on
    https://pysindy.readthedocs.io/en/latest/examples/1_feature_overview/example.html
    :param model: a model instance
    :param show: if True, plt.show() is called
    :param log: if True, uses logscale for the yscale
    :param type: the type of model, either currents, torque, ump or wcoe
    :return:
    """
    model.print()
    print("Sparsity: ", np.count_nonzero(model.coefficients()))
    xticknames = (
        model.get_feature_names()
    )
    for i in range(len(xticknames)):
        xticknames[i] = xticknames[i]
    plt.figure(figsize=(len(xticknames), 4))
    colors = ["b", "r", "k"]
    coefs = copy.deepcopy(
        model.coefficients()
    )  # copy such that they do not get overwritten

    if log:
        plt.yscale("log", base=10)
        coefs = np.abs(coefs)

    if type.lower() == 'currents':
        names = [r"$\dot{" + xticknames[i].strip("$") + "}$" for i in range(coefs.shape[0])]

    elif type.lower() == 'torque':
        names = [r"$T_{em}$"]

    elif type.lower() == 'ump':
        names = [r"$UMP_x$", r"$UMP_y$"]

    for i in range(coefs.shape[0]):
        values = coefs[i, :].T
        values[values == 0] = np.nan  # don't plot zero
        plt.scatter(
            np.arange(0, len(xticknames), 1),
            values,
            color=colors[i],
            label=r"Equation for " + names[i],
        )

    plt.grid(True)
    plt.xticks(range(len(xticknames)), [r'$' + n + '$' for n in xticknames], rotation=90)

    plt.legend()
    if show:
        plt.show()
    return


def get_date():
    """
    Returns the current date, used for saving data such that they don't get overwritten by accident
    :return: str containing the current date
    """
    now = datetime.now()
    return now.strftime("%m-%d_%H-%M-%S")


def save_model(model, name: str, libstr: str):
    """
    Function to save a model as pkl file. Note that a pysindy library cannot simply be pikled
    :param model: a model instance, to be saved
    :param name: name used for saving the model
    :param libstr: string of the libary used to generate the model
    :return:
    """
    print("Saving model")
    saving_name = name + get_date() + ".pkl"
    path = os.path.join(
        os.getcwd(), "models", saving_name
    )  # add date to avoid overwriting

    x = model.n_features_in_ - model.n_control_features_
    u = model.n_control_features_

    lib = {
        "coefs": model.coefficients(),
        "features": model.feature_names,
        "library": libstr,
        "shapes": [(1, x), (1, u), (1, x)],
    }
    with open(path, "wb") as file:
        pkl.dump(lib, file)
    return saving_name


def load_model(name: str):
    """
    Load the model from a .pkl file
    :param name: name of the model
    :return: a model
    """
    # load .pkl file
    if not name.endswith('.pkl'):
        name = name + '.pkl'
    path = os.path.join(os.getcwd(), "models", name)
    with open(path, "rb") as file:
        model_data = pkl.load(file)

    # initialize pysindy model
    x_shape, u_shape, xdot_shape = model_data["shapes"]
    lib = get_custom_library_funcs(
        model_data["library"], u_shape[1] + x_shape[1]
    )
    new_model = ps.SINDy(
        optimizer=None, feature_names=model_data["features"], feature_library=lib
    )

    # trick SINDy to fit a model
    new_model.fit(
        np.zeros(x_shape), u=np.zeros(u_shape), t=None, x_dot=np.zeros(xdot_shape)
    )

    # overwrite the coefficients of the 'fitted' model :)
    new_model.optimizer.coef_ = model_data["coefs"]

    return new_model


def set_plot_defaults():
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['text.usetex'] = True
    _, ax = plt.subplots(figsize=(3.5, 3.5))
    plt.yticks(fontsize=7)
    plt.xticks(fontsize=7)
    return ax


def plot_immec_data(path: str, simulation_number: int = None, title: str = None):
    """
    Plot the trainig or test data from a file
    :param path: path to the data
    :param simulation_number: if trainingdata, provide a simulation number
    :param title: string for the suptitle
    :return:
    """
    # if path ends with pkl, load as pkl file
    if path.endswith(".pkl"):
        with open(path, "rb") as file:
            dataset = pkl.load(file)
    else:
        dataset = dict(np.load(path))

    print('d_air hardcoded for Cantoni')
    d_air = 0.000477

    rows, cols = 4, 2
    plt.subplots(4, 2, figsize=(8, 20))

    if simulation_number is None:  # testfile
        plt.subplot(rows, cols, 1)
        plt.title("omega_rot"), plt.xlabel("time (s)"), plt.ylabel("rad/s")
        plt.plot(dataset["time"], dataset["omega_rot"])

        plt.subplot(rows, cols, 2)
        plt.title("i_st in dq0"), plt.xlabel("time (s)"), plt.ylabel("A")
        plt.plot(dataset["time"], reference_abc_to_dq0(dataset["i_st"]))

        plt.subplot(rows, cols, 3)
        plt.title("T_l and T_em"), plt.xlabel("time (s)"), plt.ylabel("Nm")
        plt.plot(dataset["time"], dataset["T_em"])

        plt.plot(dataset["time"], dataset["T_l"], "k--")
        plt.legend(["T_em", "T_l"])

        plt.subplot(rows, cols, 5)
        plt.title("Applied line Voltages"), plt.xlabel("time (s)"), plt.ylabel("V")
        plt.plot(dataset["time"], dataset["v_applied"])

        plt.subplot(rows, cols, 4)
        plt.title("UMP"), plt.xlabel("time (s)"), plt.ylabel("N")
        plt.plot(dataset["time"], dataset["F_em"])

        plt.subplot(rows, cols, 6)
        plt.title("Eccentricity"), plt.xlabel("time (s)"), plt.ylabel("% airgap")
        plt.plot(dataset["time"], dataset["ecc"] / d_air)
        # if wcoe not in keys, don't do the next plot

        if "wcoe" in dataset.keys():
            plt.subplot(rows, cols, 7)
            plt.title("Magnetic co-energy"), plt.xlabel("time (s)"), plt.ylabel("J")
            plt.plot(dataset["time"], dataset["wcoe"])

        plt.subplot(rows, cols, 8)
        plt.title("Eccentricity"), plt.xlabel("r_x"), plt.ylabel("r_y")
        plt.plot(
            dataset["ecc"][:, 0, simulation_number] / d_air,
            dataset["ecc"][:, 1, simulation_number] / d_air,
        )

    else:  # train file
        plt.subplot(rows, cols, 1)
        plt.title("omega_rot"), plt.xlabel("time (s)"), plt.ylabel("rad/s")

        plt.plot(
            dataset["time"][:, 0, simulation_number],
            dataset["omega_rot"][:, 0, simulation_number],
        )

        plt.subplot(rows, cols, 2)
        plt.title("i_st in dq0"), plt.xlabel("time (s)"), plt.ylabel("A")
        plt.plot(
            dataset["time"][:, 0, simulation_number],
            reference_abc_to_dq0(dataset["i_st"][:, :, simulation_number]),
        )

        plt.subplot(rows, cols, 3)
        plt.title("T_l and T_em"), plt.xlabel("time (s)"), plt.ylabel("Nm")
        plt.plot(
            dataset["time"][:, 0, simulation_number],
            dataset["T_em"][:, 0, simulation_number],
        )
        plt.plot(
            dataset["time"][:, 0, simulation_number],
            dataset["T_l"][:, 0, simulation_number],
            "k--",
        )
        plt.legend(["T_em", "T_l"])

        plt.subplot(rows, cols, 5)
        plt.title("Applied line Voltages"), plt.xlabel("time (s)"), plt.ylabel("V")
        plt.plot(
            dataset["time"][:, 0, simulation_number],
            dataset["v_applied"][:, :, simulation_number],
        )

        plt.subplot(rows, cols, 4)
        plt.title("UMP"), plt.xlabel("time (s)"), plt.ylabel("N")
        plt.plot(
            dataset["time"][:, 0, simulation_number],
            dataset["F_em"][:, :, simulation_number],
        )

        plt.subplot(rows, cols, 6)
        plt.title("Eccentricity"), plt.xlabel("time (s)"), plt.ylabel("% airgap")
        plt.plot(
            dataset["time"][:, 0, simulation_number],
            dataset["ecc"][:, :, simulation_number] / d_air,
        )
        plt.legend(["x", "y"])

        plt.subplot(rows, cols, 8)
        plt.title("Eccentricity"), plt.xlabel("r_x"), plt.ylabel("r_y")
        plt.plot(
            dataset["ecc"][:, 0, simulation_number] / d_air,
            dataset["ecc"][:, 1, simulation_number] / d_air,
        )

        if "wcoe" in dataset.keys():
            plt.subplot(rows, cols, 7)
            plt.title("Magnetic coenergy"), plt.xlabel("time (s)"), plt.ylabel("J")
            plt.plot(
                dataset["time"][:, 0, simulation_number],
                dataset["wcoe"][:, :, simulation_number],
            )

    if title is not None:
        plt.suptitle(title)
    # add padding so title and labels do not overlap
    plt.tight_layout()
    plt.show()
    return


def plot_everything(path_to_directory: str):
    """
    Plots every file from a folder
    :param path_to_directory:  path to a folder
    :return:
    """
    files = os.listdir(path_to_directory)
    for file in files:
        print(file)
        if file.endswith(".pkl"):
            path = os.path.join(path_to_directory, file)
            plot_data(path, show=False)
    plt.show()
    return


def plot_fourier(reference, result, dt, tmax, leg=None, show=True):
    """
    Plots the fourier spectrum of a signal and its reference
    :param reference: np.array of shape (n,k) with k the number of pulses, n is the number of timesteps
    :param result: np.array of shape (n,k) with k the number of pulses with n the number of timesteps
    :param dt: size of the time step
    :param tmax: maximum simulation time
    :param leg: Contains the entries of the legend, if None, no legend is plotted
    :param show: if True, plt.show() is called
    :return:
    """

    def transform_fft(w, n, s):
        # Perform FFT
        fft = np.fft.fft(w, axis=0)
        p = np.abs(fft / n)[: int(n / 2 + 1)]
        p[1:-1] = 2 * p[1:-1]
        freq = s * np.arange(0, p.shape[0]) / n
        if np.ndim(p) == 1:
            return p[:, np.newaxis], freq
        return p, freq

    # cols = [['tab:blue','tab:red','tab:green'], ['tab:cyan','tab:orange','tab:olive']]
    cols = [["C0", "C3", "C2"], ["C9", "C1", "C8"]]  # line colors

    n_fft = tmax / dt
    sampling_freq = 1 / dt

    ref, f1 = transform_fft(reference, n_fft, sampling_freq)
    res, f2 = transform_fft(result, n_fft, sampling_freq)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    for line in range(ref.shape[1]):
        plt.semilogy(f1, ref[:, line], cols[0][line], label="Reference")
        plt.semilogy(f2, res[:, line], cols[1][line] + "--", label="Predicted")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Reference Signal FFT")
    plt.grid()
    if leg is not None:
        plt.legend(leg)
    else:
        plt.legend()

    plt.subplot(2, 1, 2)
    d = np.abs(res - ref)
    for line in range(ref.shape[1]):
        plt.semilogy(f2, d[:, line], cols[0][line])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Delta of FFT")
    plt.grid()

    plt.tight_layout()
    if show:
        plt.show()

    return


def test_plot_fourier():
    """
    Simple test function to check if plot_fourier works
    :return:
    """
    tmax = 5.0
    dt = 4e-5
    t = np.arange(0, tmax, dt)
    x = np.sin(2 * np.pi * 50 * t)
    y = np.sin(2 * 2000 * np.pi * t)
    plot_fourier(x, y, dt=dt, tmax=tmax)
    return


def plot_tiled_curr(datalist, save_name="test_c", show =False):
    # 7.16 inch for double column
    set_plot_defaults()
    plt.close()
    #_, ax = plt.subplots(2, 3, sharex=True,sharey = 'row',figsize=(7.16 , 7.16/2)) #0.1 for legend
    _, ax = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(7.16, 7.16 / 2))  # 0.1 for legend
    [ax[1,j].set_xlabel(r'$t$ ($s$)') for j in range(3)] # share label
    ax[0,0].set_ylabel(r'$\frac{\partial i_d}{\partial t}$ ($A/s$)')
    ax[1, 0].set_ylabel(r'$i_d$ ($A$)')
    xlim = (0,5)
    #xlim = (3.2375, 3.2525)
    #plt.xlim(xlim)

    for i, path in enumerate(datalist):
        with open(path, "rb") as file:
            data = pkl.load(file)
        # only plot for  the values from data['plots'][idx_str][:,0] in xlim0
        # define time_indices as the indices of the values from data['plots'][idx_str][:,0] in xlim0
        time_indices = np.where((data['plots']['0'][:, 0] >= xlim[0]) & (data['plots']['0'][:, 0] <= xlim[1]))[0]

        for idx_str in data["plots"]:
            ax[i // 3, i%3].plot(data["plots"][idx_str][time_indices, 0],
                                 data["plots"][idx_str][time_indices, 1],
                                 linestyle = ['-',':'][int(idx_str)],
                                 c = ['#1f77b4','k'][int(idx_str)]) #only plot 1 component
    #ax[0, 0].set_ylim((-.1,1100.))
    #ax[1, 0].set_ylim((-4,4))

    #plot every other x label
    for i in range(3):
        ax[1,i].xaxis.set_major_locator(plt.MaxNLocator(4))

    # Put a legend below middle plot
    # tight layout but leave space between the rows
    plt.tight_layout(rect=(0., 0.04, 1., 1.))
    #plt.subplots_adjust(hspace=0.3)

    #ax[0, 1].legend([r"$\frac{\partial i_d}{\partial t}$ predicted", r"$\frac{\partial i_d}{\partial t}$ reference"],loc='upper center', bbox_to_anchor=(0.5, -.05), ncol=5)
    #ax[1, 1].legend([r"$i_d$ predicted", r"$i_d$ reference"],loc='upper center', bbox_to_anchor=(0.5, -.25), ncol=5)
    ax[1, 1].legend([r"Predicted", r"Reference"], loc='upper center', bbox_to_anchor=(0.5, -.25), ncol=5)

    plt.savefig('pdfs//' + save_name + '.pdf', dpi=600.0)
    if show:
        plt.show()
    return

def plot_tiled_TF(datalist, save_name = "test_TF", show=False):

    # 7.16 inch for double column
    set_plot_defaults()
    plt.close()
    #fig, ax = plt.subplots(2, 3,sharex='row', figsize=(7.16 , 7.16/2+0.6))
    fig, ax = plt.subplots(2, 3, sharex=False, figsize=(7.16, 7.16 / 2 + 0.6))
    [ax[1,j].set_xlabel(r'$t$ ($s$)') for j in range(3)]
    [ax[0,j].set_xlabel(r'$t$ ($s$)') for j in range(3)]
    ax[0,0].set_ylabel(r'Torque ($Nm$)')
    ax[1, 0].set_ylabel(r'Unbalanced magnetic pull ($N$)')
    xlim = (0,5)
    #xlim = (2.550, 2.565)
    #ax[0,0].set_xlim(xlim)

    for i, path in enumerate(datalist[:3]): # TORQUE
        with open(path, "rb") as file:
            data = pkl.load(file)

        # only plot what is in xlim
        time_indices = np.where((data['plots']['0'][:, 0] >= xlim[0]) & (data['plots']['0'][:, 0] <= xlim[1]))[0]
        for idx_str in data["plots"]: # Torque has 3; torque, ref and Alternative. For UMP x and y comp should be plotted
            ax[0, i%3].plot(data["plots"][idx_str][time_indices, 0], data["plots"][idx_str][time_indices, 1], linestyle = ['-',':', '--'][int(idx_str)], c = ['#1f77b4','k','#2ca02c'][int(idx_str)])

    #xlim = (2.545, 2.570)
    #ax[1, 0].set_xlim(xlim)
    for i, path in enumerate(datalist[3:]): #UMP
        with open(path, "rb") as file:
            data = pkl.load(file)

        # only plot what is in xlim
        time_indices = np.where((data['plots']['0'][:, 0] >= xlim[0]) & (data['plots']['0'][:, 0] <= xlim[1]))[0]
        for idx_str in data["plots"]: # fx fy ref
            ax[1, i%3].plot(data["plots"][idx_str][time_indices, 0], data["plots"][idx_str][time_indices, 1:], linestyle = ['-','-',':'][int(idx_str)], c = ['#17becf','#ff7f0e', 'b'][int(idx_str)])
        ax[1, i%3].lines[-1].set_color('r') # set last one to red

    ####plt.tight_layout()
    # ax[0, 0].set_ylim((1.25,1.5))
    # ax[0, 1].set_ylim((2.65,2.95))
    # ax[0, 2].set_ylim((0.0,.35))
    # ax[1, 0].set_ylim((-3.,3.))
    # ax[1, 1].set_ylim((0.,250.))
    # ax[1, 2].set_ylim((-400.,400.))


    # tight layout but leave space between the rows
    plt.tight_layout(rect=(0.,0.04,1.,1.))
    plt.subplots_adjust(hspace=0.5)

    # legend below middle plot
    l1 = ax[0, 1].legend([r"$T_{\mathrm{pred}}$",r"$T_{\mathrm{ref}}$",r"$T_{\mathrm{approx}}$"],loc='upper center', bbox_to_anchor=(0.5, -.25), ncol=5)
    l2 = ax[1, 1].legend([r"UMP$_{x, \mathrm{pred}}$",r"UMP$_{y, \mathrm{pred}}$",r"UMP$_{x, \mathrm{ref}}$", r"UMP$_{y, \mathrm{ref}}$"],
                         loc='upper center', bbox_to_anchor=(0.5, -.25), ncol=5)

    # draw empty colors, all legend below the plots
    '''
    plt.tight_layout()
    lab = [r"$T_{\mathrm{pred}}$",r"$T_{\mathrm{ref}}$",r"$T_{\mathrm{approx}}$", r"UMP$_{x, \mathrm{pred}}$",r"UMP$_{y, \mathrm{pred}}$",r"UMP$_{x, \mathrm{ref}}$", r"UMP$_{y, \mathrm{ref}}$"]
    lines = []
    cols = ['#1f77b4','k','#2ca02c', '#17becf','#ff7f0e', 'b', 'r']
    for j, col in enumerate(cols):
        lines.append(plt.Line2D([0], [0], color=col, label = lab[j]))

    ax[1,1].legend(handles = lines, loc = 'upper center',
                    bbox_to_anchor=(0.5, -0.2), ncol=8)
    '''
    plt.savefig('pdfs//' + save_name + '.pdf', dpi=600.0)
    if show:
        plt.show()
    return



def model_simulate(x0: np.array, u: np.array, model, t: np.array):
    """
    Implementation of model.simulate, or at least they should be identical
    :param x0: start value (1, k) with k the number of equations (usually 3)
    :param u: control parameters, shape (n, k) with k the number of control parameters
    :param model: pysindy model instance
    :param t: time to simulate for
    :return:
    """
    x = np.zeros((u.shape[0], model.n_features_in_ - model.n_control_features_))
    x[0] = x0
    u_fun = interp1d(t, u, axis=0, kind="cubic", fill_value="extrapolate")

    def rhs(t, x):
        return model.predict(x[np.newaxis, :], u_fun(t))[0]

    return solve_ivp(
        rhs, np.array([t[0], t[-1]]), np.array([0, 0, 0]), method="RK45", t_eval=t
    )
