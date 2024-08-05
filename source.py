import copy
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps
import seaborn as sns
import os
from libs import get_custom_library_funcs
from prepare_data import *


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


def read_V_from_directory(path_to_directory: str):
    """
    This function reads the voltage from the directory name, when multiple simulations are in seperate files. TO BE REMOVED
    :param path_to_directory:
    :return:
    """
    # most datafiles have the name IMMEC_history_{voltage}V_1.0sec.pkl, so this function reads the voltage from the
    # directory name, return an array with all voltages from files in this directory
    files = os.listdir(path_to_directory)
    V = []
    for file in files:
        if file.endswith(".pkl") and file.startswith("IMMEC_history"):
            V.append(int(file.split("_")[2][:-1]))
    return np.array(V)


def save_plot_data(
        save_name: str,
        xydata: list,
        title: str,
        xlab, ylab,
        legend=None,
        plot_now=False,
        specs=None,
        sindy_model=None
):
    """
    Saves the plot data in a .pkl file such that it can be plotted later
    :param save_name:
    :param xydata:
    :param title:
    :param xlab:
    :param ylab:
    :param legend:
    :param plot_now:
    :param specs:
    :param sindy_model:
    :return:
    """
    # xydata contains the data to plot, but if multiple axis should be plotted, xy data should be a list of arrays
    # if it is only one x,y then [np.array([x,y])] should be the input
    # create the dictionary to save as is
    pltdata = {
        "title": title,
        "xlab": xlab,
        "ylab": ylab,
        "legend": legend,
        "plots": {},
        "specs": specs,
        "model": sindy_model,  # todo wtf?
    }
    for i, xy_array in enumerate(xydata):
        pltdata["plots"][str(i)] = xy_array
    cwd = os.getcwd()
    save_path = os.path.join(cwd, "plot_data\\", save_name + ".pkl")

    with open(save_path, "wb") as file:
        pkl.dump(pltdata, file)
    if plot_now:
        plot_data(save_path)
    return save_path


def plot_data(path="plotdata.pkl", show=True, figure=True, limits=None):
    # todo adapt such that fourier is plotted too
    suppres_title = False
    if type(path) == str:
        paths = [path]
    else:
        paths = path
        suppres_title = True

    linetypes = ["-", "--", ":"]
    for j, path in enumerate(paths):
        with open(path, "rb") as file:
            data = pkl.load(file)

        if type(data["ylab"]) != str:  # multiple axis
            print("Multiple axis plot detected.")
            print("loglog ax1 and semilogx ax2.")
            # if subplot exist, dont' create a new subplot
            figure = plt.fignum_exists(1)  # check if figure exists
            if not figure:
                fig, ax1 = plt.subplots()

            ax1.set_xlabel(data["xlab"])

            ax1.set_ylabel(data["ylab"][0], color="r")
            ax1.loglog(data["plots"]["0"][:, 0], data["plots"]["0"][:, 1:], "r" + linetypes[j])

            ax2 = ax1.twinx()
            ax2.set_ylabel(data["ylab"][1], color="b")
            ax2.semilogx(data["plots"]["1"][:, 0], data["plots"]["1"][:, 1:], "b" + linetypes[j])

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
            for idx in data["plots"]:
                if specs[int(idx)] is not None:
                    plt.plot(data["plots"][idx][:, 0], data["plots"][idx][:, 1:], specs[int(idx)])
                else:
                    plt.plot(data["plots"][idx][:, 0], data["plots"][idx][:, 1:])

            plt.legend(data["legend"])
            plt.title(data["title"])
    if show:
        plt.show()
    return


def plot_coefs(model):
    coefs = model.coefficients()
    featurenames = model.feature_names()
    # plot coefs of the model, based on the code provided by pysindy:
    # https://pysindy.readthedocs.io/en/latest/examples/7_plasma_examples/example.html
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


def plot_coefs2(model, normalize_values=False, show=False, log=False):
    # todo fix for torque
    xticknames = model.get_feature_names()
    for i in range(len(xticknames)):
        xticknames[i] = xticknames[i]
    plt.figure(figsize=(len(xticknames), 4))
    colors = ["b", "r", "k"]
    coefs = copy.deepcopy(model.coefficients())

    if normalize_values:
        raise NotImplementedError("This function is not implemented yet.")  # todo, unsure how to

    if log:
        plt.yscale("log", base=10)
        coefs = np.abs(coefs)

    for i in range(coefs.shape[0]):
        values = coefs[i, :].T
        values[values == 0] = np.nan  # don't plot zero
        plt.scatter(
            np.arange(0, len(xticknames), 1),
            values,
            color=colors[i],
            label=r"Equation for $\dot{" + xticknames[i + 1].strip("$") + "}$",
        )

    plt.grid(True)
    plt.xticks(range(len(xticknames)), xticknames, rotation=90)

    plt.legend()
    if show:
        plt.show()
    return


def save_model(model, name, libstr):
    print("Saving model")
    path = "C:/Users/emmav/PycharmProjects/SINDY_project/models/" + name + ".pkl"
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


def load_model(name):
    path = "C:/Users/emmav/PycharmProjects/SINDY_project/models/" + name + ".pkl"
    with open(path, "rb") as file:
        model_data = pkl.load(file)

    # initialize pysindy model

    lib = get_custom_library_funcs(model_data["library"])

    new_model = ps.SINDy(optimizer=None, feature_names=model_data["features"], feature_library=lib)

    x_shape, u_shape, xdot_shape = model_data["shapes"]
    new_model.fit(np.zeros(x_shape), u=np.zeros(u_shape), t=None, x_dot=np.zeros(xdot_shape))

    new_model.optimizer.coef_ = model_data["coefs"]

    return new_model


def plot_immec_data(path, simulation_number=None, title=None):
    # if path ends with pkl, load as pkl file
    if path.endswith(".pkl"):
        with open(path, "rb") as file:
            dataset = pkl.load(file)
    else:
        dataset = dict(np.load(path))

    d_air = 0.000477  # for the Cantoni motor

    if simulation_number is None:  # testfile
        plt.subplot(2, 3, 1)
        plt.title("omega_rot"), plt.xlabel("time (s)"), plt.ylabel("rad/s")
        plt.plot(dataset["time"], dataset["omega_rot"])

        plt.subplot(2, 3, 2)
        plt.title("i_st in dq0"), plt.xlabel("time (s)"), plt.ylabel("A")  # debug
        plt.plot(dataset["time"], reference_abc_to_dq0(dataset["i_st"]))

        plt.subplot(2, 3, 3)
        plt.title("T_l and T_em"), plt.xlabel("time (s)"), plt.ylabel("Nm")
        plt.plot(dataset["time"], dataset["T_em"])

        plt.plot(dataset["time"], dataset["T_l"], "k--")
        plt.legend(["T_em", "T_l"])

        plt.subplot(2, 3, 5)
        plt.title("Applied line Voltages"), plt.xlabel("time (s)"), plt.ylabel("V")

        plt.plot(dataset["time"], dataset['v_applied'])
        # debug

        plt.subplot(2, 3, 4)
        plt.title("UMP"), plt.xlabel("time (s)"), plt.ylabel("N")
        plt.plot(dataset["time"], dataset["F_em"])

        plt.subplot(2, 3, 6)
        plt.title("Eccentricity"), plt.xlabel("time (s)"), plt.ylabel("% airgap")
        plt.plot(dataset["time"], dataset["ecc"] / d_air)
    else:  # train file
        plt.subplot(2, 3, 1)
        plt.title("omega_rot"), plt.xlabel("time (s)"), plt.ylabel("rad/s")

        plt.plot(dataset["time"][:, 0, simulation_number], dataset["omega_rot"][:, 0, simulation_number])

        plt.subplot(2, 3, 2)
        plt.title("i_st in dq0"), plt.xlabel("time (s)"), plt.ylabel("A")  # debug
        plt.plot(dataset["time"][:, 0, simulation_number],
                 reference_abc_to_dq0(dataset["i_st"][:, :, simulation_number]))

        plt.subplot(2, 3, 3)
        plt.title("T_l and T_em"), plt.xlabel("time (s)"), plt.ylabel("Nm")
        # plt.title("T_em"), plt.xlabel("time (s)")
        plt.plot(dataset["time"][:, 0, simulation_number], dataset["T_em"][:, 0, simulation_number])

        # plt.subplot(2, 3, 3)
        # plt.title("T_l"), plt.xlabel("time (s)")
        plt.plot(dataset["time"][:, 0, simulation_number], dataset["T_l"][:, 0, simulation_number], 'k--')
        plt.legend(["T_em", "T_l"])

        plt.subplot(2, 3, 5)
        plt.title("Applied line Voltages"), plt.xlabel("time (s)"), plt.ylabel("V")
        plt.plot(dataset["time"][:, 0, simulation_number], dataset["v_applied"][:, :, simulation_number])

        plt.subplot(2, 3, 4)
        plt.title("UMP"), plt.xlabel("time (s)"), plt.ylabel("N")
        plt.plot(dataset["time"][:, 0, simulation_number], dataset["F_em"][:, :, simulation_number])

        plt.subplot(2, 3, 6)
        plt.title("Eccentricity"), plt.xlabel("time (s)"), plt.ylabel("% airgap")
        plt.plot(dataset["time"][:, 0, simulation_number], dataset["ecc"][:, :, simulation_number] / d_air)
        plt.legend(["x", "y"])
    if title is not None:
        plt.suptitle(title)
    # Add padding so title and labels dont overlap
    plt.tight_layout()
    plt.show()
    return


def plot_everything(path_to_directory):
    files = os.listdir(path_to_directory)
    for file in files:
        if file.endswith(".pkl"):
            path = os.path.join(path_to_directory, file)
            plot_data(path, show=False)
    plt.show()
    return


def plot_fourier(reference, result, dt, tmax):
    def fun(w, n, s):
        # Perform FFT
        fft = np.fft.fft(w)
        p = np.abs(fft / n)[:int(n / 2 + 1)]
        p[1:-1] = 2 * p[1:-1]
        freq = s * np.arange(0, p.shape[0]) / n
        return p, freq

    n_fft = tmax / dt
    sampling_freq = 1 / dt

    ref, f1 = fun(reference, n_fft, sampling_freq)
    res, f2 = fun(result, n_fft, sampling_freq)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.semilogy(f1, ref, label="Current Signal")
    plt.semilogy(f2, res, ':',label="Current Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Reference Signal FFT")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.semilogy(f2, np.abs(res-ref), label="FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Delta of FFT")
    plt.grid()

    plt.tight_layout()
    plt.show()

    return


def test_plot_fourier():
    tmax = 5.0
    dt = 4e-5
    t = np.arange(0, tmax, dt)
    x = np.sin(2 * np.pi * 50 * t)
    y = np.sin(2*2000*np.pi*t)
    plot_fourier(x, y, dt=dt, tmax=tmax)

    return