import os.path

from optimize_parameters import plot_optuna_data
from source import *
from train_model_source import simulate_model

process = 1
coefs = False

## 1 linear model currents
if process == 1:
    testdata = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
    plot_optuna_data('currentsLinear-specific-optuna-study')
    models = ["linear_example_new_1_currents", "linear_example_2_currents", "linear_example_3_currents"]
    pref = "currents_linear\\"
    pltdata_present = False
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='currents', do_time_simulation=False,
                                       show=False)
            plot_fourier(test, sim, dt=5e-5, tmax=5.0)
    else:
        # plot modes coefs:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)

        path = os.path.join(os.getcwd(), 'plot_data', 'currents_linear')
        plot_everything(path)

## 2 linear model torque
elif process == 2:
    testdata = os.path.join(os.getcwd(), 'test-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
    testdata = os.path.join(os.getcwd(), 'test-data', '07-29', 'IMMEC_0ecc_3.0sec.npz')

    plot_optuna_data("torqueLinear-optuna-study")
    #models = ['torque_linear', 'torque_linear_2', 'torque_linear_3']
    models = ['torque_linear_4']
    pref = "torque_linear\\"
    pltdata_present = False
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='torque', show=False)
            plot_fourier(test, sim, dt=5e-5, tmax=5.0)
    else:
        # plot modes coefs:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)
        path = os.path.join(os.getcwd(), 'plot_data', 'torque_linear')
        plot_everything(path)

## 3 nonlinear model currents
elif process == 3:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_nonlin_0ecc_5.0sec.npz')
    plot_optuna_data('currentsNonlinear-extra_models-optuna-study')
    models = ["example_A_currents", "example_B_currents"]
    pref = "currents_nonlinear\\"
    pltdata_present = True
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='currents', do_time_simulation=True,
                                       show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)
    else:
        # plot modes coefs:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)
        path = os.path.join(os.getcwd(), 'plot_data', 'currents_nonlinear')
        plot_everything(path)

## 4 nonlinear model torque
elif process == 4:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-07', 'IMMEC_nonlin_0ecc_5.0sec.npz')
    plot_optuna_data('torqueNonlinear-extra_models-optuna-study', dirs='torque_nonlin_0608/')
    models = ["torque_nonlinear_280coef", "torque_nonlinear_100coef", "torque_nonlinear_20coef"]
    pref = "torque_nonlinear\\"
    pltdata_present = True
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='torque', show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)
    else:
        # plot modes coefs:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)
        path = os.path.join(os.getcwd(), 'plot_data', 'torque_nonlinear')
        plot_everything(path)

## 5 linear model currents with 50 ecc todo NOT YET DONE
elif process == 5:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-08', 'IMMEC_50ecc_ecc_5.0sec.npz')
    plot_optuna_data()
    models = []
    pref = "currents_50_linear\\"
    pltdata_present = False
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='currents', do_time_simulation=True,
                                       show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)

## 6 linear model torque with 50 ecc todo NOT YET DONE
elif process == 6:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-08', 'IMMEC_50ecc_ecc_5.0sec.npz')
    plot_optuna_data()
    models = []
    pref = "torque_50_linear\\"
    pltdata_present = False
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='torque', show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)

## 7 linear model UMP with 50 ecc
elif process == 7:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-08', 'IMMEC_50ecc_ecc_5.0sec.npz')
    plot_optuna_data("umpLinear-50ecc-optuna-study", dirs='ump_lin_0808/')
    models = ["ump_linear_50ecc_50",
              "ump_linear_50ecc_100",
              "ump_linear_50ecc_200",
              "ump_linear_50ecc_550"]
    pref = "ump_50_linear\\"
    pltdata_present = True
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='ump', show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)
    else:
        # plot modes coefs:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)
        path = os.path.join(os.getcwd(), 'plot_data', 'ump_50_linear')
        plot_everything(path)

## 8 nonlinear model currents with 50 ecc
elif process == 8:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-07-nonlin-50ecc', 'IMMEC_nonlin_50ecc_5.0sec.npz')
    plot_optuna_data('currentsNonlinear-50-specific-optuna-study')
    models = ["currents_nonlinear_50ecc_60", "currents_nonlinear_50ecc_90"]
    # "currents_nonlinear_50ecc_200", #this one diverges
    # "currents_nonlinear_50ecc_550"] #this one diverges too

    pref = "currents_50_nonlinear\\"
    pltdata_present = True
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='currents', do_time_simulation=True,
                                       show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)
    else:
        # plot modes coefs:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)
        path = os.path.join(os.getcwd(), 'plot_data', 'currents_50_nonlinear')
        plot_everything(path)

## 9 nonlinear model torque with 50 ecc todo NOT YET DONE
elif process == 9:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-07-nonlin-50ecc', 'IMMEC_nonlin_50ecc_5.0sec.npz')
    plot_optuna_data()
    models = []
    pref = "torque_50_nonlinear\\"
    pltdata_present = False
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='torque', show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)

## 10 nonlinear model ump with 50 ecc
elif process == 10:
    testdata = os.path.join(os.getcwd(), 'test-data', '08-07-nonlin-50ecc', 'IMMEC_nonlin_50ecc_5.0sec.npz')
    plot_optuna_data('umpNonlinear-50ecc-optuna-study', dirs='ump_nonlin_0708/')
    models = ["nonlinear_big_ump_50ecc", "nonlinear_ump_50ecc"]
    pref = "ump_50_nonlinear\\"
    pltdata_present = True
    if not pltdata_present:
        for model in models:
            sim, test = simulate_model(pref + model + '_model', testdata, modeltype='ump', show=False)
            plot_fourier(test, sim, dt=1e-4, tmax=5.0)
    else:
        if coefs:
            for model_ in models:
                model = load_model(pref + model_ + '_model')
                plot_coefs2(model, log=True)
        path = os.path.join(os.getcwd(), 'plot_data', 'ump_50_nonlinear')
        plot_everything(path)
