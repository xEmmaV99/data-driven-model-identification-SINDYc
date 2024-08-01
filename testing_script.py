from source import *

from optimize_parameters import plot_optuna_data
'''
path = np.zeros(6, dtype=object)
path[0] = os.path.join(os.getcwd(), 'train-data', '07-29-default', 'IMMEC_0ecc_5.0sec.npz')
path[1] = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-50', 'IMMEC_50ecc_5.0sec.npz')
path[2] = os.path.join(os.getcwd(), 'train-data', '07-31-ecc-90', 'IMMEC_90ecc_5.0sec.npz')
path[3] = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin', 'IMMEC_nonlinear-0ecc_5.0sec.npz')
path[4] = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin50', 'IMMEC_nonlinear-50ecc_5.0sec.npz')
path[5] = os.path.join(os.getcwd(), 'train-data', '07-31-nonlin90', 'IMMEC_nonlinear-90ecc_5.0sec.npz')

titles = ['0 ecc', '50 ecc', '90 ecc', 'non-linear 0 ecc', 'non-linear 50 ecc', 'non-linear 90 ecc']

for i in range(6):
    plot_immec_data(path[i], simulation_number=10, title=titles[i])
'''
#plot_immec_data(os.path.join(os.getcwd(), "train-data", "07-31-ecc-90", "IMMEC_90ecc_5.0sec.npz"),
#                simulation_number=19, title=' 90 ecc, different simulation')

paretos = np.zeros(3, dtype=object)
#paretos[0] = 'torque-optuna-study'
paretos[1] = 'currents-optuna-study'
#paretos[2] = 'ump2-optuna-study'

for i in range(3):
    plot_optuna_data(paretos[i])

#plot_optuna_data('example-study-poly-vs-best')