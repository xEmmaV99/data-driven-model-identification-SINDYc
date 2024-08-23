from prepare_data import *
from train_model_source import *
import os
'''
folderlist = ['07-31-ecc-50', '07-31-ecc-50','08-09', '08-13-default', '08-16']
folderpaths = [os.path.join(os.getcwd(), 'train-data', folder) for folder in folderlist]

for path in folderpaths:
    # get name of npz file inside this folder
    npzfile = [f for f in os.listdir(path) if f.endswith('.npz')][0]
    prepare_data(os.path.join(path, npzfile), seed = 1) #by defining the seed, we can reproduce the same result
'''

# create a model for UMP trained on random direction (fixed) 50% eccentricity
'''
path = os.path.join(os.getcwd(), 'train-data', 'not_used','ecc_random_direction','IMMEC_lin_ecc_randomecc_5.0sec.npz')
model = make_model(path_to_data_files = path,
    modeltype= 'ump',
    optimizer= 'STLSQ',
    lib= "poly_2nd_order",
    nmbr_of_train = -1,  alpha = 5.076, threshold = 0.7317,
    modelname= 'random_dir_ump',
    ecc_input = True)'''
'''
testpath = os.path.join(os.getcwd(), 'test-data', '08-05', 'IMMEC_50eccecc_5.0sec.npz')

simulate_model('random_dir_ump_model',     path_to_test_file= testpath,
    modeltype="ump", show = True,
    ecc_input=True)
'''
#why is this so bad?

# conclusion: training on random direction + Including ecc does not work..

# what If I create a dynamic model without ecc input?

traindata = os.path.join(os.getcwd(), 'train-data', '08-16', 'IMMEC_dynamic_nonlinear_5.0sec.npz')
model = make_model(path_to_data_files = traindata,
    modeltype= 'ump',
    optimizer= 'STLSQ',
    lib= "poly_2nd_order",
    nmbr_of_train = -1,  alpha = 5.076, threshold = 0.7317,
    modelname= 'dynamic_no_ecc',
    ecc_input = False)
# 38k MSE like expected. 130 terms though
'''
test = os.path.join(os.getcwd(), 'test-data', '08-18', 'IMMEC_dynamic_nonlinear_5.0sec.npz')
simulate_model('dynamic_no_ecc_model',     path_to_test_file= test,
    modeltype="ump", show = True,
    ecc_input=False)
'''
# I think I did this, and remember getting always around 40k MSE -> could just be bad luck

# Try with a model from the optuna study (with) ecc that had lower MSE because pareto plot is very expensive
