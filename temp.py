from prepare_data import *
import os
folderlist = ['07-31-ecc-50', '07-31-ecc-50','08-09', '08-13-default', '08-16']
folderpaths = [os.path.join(os.getcwd(), 'train-data', folder) for folder in folderlist]

for path in folderpaths:
    # get name of npz file inside this folder
    npzfile = [f for f in os.listdir(path) if f.endswith('.npz')][0]
    prepare_data(os.path.join(path, npzfile), seed = 1) #by defining the seed, we can reproduce the same result