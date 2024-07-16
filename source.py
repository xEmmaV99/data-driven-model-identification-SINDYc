from tqdm import tqdm
from immec import *
import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps

def reference_abc_to_dq0(coord_array):
    # coord_array is a Nx3 array with N the number of samples

    #align a to d axis and project b and c axis on q
    T = 3/2*np.array([[1,-0.5,-0.5],
                     [0,np.sqrt(3)/2,-np.sqrt(3)/2],
                     [0.5,0.5,0.5]])

    return np.dot(T,coord_array.T).T
    # ugly selection but otherwise it has size (N,) which is not desired for hstack



if __name__ == '__main__' :
    def f(t,offset): return np.sin(2 * np.pi * t + offset)
    # test reference_abc_to_dq0 : for a balanced system
    array = np.array([[f(0,0),f(0,2/3*np.pi),f(0,4/3*np.pi)],
                      [f(0.2,0),f(0.2,2/3*np.pi),f(0.2,4/3*np.pi)]])
    print(reference_abc_to_dq0(array))