# Sparse Identification of Induction Motor Nonlinear Dynamics in Unbalanced Conditions

**Table of Contents**
<!-- TOC -->
* [Sparse Identification of Induction Motor Nonlinear Dynamics in Unbalanced Conditions](#sparse-identification-of-induction-motor-nonlinear-dynamics-in-unbalanced-conditions)
  * [Description](#description)
  * [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Dependencies](#dependencies)
  * [Configuration](#configuration)
    * [Choice of Regulagization](#choice-of-regulagization-)
    * [Choice of Library](#choice-of-library)
  * [Usage](#usage)
    * [1) Data Generation](#1-data-generation)
    * [2) Data Preparation](#2-data-preparation)
    * [3) Optimization of hyperparameters](#3-optimization-of-hyperparameters)
    * [4) Model Identification](#4-model-identification)
    * [5) Model Evaluation](#5-model-evaluation)
  * [Additional Info](#additional-info)
<!-- TOC -->

## Description
This project aims to identify a model feasible for control purposes of an induction machine. 
This is obtained by a data-driven based system identification called 'SINDy with control' or 'SINDYc', 
solving the differential equation 
```math
\dot{X} = f(x,u)
```
where $`X`$ is the state vector, $`u`$ is the input control vector and $`f`$ is the function to be identified. 
SINDYc uses a library of candidate functions to identify the dynamics of the system $`\Theta(X)`$ and a sparse vector
$`\Xi`$ to be identified, such that
```math
\dot{X} = \Theta(X) \Xi
```
In this project, the l1 regularization is used to enforce sparsity in the solution.
We aim to find a dynamical model of the stator current $`i_{dq0,s}`$ by using $`x = [i_{dq0,s}]`$ 
and $`u = [v_{dq0,s}, \int_0^t i_{dq0,s}(\tau)d\tau, \int_0^t v_{dq0,s}(\tau)d\tau, \gamma^r, \omega^r]`$ as
the state and input vectors, respectively.
Here, $`v_{dq0,s}`$ is the stator voltage, $`\gamma^r`$ is the rotor angle 
and $`\omega^r`$ is the rotor angular speed.

Additionally, Torque and UMP are also identified through little manipulation to the SINDYc model.

Static and dynamic eccentricities are considered.

The data is obtained from simulations using the Python package IMMEC.

## Installation
### Prerequisites
### Dependencies
<!-- pip install or conda install !-->

## Configuration
### Choice of Regularization



### Choice of Library
For pySINDy to work, a library of candidate functions must be defined. 
Pre-defined libraries from pySINDy are polynomials or Fourier terms, though pySINDy allows for
custom libraries to be defined and fine-tuned for each input variable. 
Some of our own libraries are predefined in `libs.py` and can be called by the user: `get_custom_library_funcs('nonlinear_terms')`.

## Usage

### 1) Data Generation
The data generation is completely distict from the other aspects of the project. 
It uses the time-stepping solver IMMEC to model the induction machine and obtain the data.
The user can generate the data by running the script `data_generation.py` and setting the parameters in the script.
The user can choose to generate test or training data, resulting in one or multiple (`numbr_of_simulations`) 
simulations respectively. An initial eccentricity can also be set by `ecc_value` (value between 0 and 1) and `ecc_dir` 
(the direction in $`x, y`$ coordinates). The mode can be set to `linear`or `nonlinear` to simulate the machine.

The function automatically creates a folder inside the `train-data` or `test-data` folder with the name of the date, 
and saves the simulation data in a `.npz`-file with the name provided by the user by `save_name`.

The `.npz`-file contains the following arrays:
- `i_st` - Stator current
- `omega_rot` - Rotor angular speed
- `T_em` - Electromagnetic torque
- `F_em` - Electromagnetic force or UMP
- `v_applied` - Applied line voltages
- `T_l` - Load torque
- `ecc` - Eccentricity
- `time` - Simulated time
- `flux_st_yoke` - Stator yoke flux
- `gamma_rot` - Rotor angle
- `wcoe` - Magnetic coenergy

If multiple simulations are saved in one file, which is the case for traindata, 
the arrays have an additional (3rd) dimension for each simulation.

### 2) Data Preparation
COMMENT FOR THE AUTHOR: this function is just very important and useful if the `train_model.py` is not as desired. 


In order to create a model, the training data must be prepared. This is done in the script `data_preparation.py`, which is 
called by the  `prepare_data` function during training and postprocessing by other scripts. The user can call this function if desired,
but this is not necessary. The function takes the following arguments:
- `path_to_data_file` - Path to the `.npz`-file containing the data
- `test_data` - default False, this omits the extra preperation needed for trainingdata
- `number_of_trainfiles` - default -1 (all files), can be set to a number if not all simulations should be considered. The choise of selected simualtions is random. This can be useful to reduce the trainingssamples for large datasets.
- `use_estimate_for_v` - default False, if True, the `v_abc` are estimated from the line voltages.
- `usage_per_trainfile` - default 0.5, the percentage of the data used from each simulations.
- `ecc_input`- default False, if True, the eccentricity is used as an input variable to the model.

The function returns a dictionary containing the following arrays:
- `x`- Currents
- `u`- Input values, if `ecc_input` is True, the eccentricity is also included
- `xdot` - Time derivative of the currents
- `feature_names` - Names of the features to pass to the SINDy model

Additionally, as one might want to fit a SINDy model for the torque or UMP (by replacing `xdot`), the following are also present:
- `UMP` - Unbalanced magnetic pull
- `T_em` - Electromagnetic torque
- `wcoe` - The magentic coenergy



If the data is trainingsdata, it is split up into train and validation data (80% - 20%),    
in which case the dictionary also contains all the previous values but ending with `_train` and `_val`. 

### 3) Optimization of hyperparameters
As described in [click on this link](#choice-of-regularisation), the Lasso and SR3 regulators are considered, yielding 1 and 2 hyperparameters respectively.
Hence, the validation data is used to select the best parameter values. This can be combined with a different selection of
library candidate functions


### 4) Model Identification
### 5) Model Evaluation


## Additional Information
### Examples

