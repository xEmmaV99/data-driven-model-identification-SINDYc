# Sparse Identification of Induction Motor Nonlinear Dynamics in Unbalanced Conditions

**Table of Contents**
<!-- TOC -->
* [Sparse Identification of Induction Motor Nonlinear Dynamics in Unbalanced Conditions](#sparse-identification-of-induction-motor-nonlinear-dynamics-in-unbalanced-conditions)
  * [Description](#description)
  * [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Dependencies](#dependencies)
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
After installing the package manager [Conda](https://docs.conda.io/en/latest/), a new virtual environment can be created in the command line using:
~~~
conda create -n environment_name
conda activate environment_name
~~~
Then, the dependencies are installed by (note that cvxpy only works with pip):
~~~
conda install numpy numba matplotlib tqdm 
conda install -c conda-forge optuna
pip install cvxpy
~~~

To install pysindy either clone it from their Github
~~~
git clone https://github.com/dynamicslab/pysindy.git
pip install .
~~~
or
~~~
conda install conda-forge::pysindy
~~~
The tutorials are present as a .ipynb file, which can be opened by installing
~~~
conda install notebook
~~~
and running
~~~
jupyter notebook
~~~
in the command line.
### User Manual
The user manual is available as a Jupyter Notebook called `User_Manual.ipynb`.



