import pysindy as ps
import numpy as np


def get_library_names():
    """
    Returns the library names to be considered for the parameter optimisation
    :return: list of str
    """
    # return ['w_co_linear_0ecc','poly_2nd_order', 'torque']
    return [
        "poly_2nd_order",
        "linear-specific",
        "torque",
        "nonlinear_terms",
        "interaction_only",
    ]
    # return ['nonlinear_terms', 'poly_2nd_order', 'torque']


def get_custom_library_funcs(type, nmbr_input_features=15):
    """
    Returns a pysindy library corresponding to the "type"
    :param type: str, name of the library
    :param nmbr_input_features: number of input features, important for 'inputs_per_library', should contain both x and u
    :return: library
    """
    # Generalized library, sine and cos functions for gamma
    gamma = [12]
    fr = [14]
    i0_idx = [2, 6]

    # some pre-defined input lists
    all = [i for i in range(nmbr_input_features)]
    all_but_gamma = [i for i in range(nmbr_input_features) if i not in gamma]
    all_but_gammafr = [
        i for i in range(nmbr_input_features) if i not in gamma and i not in fr
    ]
    all_but_i0 = [i for i in range(nmbr_input_features) if i not in i0_idx]

    if type == "poly_2nd_order":
        """
        contains all the features (but gamma) up to 2nd order and their crossterms, sin(gamma) and cos(gamma)
        """
        inputs_per_library = [all_but_gamma, gamma]
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=2, include_interaction=True),
                ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True),
            ],
            tensor_array=None,  # don't merge the libraries
            inputs_per_library=inputs_per_library,
        )

    elif type == "w_co":
        """
        contains all the features (but gamma, rx, ry and f) up to 2nd order and their crossterms, tensored with gamma, rx and ry
        """
        all_but_grf = [
            i for i in range(nmbr_input_features) if i not in [13, 14, 15, 16]
        ]
        poly = ps.PolynomialLibrary(degree=2, include_interaction=True)
        polygr = ps.PolynomialLibrary(degree=1, include_interaction=True)
        custom_lib = ps.GeneralizedLibrary(
            [poly, polygr],
            tensor_array=None,
            inputs_per_library=[all_but_grf, [13, 15, 16]],
        )

    elif type == "w_co_linear_0ecc":
        """
        contains i_d, i_q, I_d, I_q, V_d, V_q cross terms, tensored with gamma
        """
        library_functions2 = [lambda x, y: x * y]
        library_function_names2 = [lambda x, y: x + y]
        # i i i v v v I I I V V V gam om f
        input_per_library = [[9, 10, 0, 1, 6, 7], [12]]

        # ps.PolynomialLibrary(degree=2, include_interaction=True),
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.CustomLibrary(
                    library_functions2, library_function_names2, interaction_only=False
                ),
                ps.PolynomialLibrary(degree=1),
            ],
            tensor_array=[[1, 1]],
            inputs_per_library=input_per_library,
        )

    elif type == "pca":
        """
        Contains all features up to 2nd order and crossterms (without the quadratic terms)
        """
        custom_lib = ps.PolynomialLibrary(degree=2, interaction_only=True)

    elif type == "interaction_only":
        """
        Contains all features (but gamma) up to 3rd order and crossterms (without the the cubic terms), sin(gamma) and cos(gamma)
        """
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=3, interaction_only=True),
                ps.FourierLibrary(n_frequencies=1),
            ],
            tensor_array=None,
            inputs_per_library=[all_but_gamma, gamma],
        )

    elif type == "nonlinear_terms":
        """
        Contains all features (but gamma) up to 2nd order and crossterms TENSORED with sin(gamma) and cos(gamma)
        """
        inputs_per_library = [all_but_gamma, gamma]
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=2, include_interaction=True),
                ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True),
            ],
            tensor_array=[[1, 1]],  # merge libraries
            inputs_per_library=inputs_per_library,
        )

    elif type == "poly_2nd_order_extra_fourier":
        """
        Idem as poly_2nd_order, but with an extra Fourier library (1st deg) and crossterms like sin(x*y) and cos(x*y)
        """
        library_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
        ]
        function_names = [
            lambda x: "\\sin{" + x + "}",
            lambda x: "\\cos{" + x + "}",
            lambda x, y: "\\cos{" + x + y + "}",
            lambda x, y: "\\sin{" + x + y + "}",
        ]

        inputs_per_library = [all_but_gamma, [12], all_but_i0]
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=2, include_interaction=True),
                ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True),
                ps.CustomLibrary(
                    library_functions, function_names, interaction_only=False
                ),
            ],
            tensor_array=None,  # don't merge the libraries
            inputs_per_library=inputs_per_library,
        )

    elif type == "higher_order":
        """
        Consists of polynomials up to 8th order, without crossterms
        """
        custom_lib = ps.PolynomialLibrary(degree=8, include_interaction=False)

    elif type == "custom":
        """
        Consists of terms like sin(x), cos(x), y*sin(x), y*cos(x), cos(x*y), sin(x*y), but this library is too big to be used in practice
        """
        library_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x, y: y * np.sin(x),
            lambda x, y: y * np.cos(x),
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
            # lambda x, y, z: z * np.sin(x * y),
            # lambda x, y ,z: z * np.cos(x * y),
        ]
        function_names = [
            lambda x: "\\sin{" + x + "}",
            lambda x: "\\cos{" + x + "}",
            lambda x, y: y + "\\sin{" + x + "}",
            lambda x, y: y + "\\cos{" + x + "}",
            lambda x, y: "\\cos{" + x + y + "}",
            lambda x, y: "\\sin{" + x + y + "}",
            # lambda x, y,z: z+'\\sin{' + x + y + '}',
            # lambda x, y,z: z+'\\cos{' + x + y + '}',
        ]

        inputs_per_library = [all_but_gamma, all_but_i0]
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.PolynomialLibrary(degree=2, include_interaction=True),
                ps.CustomLibrary(
                    library_functions, function_names, interaction_only=False
                ),
            ],
            tensor_array=None,  # don't merge the libraries
            inputs_per_library=inputs_per_library,
        )

    elif type == "torque":
        """
        Contains all combinations of x*y of i_q, i_d, I_q, I_d, V_q, V_d
        """
        library_functions2 = [lambda x, y: x * y]
        library_function_names2 = [lambda x, y: x + y]
        # i i i v v v I I I V V V gam om f
        input_per_library = np.array([[9, 10, 0, 1, 6, 7]])
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.CustomLibrary(
                    library_functions2, library_function_names2, interaction_only=True
                )
            ],
            tensor_array=None,  # don't merge the libraries
            inputs_per_library=input_per_library,
        )

    elif type == "currents":
        """
        Contains combinations x and x*y, expected to show for the time derivative of the currents (based on note in Teams from 2024-07-25)
        """
        library_functions = [lambda x: x]
        library_function_names = [lambda x: x]
        library_functions2 = [lambda x, y: x * y]
        library_function_names2 = [lambda x, y: x + y]
        # i i i v v v I I I V V V gam om f
        input_per_library = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
        ]
        custom_lib = ps.GeneralizedLibrary(
            [
                ps.CustomLibrary(
                    library_functions, library_function_names, interaction_only=False
                ),
                ps.CustomLibrary(
                    library_functions2, library_function_names2, interaction_only=False
                ),
            ],
            tensor_array=None,  # don't merge the libraries
            inputs_per_library=input_per_library,
        )

    elif type == "linear-specific":
        '''
        Contains linear terms of i, v, I, V and combinations gamma, omega, v with i, I and V, (based on note in Teams from 2024-07-25)
        '''
        ins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # i i i v v v I I I V V V
        ins2 = [12, 13, 3, 4, 5]  # gamma omega v v v
        ins3 = [0, 1, 2, 6, 7, 8, 9, 10, 11]  # i i i I I I V V V
        lin = [lambda x: x]
        lin_name = [lambda x: x]
        linear_terms = ps.GeneralizedLibrary(
            [ps.CustomLibrary(lin, lin_name, interaction_only=False)],
            inputs_per_library=[ins],
        )
        cross_terms = ps.GeneralizedLibrary(
            [ps.CustomLibrary(lin, lin_name, interaction_only=False)],
            inputs_per_library=[ins2],
        ) * ps.GeneralizedLibrary(
            [ps.CustomLibrary(lin, lin_name, interaction_only=False)],
            inputs_per_library=[ins3],
        )

        custom_lib = linear_terms + cross_terms

    else:
        raise ValueError("Library unknown")

    return custom_lib
