import pysindy as ps
import numpy as np

def get_library_names():
    """
    Returns the library names to be considered for the parameter optimisation
    :return: list of str
    """
    return ['poly_2nd_order', 'linear-specific', 'torque', 'nonlinear_terms', 'nonlinear_terms_with_f']

    #return ['nonlinear_terms', 'nonlinear_terms_with_f', 'poly_2nd_order', 'torque']


def get_custom_library_funcs(type, nmbr_input_features = 15):
    """
    Returns a pysindy library corresponding to the "type"
    :param type: str, name of the library
    :param nmbr_input_features: number of input features, important for 'inputs_per_library'
    :return: library
    """
    # Generalized library, sine and cos functions for gamma
    gamma = [12]
    fr = [14]
    i0_idx = [2,6]

    all = [i for i in range(nmbr_input_features)]
    all_but_gamma = [i for i in range(nmbr_input_features) if i not in gamma]
    all_but_gammafr = [i for i in range(nmbr_input_features) if i not in gamma and i not in fr]
    all_but_i0 = [i for i in range(nmbr_input_features) if i not in i0_idx]

    if type == 'poly_2nd_order':
        inputs_per_library = [all_but_gamma, gamma]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    elif type == 'nonlinear_terms':
        inputs_per_library = [all_but_gamma, gamma]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                           tensor_array= [[1,1]],  # merge libraries
                                           inputs_per_library=inputs_per_library)
    elif type == 'nonlinear_terms_with_f':
        library_functions = [
            lambda x: np.sin(2*np.pi*x),
            lambda x: np.cos(2*np.pi*x)]
        function_names = [
            lambda x: r'\\sin{2\pi ' + x + '}',
            lambda x: r'\\cos{2\pi ' + x + '}']

        inputs_per_library = [all_but_gammafr, gamma, fr]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True),
                                            ps.CustomLibrary(library_functions=library_functions, function_names=function_names)],
                                           tensor_array= [[1,1,0],[1,0,1]],  # merge libraries
                                           inputs_per_library=inputs_per_library)


    elif type == 'sincos_cross':
        library_functions = [
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
        ]
        function_names = [
            lambda x, y: '\\cos{' + x + y + '}',
            lambda x, y: '\\sin{' + x + y + '}',

        ]

        lib = ps.CustomLibrary(library_functions=library_functions, function_names=function_names,
                               interaction_only=False)

        inputs_per_library = [all_but_gamma, all_but_i0, gamma]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            lib,
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                           tensor_array=None,
                                           inputs_per_library=inputs_per_library)

    elif type == 'poly_2nd_order_extra_fourier':
        library_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y)]
        function_names = [
            lambda x: '\\sin{' + x + '}',
            lambda x: '\\cos{' + x + '}',
            lambda x, y: '\\cos{' + x + y + '}',
            lambda x, y: '\\sin{' + x + y + '}']

        inputs_per_library = [all_but_gamma, [12], all_but_i0]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True),
                                            ps.CustomLibrary(library_functions, function_names,
                                                             interaction_only=False)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    elif type == 'higher_order':
        custom_lib = ps.PolynomialLibrary(degree=8, include_interaction=False)

    elif type == 'custom':
        library_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x, y: y * np.sin(x),
            lambda x, y: y * np.cos(x),
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
            #lambda x, y, z: z * np.sin(x * y),
            #lambda x, y ,z: z * np.cos(x * y),
        ]
        function_names = [
            lambda x: '\\sin{' + x + '}',
            lambda x: '\\cos{' + x + '}',
            lambda x, y: y + '\\sin{' + x + '}',
            lambda x, y: y + '\\cos{' + x + '}',
            lambda x, y: '\\cos{' + x + y + '}',
            lambda x, y: '\\sin{' + x + y + '}',
            #lambda x, y,z: z+'\\sin{' + x + y + '}',
            #lambda x, y,z: z+'\\cos{' + x + y + '}',

        ]

        inputs_per_library = [all_but_gamma, all_but_i0]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.CustomLibrary(library_functions, function_names,
                                                             interaction_only=False)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    elif type == 'fourier':
        library_functions = [
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
            lambda x, y: np.cos(2 * x * y),
            lambda x, y: np.sin(2 * x * y),
        ]
        function_names = [
            lambda x, y: '\\cos{' + x + y + '}',
            lambda x, y: '\\sin{' + x + y + '}',
            lambda x, y: '\\cos{2' + x + y + '}',
            lambda x, y: '\\sin{2' + x + y + '}',
        ]
        lib_fourier = ps.FourierLibrary(n_frequencies=10, include_cos=True, include_sin=True)
        fourier_terms = [7, 8, 9]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.CustomLibrary(library_functions=library_functions,
                                                             function_names=function_names, interaction_only=False),
                                            lib_fourier],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=[all_but_gamma, fourier_terms])

    elif type == 'torque':
        library_functions2 = [
            lambda x,y: x*y
        ]
        library_function_names2 = [
            lambda x,y: x+y
        ]
        # i i i v v v I I I V V V om gam f
        input_per_library = [[9,10,0,1,6,7]]
        custom_lib = ps.GeneralizedLibrary([ps.CustomLibrary(library_functions2, library_function_names2, interaction_only=False)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=input_per_library)

    elif type == 'currents':
        library_functions = [
            lambda x : x
        ]
        library_function_names = [
            lambda x: x
        ]
        library_functions2 = [
            lambda x,y: x*y
        ]
        library_function_names2 = [
            lambda x,y: x+y
        ]
        # i i i v v v I I I V V V om gam f
        input_per_library = [[0,1,2,3,4,5,6,7,8,9,10,11],[0,1,2,6,7,8,9,10,11,12]]
        custom_lib = ps.GeneralizedLibrary([ps.CustomLibrary(library_functions , library_function_names , interaction_only=False),
                                            ps.CustomLibrary(library_functions2, library_function_names2, interaction_only=False)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=input_per_library)

    elif type == 'linear-specific':
        ins = [0,1,2,3,4,5,6,7,8,9,10,11] # i i i v v v I I I V V V
        ins2 = [12,13,3,4,5] # omega gamma v v v
        ins3 = [0,1,2,6,7,8,9,10,11] # i i i I I I V V V
        lin = [
            lambda x: x
        ]
        lin_name = [
            lambda x: x
        ]
        linear_terms = ps.GeneralizedLibrary([ps.CustomLibrary(lin, lin_name, interaction_only=False)],
                                             inputs_per_library=[ins])
        cross_terms = ps.GeneralizedLibrary([ps.CustomLibrary(lin, lin_name, interaction_only=False)],
                                             inputs_per_library=[ins2]) * \
                      ps.GeneralizedLibrary([ps.CustomLibrary(lin, lin_name, interaction_only=False)],
                                             inputs_per_library=[ins3])

        custom_lib = linear_terms + cross_terms


    else:
        raise ValueError('Library unknown')

    return custom_lib
