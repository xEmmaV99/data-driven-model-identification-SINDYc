import pysindy as ps
import numpy as np

# u names = v, I , V, gamma, omega, f
def get_library_keys():
    return ['poly_2nd_order', 'sincos_cross', 'system', 'higher_order', 'best', 'fourier']


def get_custom_library_funcs(type='default'):
    # Generalized library, sine and cos functions for gamma
    all_but_gamma = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    all_but_i0 = [0, 1, 4, 5, 9, 10, 11, 12, 13, 14]
    gamma = [12]
    if type == 'poly_2nd_order':
        inputs_per_library = [all_but_gamma, gamma]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=2, include_interaction=True),
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                           tensor_array=None,  # don't merge the libraries
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
    elif type == 'system':
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

    elif type == 'best':
        library_functions = [
            lambda x: np.sin(x),
            lambda x: np.cos(x),
            lambda x, y: y * np.sin(x),
            lambda x, y: y * np.cos(x),
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
            lambda x, y, z: z * np.sin(x * y),
            lambda x, y ,z: z * np.cos(x*y),


        ]
        function_names = [
            lambda x: '\\sin{' + x + '}',
            lambda x: '\\cos{' + x + '}',
            lambda x, y: y + '\\sin{' + x + '}',
            lambda x, y: y + '\\cos{' + x + '}',
            lambda x, y: '\\cos{' + x + y + '}',
            lambda x, y: '\\sin{' + x + y + '}',
            lambda x, y,z: z+'\\sin{' + x + y + '}',
            lambda x, y,z: z+'\\cos{' + x + y + '}',

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

    else:
        raise ValueError('Library unknown')

    return custom_lib
