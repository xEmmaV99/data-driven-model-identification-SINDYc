from source import *
import pysindy as ps
import numpy as np
# u names = v, I , V, gamma, omega, f

def get_custom_library_funcs(type='default'):
    # Generalized library, sine and cos functions for gamma
    all_but_gam = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    all_but_i0 = [0, 1, 3, 4, 5, 9, 10, 11, 12, 13, 14]
    lib_2nd_order = ps.PolynomialLibrary(degree=2, include_interaction=True)  # first order interaction
    lib_higher_order = ps.PolynomialLibrary(degree=5, include_interaction=False)


    if type == 'default':
        inputs_per_library = [all_but_gam, [12]]
        custom_lib = ps.GeneralizedLibrary([lib_2nd_order,
                                            ps.FourierLibrary(n_frequencies=10, include_cos=True, include_sin=True)], # does really nothing
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    elif type == 'exp':
        library_functions = [
            lambda x, y: np.cos(x * y),
            lambda x, y: np.sin(x * y),
            #lambda x, y: np.cos(2*x * y),
            #lambda x, y: np.sin(2*x * y),
            #lambda x, y: np.cos(3 * x * y),
            #lambda x, y: np.sin(3 * x * y),
            #lambda x, y: np.cos(4 * x * y),
            #lambda x, y: np.sin(4 * x * y),
            #lambda x, y: np.cos(5 * x * y),
            #lambda x, y: np.sin(5 * x * y),
            # lambda x: np.log(np.abs(x + 1)),
        ]
        function_names = [
            lambda x, y: '\\cos{' + x + y + '}',
            lambda x, y: '\\sin{' + x + y + '}',
            #lambda x, y: '\\cos{2' + x + y + '}',
            #lambda x, y: '\\sin{2' + x + y + '}',
            #lambda x, y: '\\cos{3' + x + y + '}',
            #lambda x, y: '\\sin{3' + x + y + '}',
            #lambda x, y: '\\cos{4' + x + y + '}',
            #lambda x, y: '\\sin{4' + x + y + '}',
            #lambda x, y: '\\cos{5' + x + y + '}',
            #lambda x, y: '\\sin{5' + x + y + '}',

            # lambda x: '\\log{' + x + '}',
        ]

        lib_exp = ps.CustomLibrary(library_functions=library_functions, function_names=function_names,
                                   interaction_only=False)

        inputs_per_library = [all_but_gam, all_but_i0, [12]] # remove i0 from exp
        custom_lib = ps.GeneralizedLibrary([lib_2nd_order,
                                            lib_exp,
                                            ps.FourierLibrary(n_frequencies=1, include_cos=True, include_sin=True)],
                                           tensor_array=None,
                                           inputs_per_library=inputs_per_library)
    elif type == 'higher_order':
        custom_lib = ps.PolynomialLibrary(degree=8, include_interaction=False)

    elif type == 'NEW':
        library_functions = [
            lambda x: 1/(1+abs(x)),
            lambda x: x*np.log(abs(1+x))
            # lambda x: np.log(np.abs(x + 1)),
        ]
        custom_lib = ps.GeneralizedLibrary([ps.PolynomialLibrary(degree=5, include_interaction=False),
                                            ps.CustomLibrary(library_functions=library_functions, interaction_only=False)],
                                             tensor_array=None)
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
        fourier_terms = [7,8,9]
        custom_lib = ps.GeneralizedLibrary([lib_2nd_order,
                                            ps.CustomLibrary(library_functions=library_functions, function_names=function_names, interaction_only=False),
                                            lib_fourier],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=[all_but_gam, all_but_i0+[12], fourier_terms])

    else:
        inputs_per_library = [all_but_gam, all_but_gam, [12]]
        custom_lib = ps.GeneralizedLibrary([lib_2nd_order, lib_higher_order,
                                            ps.FourierLibrary(n_frequencies=10, include_cos=True, include_sin=True)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    return custom_lib
