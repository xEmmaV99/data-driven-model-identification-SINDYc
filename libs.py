from source import *

# u names = v, I , V, gamma, omega, f

def get_custom_library_funcs(type='default'):
    # Generalized library, sine and cos functions for gamma
    all_but_gam = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]
    lib_2nd_order = ps.PolynomialLibrary(degree=2, include_interaction=True)  # first order interaction
    lib_higher_order = ps.PolynomialLibrary(degree=5, include_interaction=False)
    library_functions=[
        lambda x: np.exp(np.abs(x/1000)),
        lambda x: np.exp(-np.abs(x)),
        lambda x: np.log(np.abs(x+.1)),
    ]
    function_names = [
        lambda x: '\\exp{x}',
        lambda x: '\\exp{-x}',
        lambda x: '\\log{x}',
    ]


    lib_exp = ps.CustomLibrary(library_functions=library_functions, function_names=function_names, interaction_only=False)

    if type == 'default':
        inputs_per_library = [all_but_gam, [12]]
        custom_lib = ps.GeneralizedLibrary([lib_2nd_order,
                                           ps.FourierLibrary(n_frequencies=10, include_cos=True, include_sin=True)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    elif type == 'exp':
        inputs_per_library = [all_but_gam, all_but_gam, [12]]
        custom_lib = ps.GeneralizedLibrary([lib_2nd_order,
                                           lib_exp,
                                           ps.FourierLibrary(n_frequencies=10, include_cos=True, include_sin=True)],
                                           tensor_array=None,
                                           inputs_per_library=inputs_per_library)
    else:
        inputs_per_library = [all_but_gam, all_but_gam, [12]]
        custom_lib = ps.GeneralizedLibrary([lib_higher_order, lib_higher_order,
                                           ps.FourierLibrary(n_frequencies=10, include_cos=True, include_sin=True)],
                                           tensor_array=None,  # don't merge the libraries
                                           inputs_per_library=inputs_per_library)
    return custom_lib
