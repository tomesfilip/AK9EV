import numpy as np
import matplotlib.pyplot as plt
from algorithms.de import de_best_1_bin
from algorithms.de import de_rand_1_bin

import pybenchfunction as bench


def plot_function_2d_3d(func, input_domain, n_space=100):
    func.input_domain = input_domain

    bench.plot_2d(func, n_space=n_space, ax=None)
    bench.plot_3d(func, n_space=n_space, ax=None)

    latex = func.latex_formula
    latex_img = bench.latex_img(latex)
    plt.imshow(latex_img)
    plt.show()


if __name__ == '__main__':
    INPUT_DOMAIN = ([-100, 100], [-100, 100])  # Used for plotting
    DIMENSIONS = 10  # Could be: 2, 10, 30

    # get all the available functions accepting defined DIMENSIONS
    any_dim_functions = bench.get_functions(None)
    continous_multimodal_nonconvex_2d_functions = bench.get_functions(
        d=DIMENSIONS,
        continuous=True,
        convex=None,
        separable=None,
        differentiable=None,
        mutimodal=True,
        randomized_term=None
    )

    bench_functions = [
        bench.function.Ackley(DIMENSIONS),
        bench.function.HappyCat(DIMENSIONS),
        bench.function.Michalewicz(DIMENSIONS),
        bench.function.Periodic(DIMENSIONS),
        bench.function.PermDBeta(DIMENSIONS),
        bench.function.Qing(DIMENSIONS),
        bench.function.Quartic(DIMENSIONS),
        bench.function.Rastrigin(DIMENSIONS),
        bench.function.Rosenbrock(DIMENSIONS),
        bench.function.Salomon(DIMENSIONS),
        bench.function.Schwefel(DIMENSIONS),
        bench.function.Shubert(DIMENSIONS),
        bench.function.ShubertN3(DIMENSIONS),
        bench.function.ShubertN4(DIMENSIONS),
        bench.function.StyblinskiTank(DIMENSIONS),
        bench.function.Thevenot(DIMENSIONS),
    ]

    print('TOTAL FUNCTIONS:')
    print(len(bench_functions))

    bounds = [(-100, 100) for _ in range(DIMENSIONS)]
    for func in bench_functions:
        print('BEST: ' + func.name)
        for result in de_best_1_bin(func, bounds):
            print(result)

        print('\nRANDOM: ' + func.name)
        for result in de_rand_1_bin(func, bounds):
            print(result)

    # plot_function_2d_3d(func, INPUT_DOMAIN)
