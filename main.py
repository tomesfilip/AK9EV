import numpy as np
import matplotlib.pyplot as plt
from algorithms.de import de_best_1_bin

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
    INPUT_DOMAIN = ([-100, 100], [-100, 100])

    # get all the available functions accepting ANY dimension
    any_dim_functions = bench.get_functions(None)
    continous_multimodal_nonconvex_2d_functions = bench.get_functions(
        2,  # dimension
        continuous=True,
        convex=None,
        separable=None,
        differentiable=None,
        mutimodal=True,
        randomized_term=None
    )

    functions_to_plot = [
        bench.function.Ackley(2),
        bench.function.Adjiman(2),
        bench.function.AlpineN2(2),
        bench.function.Beale(2),
        bench.function.Bird(2),
        bench.function.BohachevskyN2(2),
        bench.function.BohachevskyN3(2),
        bench.function.Branin(2),
        bench.function.BukinN6(6),
        bench.function.CrossInTray(2),
        bench.function.DeJongN5(2),
        bench.function.DeckkersAarts(2),
        bench.function.Easom(2),
        bench.function.EggCrate(2),
        bench.function.HappyCat(2),
        bench.function.Himmelblau(2),
        bench.function.HolderTable(2),
        bench.function.Keane(2),
        bench.function.Langermann(2),
        bench.function.LevyN13(2),
        bench.function.McCormick(2),
        bench.function.Michalewicz(2),
        bench.function.Periodic(2),
        bench.function.PermDBeta(2),
        bench.function.Qing(2),
        bench.function.Quartic(2),
        bench.function.Rastrigin(2),
        bench.function.Rosenbrock(2),
        bench.function.Salomon(2),
        bench.function.Schwefel(2),
        bench.function.Shubert(2),
        bench.function.ShubertN3(2),
        bench.function.ShubertN4(2),
        bench.function.StyblinskiTank(2),
        bench.function.Thevenot(2),
    ]

    for func in functions_to_plot:
        plot_function_2d_3d(func, INPUT_DOMAIN)
