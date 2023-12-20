import numpy as np
import matplotlib.pyplot as plt
from algorithms.de import de_best_1_bin

import pybenchfunction as bench


if __name__ == '__main__':
    continous_multimodal_nonconvex_2d_functions = bench.get_functions(
        2,  # dimension
        continuous=True,
        convex=None,
        separable=None,
        differentiable=None,
        mutimodal=True,
        randomized_term=None
    )

    for fun in continous_multimodal_nonconvex_2d_functions:
        print(fun.name)

    ackley = bench.function.Ackley(2)
    ackley.input_domain = ([-100, 100], [-100, 100])

    bench.plot_2d(ackley, n_space=100, ax=None)
    bench.plot_3d(ackley, n_space=100, ax=None)

    latex = ackley.latex_formula
    latex_img = bench.latex_img(latex)
    plt.imshow(latex_img)
    plt.show()
