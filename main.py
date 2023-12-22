import numpy as np
import matplotlib.pyplot as plt
import json
import os

from algorithms.de import de_best_1_bin, de_rand_1_bin
from algorithms.pso import pso
from algorithms.soma import soma_all_to_all, soma_all_to_one

from algorithms.settings import REPETITIONS

import pybenchfunction as bench
from tqdm import tqdm  # Add this import for the progress bar


def plot_function_2d_3d(func, input_domain, n_space=100):
    func.input_domain = input_domain

    bench.plot_2d(func, n_space=n_space, ax=None)
    bench.plot_3d(func, n_space=n_space, ax=None)

    latex_img = bench.latex_img(func.latex_formula)
    plt.imshow(latex_img)
    plt.show()


def average_results(algorithm, func, bounds, runs=REPETITIONS):
    best_results = []
    best_fitness_values = []
    for _ in range(runs):
        best, fitness = algorithm(func, bounds)
        best_results.append(best)
        best_fitness_values.append(fitness)
    average_best = np.mean(best_results, axis=0)
    average_fitness = np.mean(best_fitness_values)
    return average_best, average_fitness


def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_images(func, input_domain, output_dir='images', n_space=100):
    func.input_domain = input_domain

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save 2D plot
    plot_2d_path = os.path.join(output_dir, f'{func.name}-2D.jpg')
    fig_2d = bench.plot_2d(func, n_space=n_space, ax=None, show=False)
    plt.savefig(plot_2d_path, dpi=300)
    plt.close(fig_2d)

    # Save 3D plot
    plot_3d_path = os.path.join(output_dir, f'{func.name}-3D.jpg')
    fig_3d = bench.plot_3d(func, n_space=n_space, ax=None, show=False)
    plt.savefig(plot_3d_path, dpi=300)
    plt.close(fig_3d)

    # Save Latex image
    latex_path = os.path.join(output_dir, f'{func.name}-latex.jpg')
    latex_img = bench.latex_img(func.latex_formula)
    plt.axis('off')  # Turn off axis for Latex image
    plt.imshow(latex_img)
    plt.savefig(latex_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    INPUT_DOMAIN = ([-100, 100], [-100, 100])  # Used for plotting
    DIMENSIONS = 30  # Could be: 2, 10, 30
    OUTPUT_FILE = 'average_results.json'

    results_dict = {}

    # get all the available functions accepting defined DIMENSIONS
    # any_dim_functions = bench.get_functions(None)
    # continous_multimodal_nonconvex_2d_functions = bench.get_functions(
    #     d=DIMENSIONS,
    #     continuous=True,
    #     convex=None,
    #     separable=None,
    #     differentiable=None,
    #     mutimodal=True,
    #     randomized_term=None
    # )

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
        bench.function.CustomFunction(DIMENSIONS),
        bench.function.CustomFunction2(DIMENSIONS),
        bench.function.CustomFunction3(DIMENSIONS),
        bench.function.CustomFunction4(DIMENSIONS),
        bench.function.CustomFunction5(DIMENSIONS),
        bench.function.CustomFunction7(DIMENSIONS),
        bench.function.CustomFunction8(DIMENSIONS),
        bench.function.CustomFunction9(DIMENSIONS),
        bench.function.CustomFunction10(DIMENSIONS),
    ]

    bounds = [(-100, 100) for _ in range(DIMENSIONS)]

    for func in tqdm(bench_functions, desc="Progress"):
        save_images(func, INPUT_DOMAIN)

        print('DE_BEST: ' + func.name)
        average_result = average_results(de_best_1_bin, func, bounds)
        results_dict[f'DE_BEST_{func.name}'] = {
            'average_best': average_result[0].tolist(),
            'average_fitness': float(average_result[1])
        }

        print('DE_RANDOM: ' + func.name)
        average_result = average_results(de_rand_1_bin, func, bounds)
        results_dict[f'DE_RANDOM_{func.name}'] = {
            'average_best': average_result[0].tolist(),
            'average_fitness': float(average_result[1])
        }

        print('PSO: ' + func.name)
        pso_avg_res = average_results(pso, func, bounds)
        results_dict[f'PSO_{func.name}'] = {
            'average_best': pso_avg_res[0].tolist(),
            'average_fitness': float(pso_avg_res[1])
        }

        print('SOMA AT1: ' + func.name)
        soma_at1_avg_res = average_results(soma_all_to_one, func, bounds)
        results_dict[f'SOMA_AT1_{func.name}'] = {
            'average_best': soma_at1_avg_res[0].tolist(),
            'average_fitness': float(soma_at1_avg_res[1])
        }

        print('SOMA ATA: ' + func.name)
        soma_ata_avg_res = average_results(soma_all_to_all, func, bounds)
        results_dict[f'SOMA_ATA_{func.name}'] = {
            'average_best': soma_ata_avg_res[0].tolist(),
            'average_fitness': float(soma_ata_avg_res[1])
        }

        save_to_json(results_dict, OUTPUT_FILE)
