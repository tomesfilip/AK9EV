import numpy as np
from numpy import clip
from .settings import *
from .helpers import *


def de_best_1_bin(fobj, bounds, F=DE_BEST_1_BIN['F'], cr=DE_BEST_1_BIN['CR'], dimensions=None):
    if dimensions is None:
        dimensions = len(bounds)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    population_size = POPULATION_SIZE[f'{dimensions}D']
    iterations = int(FUNCTION_EVALUATIONS(dimensions) / population_size)

    population = np.random.rand(population_size, dimensions)
    pop_denorm = min_b + population * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(iterations):
        for j in range(population_size):
            idxs = [idx for idx in range(population_size) if idx != j]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            best_vector = population[best_idx]
            mutant = np.clip(best_vector + F * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < cr

            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, best_vector)
            trial_denorm = min_b + trial * diff
            trial_denorm = check_bounds(trial_denorm, min_b, max_b)

            f = fobj(trial_denorm)

            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

    return best, fitness[best_idx]


def de_rand_1_bin(fobj, bounds, F=DE_RAND_1_BIN['F'], cr=DE_RAND_1_BIN['CR'], dimensions=None):
    if dimensions is None:
        dimensions = len(bounds)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    population_size = POPULATION_SIZE[f'{dimensions}D']
    iterations = int(FUNCTION_EVALUATIONS(dimensions) / population_size)

    population = np.random.rand(population_size, dimensions)
    pop_denorm = min_b + population * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(iterations):
        for j in range(population_size):
            idxs = [idx for idx in range(population_size) if idx != j]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < cr

            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, population[j])
            trial_denorm = min_b + trial * diff
            trial_denorm = check_bounds(trial_denorm, min_b, max_b)

            f = fobj(trial_denorm)

            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

    return best, fitness[best_idx]


# def objective_function(x):
#     return np.sum(x**2)


# bounds = [(-5, 5) for _ in range(10)]

# print('BEST:')
# for result in de_best_1_bin(objective_function, bounds, dimensions=10):
#     print(result)

# print('\nRANDOM:')
# for result in de_rand_1_bin(objective_function, bounds, dimensions=10):
#     print(result)
