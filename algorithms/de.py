import numpy as np
from numpy import clip
from settings import *


def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1])
                     for i in range(len(bounds))]
    return mutated_bound


def de_best_1_bin(fobj, bounds, F=DE_BEST_1_BIN['F'], cr=DE_BEST_1_BIN['CR'], dimensions=None, repetitions=REPETITIONS):
    if dimensions is None:
        dimensions = len(bounds)

    bounds = np.asarray(bounds)  # Convert bounds to a NumPy array

    min_b, max_b = bounds.T
    diff = np.fabs(min_b - max_b)

    # Adjust population size based on dimensions
    population_size = POPULATION_SIZE[f'{dimensions}D']

    # Initialize population
    pop = np.random.rand(population_size, dimensions)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(repetitions):
        for j in range(population_size):
            idxs = [idx for idx in range(population_size) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(best + F * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < cr

            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trial_denorm = check_bounds(trial_denorm, bounds)
            f = fobj(trial_denorm)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        yield best, fitness[best_idx]


def de_rand_1_bin(fobj, bounds, F=DE_RAND_1_BIN['F'], cr=DE_RAND_1_BIN['CR'], dimensions=None, repetitions=REPETITIONS):
    if dimensions is None:
        dimensions = len(bounds)

    bounds = np.asarray(bounds)  # Convert bounds to a NumPy array

    min_b, max_b = bounds.T
    diff = np.fabs(min_b - max_b)

    # Adjust population size based on dimensions
    population_size = POPULATION_SIZE[f'{dimensions}D']

    # Initialize population
    pop = np.random.rand(population_size, dimensions)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(repetitions):
        for j in range(population_size):
            idxs = [idx for idx in range(population_size) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < cr

            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trial_denorm = check_bounds(trial_denorm, bounds)
            f = fobj(trial_denorm)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        yield best, fitness[best_idx]


def objective_function(x):
    x = np.array(x)  # Convert the list to a NumPy array
    return np.sum(x**2)


bounds = [(-100, 100) for _ in range(30)]  # Example bounds for 30D

for result in de_best_1_bin(objective_function, bounds, dimensions=30):
    print(result)
for result in de_rand_1_bin(objective_function, bounds, dimensions=30):
    print(result)
