from .settings import *
from .helpers import *
import numpy as np
import matplotlib.pyplot as plt


def pso(fobj, bounds, w=PSO['w'], c1=PSO['c1'], c2=PSO['c2'], dimensions=None):
    if dimensions is None:
        dimensions = len(bounds)

    population_size = POPULATION_SIZE[f'{dimensions}D']
    iterations = int(FUNCTION_EVALUATIONS(dimensions) / population_size)

    particles = np.random.uniform(-100, 100, (population_size, dimensions))
    velocities = np.zeros((population_size, dimensions))

    best_positions = np.copy(particles)
    best_fitness = np.array([fobj(p) for p in particles])

    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    for i in range(iterations):
        r1 = np.random.uniform(0, 1, (population_size, dimensions))
        r2 = np.random.uniform(0, 1, (population_size, dimensions))
        velocities = w * velocities + c1 * r1 * \
            (best_positions - particles) + c2 * \
            r2 * (swarm_best_position - particles)

        particles += velocities
        particles = check_bounds(
            particles, np.array(bounds)[:, 0], np.array(bounds)[:, 1])

        fitness_values = np.array([fobj(p) for p in particles])

        improved_indices = np.where(fitness_values < np.ravel(best_fitness))
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    return swarm_best_position, swarm_best_fitness


# Example usage
# def objective_function(x):
#     x = np.array(x)  # Convert the list to a NumPy array
#     return np.sum(x**2)


# bounds = [(-100, 100) for _ in range(30)]  # Example bounds for 30D
# for result in pso(objective_function, bounds, dimensions=30):
#     print(result)
