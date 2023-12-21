from .settings import *
from .helpers import *
import numpy as np


class Individual:
    """Individual of the population. It holds parameters of the solution as well as the fitness of the solution."""

    def __init__(self, params, fitness):
        self.params = params
        self.fitness = fitness

    def __repr__(self):
        return '{} fitness: {}'.format(self.params, self.fitness)


def evaluate(fobj, params):
    """Returns fitness of the params"""
    return fobj(params)


def bounded(params, bounds):
    """
    Returns bounded version of params
    All params that are outside of bounds are reassigned by a random number within bounds
    """
    return np.array([np.random.uniform(bound[0], bound[1])
                     if params[d] < bound[0] or params[d] > bound[1]
                     else params[d]
                     for d, bound in enumerate(bounds)])


def generate_population(size, bounds, dimensions, fobj):
    def generate_individual():
        params = np.random.uniform([bound[0] for bound in bounds], [
                                   bound[1] for bound in bounds], dimensions)
        fitness = evaluate(fobj, params)
        return Individual(params, fitness)

    return [generate_individual() for _ in range(size)]


def generate_prt_vector(prt, dimensions):
    return np.random.choice([0, 1], dimensions, p=[prt, 1 - prt])


def get_leader(population):
    """Finds leader of the population by its fitness (the lower the better)."""
    return min(population, key=lambda individual: individual.fitness)


def soma_all_to_one(fobj, bounds, dimensions=None, prt=SOMA_ALL_TO_ONE['PRT'], path_length=SOMA_ALL_TO_ONE['PathLength'], step=SOMA_ALL_TO_ONE['StepSize']):
    if dimensions is None:
        dimensions = len(bounds)

    population_size = POPULATION_SIZE[f'{dimensions}D']
    population = generate_population(
        population_size, bounds, dimensions, fobj)

    migrations = int(FUNCTION_EVALUATIONS(dimensions))
    iterator = 0

    while iterator < migrations:
        leader = get_leader(population)
        for individual in population:
            if individual is leader:
                continue
            next_position = individual.params
            prt_vector = generate_prt_vector(prt, dimensions)
            for t in np.arange(step, path_length, step):
                current_position = individual.params + \
                    (leader.params - individual.params) * t * prt_vector
                current_position = bounded(current_position, bounds)
                fitness = evaluate(fobj, current_position)
                iterator += 1
                if fitness <= individual.fitness:
                    next_position = current_position
                    individual.fitness = fitness
            individual.params = next_position
    leader = get_leader(population)
    return leader.params, leader.fitness


def soma_all_to_all(fobj, bounds, dimensions=None, prt=SOMA_ALL_TO_ALL['PRT'], path_length=SOMA_ALL_TO_ALL['PathLength'], step=SOMA_ALL_TO_ALL['StepSize']):
    if dimensions is None:
        dimensions = len(bounds)

    population_size = POPULATION_SIZE[f'{dimensions}D']
    population = generate_population(
        population_size, bounds, dimensions, fobj)

    migrations = int(FUNCTION_EVALUATIONS(dimensions))
    iterator = 0

    while iterator < migrations:
        for individual in population:
            next_position = individual.params
            prt_vector = generate_prt_vector(prt, dimensions)
            for leading in population:
                if individual is leading:
                    continue
                for t in np.arange(step, path_length, step):
                    current_position = individual.params + \
                        (leading.params - individual.params) * t * prt_vector
                    current_position = bounded(current_position, bounds)
                    fitness = evaluate(fobj, current_position)
                    iterator += 1
                    if fitness <= individual.fitness:
                        next_position = current_position
                        individual.fitness = fitness
            individual.params = next_position
    leader = get_leader(population)
    return leader.params, leader.fitness


# https://www.dropbox.com/sh/u3cpa39t9yh5fsf/AAB6EarzZ1NTl6iRGoyxAeB7a?dl=0
