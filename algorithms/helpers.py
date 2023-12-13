from numpy import clip


def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1])
                     for i in range(len(bounds))]
    return mutated_bound
