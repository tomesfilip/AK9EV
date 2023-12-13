dimensions = [2, 10, 30]
population_sizes = {'2D': 10, '10D': 20, '30D': 50}
function_evaluations = lambda D: 2000 * D
border_control = "Reflection"
repetitions = 30

de_rand_1_bin = {'F': 0.8, 'CR': 0.9}
de_best_1_bin = {'F': 0.5, 'CR': 0.9}
pso = {'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298}
soma_all_to_one = {'PathLength': 3, 'StepSize': 0.11, 'PRT': 0.7}
soma_all_to_all = {'PathLength': 3, 'StepSize': 0.11, 'PRT': 0.7}
