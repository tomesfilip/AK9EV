import json
from collections import defaultdict

de_best = r'C:\Users\filip\Documents\school\AK9EV\results\30D\average_results_de_best_30D_all.json'
de_rand = r'C:\Users\filip\Documents\school\AK9EV\results\30D\average_results_de_rand_30D_all.json'
pso = r'C:\Users\filip\Documents\school\AK9EV\results\30D\average_results_pso_30D_all.json'
soma_all_to_one = r'C:\Users\filip\Documents\school\AK9EV\results\30D\average_results_soma_1_30D_all.json'
soma_all_to_all = r'C:\Users\filip\Documents\school\AK9EV\results\30D\average_results_soma_ATA_30D_all.json'

merge_file_paths = [de_best, de_rand, pso, soma_all_to_one, soma_all_to_all]
output_file_path = 'avg_res_30D.json'


def merge_json_files(file_list):
    merged_data = defaultdict(
        lambda: {"FUNCTION_NAME": "", "algorithms": []})

    for idx, file_path in enumerate(file_list, start=1):
        with open(file_path, 'r') as file:
            data = json.load(file)

        for key, value in data.items():
            parts = key.split('_')
            algorithm_name = '-'.join(parts[:-1])
            function_name = parts[-1]

            merged_data[function_name]["FUNCTION_NAME"] = function_name
            merged_data[function_name]["algorithms"].append({
                "algorithm_name": algorithm_name,
                "average_best": value["average_best"],
                "average_fitness": value["average_fitness"]
            })

    result_list = list(merged_data.values())
    return result_list


def save_to_file(result_list, output_file):
    with open(output_file, 'w') as out_file:
        json.dump(result_list, out_file, indent=2)


result = merge_json_files(file_list=merge_file_paths)
save_to_file(result_list=result, output_file=output_file_path)
