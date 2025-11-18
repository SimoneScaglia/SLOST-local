import os
import json

def create_config_files(base_dir, test_set_size, iterations=5):
    base_config = {
        "experiment_name": "",
        "num_nodes": 5,
        "node_weights": {
            "1": 20.0,
            "2": 20.0,
            "3": 20.0,
            "4": 20.0,
            "5": 20.0
        },
        "min_responses_for_aggregation": 5,
        "num_aggregation_rounds": 5,
        "aggregation_per_epoch": 5,
        "network_file": "src/nets/net_basic.py",
        "hyperparameters": {
            "learning_rate": 0.0025,
            "batch_size": 128
        },
        "data_directory": "datasets/40nodes/",
        "test_file": "",
        "results_directory": "",
        "iteration": 0
    }

    os.makedirs(base_dir, exist_ok=True)

    for test_size in test_set_size:
        for iteration in range(iterations):
            config = base_config.copy()
            config["experiment_name"] = f"5_{iteration}_testsize{test_size}"
            config["test_file"] = f"datasets/40nodes_testset/test_40nodes_for_iteration_{iteration}_{test_size}.csv"
            config["results_directory"] = f"results/var_testset_experiments/5_0_testsize{test_size}/"
            config["iteration"] = iteration

            file_name = f"5_{iteration}_testsize{test_size}.json"
            file_path = os.path.join(base_dir, file_name)

            with open(file_path, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Created: {file_path}")

if __name__ == "__main__":
    base_directory = "../configs"
    test_set_size = [5000, 10000, 15000, 20000, 25000, 30000]
    create_config_files(base_directory, test_set_size, 5)
