import json
import os
import argparse

def generate_configs(total_nodes, learning_rate, batch_size):
    configs_dir = "../configs"
    os.makedirs(configs_dir, exist_ok=True)

    for nodes in range(2, total_nodes + 1):
        node_weight = 100.0 / nodes
        node_weights = {str(i): node_weight for i in range(1, nodes + 1)}

        for iteration in range(10):
            config = {
                "experiment_name": f"{nodes}_{iteration}",
                "num_nodes": nodes,
                "node_weights": node_weights,
                "min_responses_for_aggregation": nodes,
                "num_aggregation_rounds": 5,
                "aggregation_per_epoch": 5,
                "network_file": "src/nets/net_basic.py",
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size
                },
                "data_directory": f"datasets/{total_nodes}nodes/",
                "results_directory": f"results/{total_nodes}nodes_results/1host_{nodes}nodes/swarm_results/",
                "iteration": iteration
            }

            filename = f"{nodes}_{iteration}_config.json"
            filepath = os.path.join(configs_dir, filename)

            with open(filepath, "w") as f:
                json.dump(config, f, indent=4)

    print(f"Configs generated in {configs_dir}")

def generate_run_scripts(total_nodes, learning_rate, batch_size):
    # Generate run-swarm.sh
    swarm_script_path = "../run-swarm.sh"
    with open(swarm_script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Run all configurations\n")
        f.write(f"for nodes in $(seq 2 {total_nodes}); do\n")
        f.write("    for iteration in {0..9}; do\n")
        f.write("        config_file=\"configs/${nodes}_${iteration}_config.json\"\n")
        f.write("        if [ -f \"$config_file\" ]; then\n")
        f.write("            echo \"Running $config_file\"\n")
        f.write("            (\n")
        f.write("                python src/learner/swarm_learner.py \"$config_file\"\n")
        f.write("            )\n")
        f.write("            sleep 10\n")
        f.write("            sync\n")
        f.write("        else\n")
        f.write("            echo \"Config file $config_file not found!\"\n")
        f.write("        fi\n")
        f.write("    done\n")
        f.write("done\n")

    print(f"Swarm run script generated at {swarm_script_path}")

    # Generate run-local.sh
    local_script_path = "../run-local.sh"
    with open(local_script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Run local learner\n")
        f.write(f"python src/learner/local_learner.py -n {total_nodes} -e 25 -b {batch_size} -l {learning_rate}\n")

    print(f"Local learner run script generated at {local_script_path}")

    # Generate run-central.sh
    central_script_path = "../run-central.sh"
    with open(central_script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Run central learner\n")
        f.write(f"python src/learner/central_learner.py -n {total_nodes} -e 25 -b {batch_size} -l {learning_rate}\n")

    print(f"Central learner run script generated at {central_script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate configuration files and run scripts for swarm learning.")
    parser.add_argument("-n", "--total_nodes", type=int, required=True, help="Total number of nodes.")
    parser.add_argument("-l", "--learning_rate", type=float, required=True, help="Learning rate for the model.")
    parser.add_argument("-b", "--batch_size", type=int, required=True, help="Batch size for training.")

    args = parser.parse_args()

    generate_configs(args.total_nodes, args.learning_rate, args.batch_size)
    generate_run_scripts(args.total_nodes, args.learning_rate, args.batch_size)
