import json
import os
import time
import threading
import queue
import hashlib
from typing import Dict, List, Any
import tensorflow as tf
import numpy as np
import pandas as pd
from importlib import import_module
import h5py
import datetime
import getpass
from tensorflow.keras.metrics import AUC, Precision, Recall, Accuracy
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class SwarmLearner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.experiment_name = self.config["experiment_name"]
        self.num_nodes = self.config["num_nodes"]
        self.min_responses = self.config["min_responses_for_aggregation"]
        self.num_aggregation_rounds = self.config["num_aggregation_rounds"]
        self.aggregation_per_epoch = self.config["aggregation_per_epoch"]
        self.data_dir = self.config["data_directory"]
        self.iteration = self.config["iteration"]
        self.node_weights_config = self.config["node_weights"]
        
        # Load test data to get input_dim
        self.test_data = self.load_test_data()
        self.input_dim = self.test_data[0].shape[1]
        
        # Import network dynamically
        network_path = self.config["network_file"].replace('.py', '')
        if network_path.startswith('src/'):
            network_path = network_path[4:]
        network_module = import_module(f"src.{network_path.replace('/', '.')}")
        self.model_creator = network_module.create_model
        
        # Create weights directory
        self.weights_dir = f"weights_{self.experiment_name}"
        os.makedirs(self.weights_dir, exist_ok=True)
        
        # Create results directory
        self.results_dir = f"results/{self.experiment_name}_results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = f"{self.results_dir}/swarm_results.csv"
        self.init_results_file()
        
        # Queues for communication - CORRETTO: ora usa indici da 0 a num_nodes-1
        self.aggregator_queue = queue.Queue()
        self.node_queues = [queue.Queue() for _ in range(self.num_nodes)]
        
        # State tracking
        self.current_round = 0
        self.aggregator_node = 0
        self.node_weights = {}
        
    def init_results_file(self):
        """Initialize results CSV file with headers"""
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                f.write("datetime,user,splits,loss,auc,auprc,accuracy,precision,recall,iteration\n")
    
    def save_metrics(self, user: str, splits: float, metrics: Dict[str, float]):
        """Save metrics to results CSV"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.results_file, 'a') as f:
            f.write(f"{current_time},{user},{splits:.4f},{metrics['loss']:.4f},{metrics['auc']:.4f},{metrics['auprc']:.4f},{metrics['accuracy']:.4f},{metrics['precision']:.4f},{metrics['recall']:.4f},{self.iteration}\n")
    
    def load_test_data(self):
        """Load test dataset"""
        test_path = os.path.join(self.config["data_directory"], "test.csv")
        test_data = pd.read_csv(test_path)
        
        # Assume last column is target
        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
        
        return x_test, y_test
    
    def evaluate_model(self, model):
        """Evaluate model and return metrics"""
        x_test, y_test = self.test_data

        # Predict probabilities
        y_pred_proba = model.predict(x_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        loss = model.evaluate(x_test, y_test, verbose=0)[0]
        auc = AUC()(y_test, y_pred_proba).numpy()
        precision = Precision()(y_test, y_pred).numpy()
        recall = Recall()(y_test, y_pred).numpy()
        accuracy = Accuracy()(y_test, y_pred).numpy()

        return {
            'loss': loss,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }

    def create_model(self):
        """Create model instance with correct input dimension"""
        model = self.model_creator(self.input_dim)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["hyperparameters"]["learning_rate"]),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_node_data(self, node_id: int):
        """Load dataset for specific node"""
        # CORRETTO: ora node_id va da 0 a num_nodes-1
        data_path = f"{self.data_dir}node{node_id+1}.csv"
        data = pd.read_csv(data_path)
        
        # Assume last column is target
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        return x, y

    def calculate_node_weight(self, node_id: int) -> float:
        """Calculate weight based on configuration"""
        # CORRETTO: ora node_id va da 0 a num_nodes-1, ma nel JSON sono da 1 a num_nodes
        return self.node_weights_config[str(node_id+1)]

    def orchestrator_process(self):
        """Orchestrator that manages aggregation rounds and selects aggregators"""
        print("Orchestrator started")
        
        while self.current_round < self.num_aggregation_rounds:
            print(f"Starting aggregation round {self.current_round}")
            
            # Calculate weights for this round from configuration
            self.node_weights = {}
            total_weight = 0
            
            # CORRETTO: ora node_id va da 0 a num_nodes-1
            for node_id in range(self.num_nodes):
                weight = self.calculate_node_weight(node_id)
                self.node_weights[node_id] = weight
                total_weight += weight
            
            # Normalize weights
            for node_id in self.node_weights:
                self.node_weights[node_id] /= total_weight
            
            # Select aggregator (round-robin) - CORRETTO: ora node_id va da 0 a num_nodes-1
            self.aggregator_node = self.current_round % self.num_nodes
            print(f"Selected node {self.aggregator_node} as aggregator for round {self.current_round}")
            
            # Notify aggregator - CORRETTO: usa indici 0-based
            self.node_queues[self.aggregator_node].put({
                'type': 'aggregator',
                'round': self.current_round,
                'weights': self.node_weights
            })
            
            # Notify other nodes - CORRETTO: usa indici 0-based
            for node_id in range(self.num_nodes):
                if node_id != self.aggregator_node:
                    self.node_queues[node_id].put({
                        'type': 'worker',
                        'round': self.current_round,
                        'aggregator': self.aggregator_node
                    })
            
            # Wait for aggregation completion
            aggregator_done = self.aggregator_queue.get()
            if aggregator_done['round'] == self.current_round:
                print(f"Aggregation round {self.current_round} completed")
                self.current_round += 1
            
            time.sleep(1)  # Small delay between rounds

    def node_process(self, node_id: int):
        """Process for individual node"""
        print(f"Node {node_id} started")
        
        model = self.create_model()
        x, y = self.load_node_data(node_id)
        
        # Get node weight for metrics
        node_weight = self.calculate_node_weight(node_id)
        
        # CORRETTO: current_weights deve essere inizializzata qui e resa persistente
        current_weights = model.get_weights()
        
        while True:
            try:
                message = self.node_queues[node_id].get(timeout=300)  # 5min timeout
                
                if message['type'] == 'aggregator':
                    current_weights = self.aggregator_behavior(node_id, model, x, y, message, current_weights)
                elif message['type'] == 'worker':
                    current_weights = self.worker_behavior(node_id, model, x, y, message, node_weight, current_weights)
                    
            except queue.Empty:
                print(f"Node {node_id} timeout")
                break
        
        print(f"Node {node_id} finished")

    def worker_behavior(self, node_id: int, model, x, y, message, node_weight: float, current_weights):
        """Behavior for worker nodes"""
        round_num = message['round']
        aggregator = message['aggregator']
        
        print(f"Node {node_id} working on round {round_num}")
        
        # Train for local epochs - CORRETTO: usa current_weights passata come parametro
        model.set_weights(current_weights)
        history = model.fit(x, y, 
                          epochs=self.aggregation_per_epoch,
                          batch_size=self.config["hyperparameters"]["batch_size"],
                          verbose=0)
        
        # Save weights - CORRETTO: ora node_id è 0-based, ma nel filename usiamo 1-based
        weight_file = f"{self.weights_dir}/{self.experiment_name}_node{node_id+1}.weights.h5"
        model.save_weights(weight_file)
        print(f"Node {node_id} saved weights for round {round_num}")
        
        # Evaluate and save metrics
        metrics = self.evaluate_model(model)
        self.save_metrics(f"node_{node_id+1}", node_weight, metrics)
        
        # Wait for aggregated weights
        aggregated_file = f"{self.weights_dir}/{self.experiment_name}_swarm_round{round_num}.weights.h5"
        
        while not os.path.exists(aggregated_file):
            time.sleep(5)
        
        # Load aggregated weights
        model.load_weights(aggregated_file)
        current_weights = model.get_weights()
        print(f"Node {node_id} loaded aggregated weights for round {round_num}")
        
        return current_weights  # CORRETTO: restituisce i pesi aggiornati

    def aggregator_behavior(self, node_id: int, model, x, y, message, current_weights):
        """Behavior for aggregator node"""
        round_num = message['round']
        weights_map = message['weights']
        
        print(f"Node {node_id} acting as aggregator for round {round_num}")
        
        # Train locally first - CORRETTO: usa current_weights passata come parametro
        model.set_weights(current_weights)
        history = model.fit(x, y,
                          epochs=self.aggregation_per_epoch,
                          batch_size=self.config["hyperparameters"]["batch_size"],
                          verbose=0)
        
        # Save local weights - CORRETTO: ora node_id è 0-based, ma nel filename usiamo 1-based
        local_weight_file = f"{self.weights_dir}/{self.experiment_name}_node{node_id+1}.weights.h5"
        model.save_weights(local_weight_file)
        
        # Evaluate and save local metrics
        node_weight = self.calculate_node_weight(node_id)
        metrics = self.evaluate_model(model)
        self.save_metrics(f"node_{node_id+1}", node_weight, metrics)
        
        # Wait for enough nodes to complete - CORRETTO: ora node_id è 0-based, ma nel filename usiamo 1-based
        completed_nodes = 0
        while completed_nodes < self.min_responses:
            completed_nodes = 0
            for nid in range(self.num_nodes):
                node_file = f"{self.weights_dir}/{self.experiment_name}_node{nid+1}.weights.h5"
                if os.path.exists(node_file):
                    completed_nodes += 1
            time.sleep(2)
        
        print(f"Aggregator found {completed_nodes} completed nodes")
        
        # Perform weighted aggregation
        aggregated_weights = self.aggregate_weights(round_num, weights_map)
        
        # Save aggregated weights
        aggregated_file = f"{self.weights_dir}/{self.experiment_name}_swarm_round{round_num}.weights.h5"
        model.set_weights(aggregated_weights)
        model.save_weights(aggregated_file)
        
        # Evaluate aggregated model and save metrics
        aggregated_metrics = self.evaluate_model(model)
        self.save_metrics("swarm_aggregated", 1.0, aggregated_metrics)
        
        # Notify orchestrator
        self.aggregator_queue.put({
            'node': node_id,
            'round': round_num,
            'completed': True
        })
        
        # Update current weights for next round
        current_weights = aggregated_weights
        print(f"Aggregator completed round {round_num}")
        
        return current_weights  # CORRETTO: restituisce i pesi aggiornati

    def aggregate_weights(self, round_num: int, weights_map: Dict[int, float]):
        """Perform weighted averaging of weights"""
        all_weights = []
        node_weights = []
        
        for node_id, weight in weights_map.items():
            # CORRETTO: ora node_id è 0-based, ma nel filename usiamo 1-based
            weight_file = f"{self.weights_dir}/{self.experiment_name}_node{node_id+1}.weights.h5"
            
            if os.path.exists(weight_file):
                model = self.create_model()
                model.load_weights(weight_file)
                weights = model.get_weights()
                all_weights.append(weights)
                node_weights.append(weight)
        
        # Weighted average
        aggregated_weights = []
        for i in range(len(all_weights[0])):
            layer_weights = np.zeros_like(all_weights[0][i])
            for j, weights in enumerate(all_weights):
                layer_weights += weights[i] * node_weights[j]
            aggregated_weights.append(layer_weights)
        
        return aggregated_weights

    def run(self):
        """Start the swarm learning process"""
        print(f"Starting Swarm Learning Experiment: {self.experiment_name}")
        
        # Start orchestrator thread
        orchestrator_thread = threading.Thread(target=self.orchestrator_process)
        orchestrator_thread.daemon = True
        orchestrator_thread.start()
        
        # Start node threads - CORRETTO: ora node_id va da 0 a num_nodes-1
        node_threads = []
        for node_id in range(self.num_nodes):
            thread = threading.Thread(target=self.node_process, args=(node_id,))
            thread.daemon = True
            thread.start()
            node_threads.append(thread)
        
        # Wait for completion
        try:
            orchestrator_thread.join()
            for thread in node_threads:
                thread.join()
        except KeyboardInterrupt:
            print("Experiment interrupted by user")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python swarm_learner.py <config_json>")
        sys.exit(1)
    
    learner = SwarmLearner(sys.argv[1])
    learner.run()