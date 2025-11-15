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
from tensorflow.keras.metrics import AUC, Precision, Recall, BinaryAccuracy
import sys
import glob
import traceback
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # CRITICAL: Ensure deterministic operations
os.environ['PYTHONHASHSEED'] = str(42)

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
        self.results_dir = self.config["results_directory"]
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = f"{self.results_dir}/swarm_results.csv"
        self.init_results_file()
        
        # Queues for communication
        self.aggregator_queue = queue.Queue()
        self.node_queues = [queue.Queue() for _ in range(self.num_nodes)]
        self.final_metrics_queue = queue.Queue()
        
        # State tracking
        self.current_round = 0
        self.aggregator_node = 0
        self.node_weights = {}
        
        # Error handling
        self.error_occurred = False
        self.error_lock = threading.Lock()
        self.error_message = None
        self.error_traceback = None
        
        # Thread tracking
        self.active_threads = []
        self.shutdown_event = threading.Event()
        self.final_metrics_saved = False
        
        # Track completion
        self.completed_rounds = 0
        self.rounds_lock = threading.Lock()
        self.final_training_done = False
    
    def record_error(self, error_msg: str):
        """Record an error and signal shutdown"""
        with self.error_lock:
            if not self.error_occurred:
                self.error_occurred = True
                self.error_message = error_msg
                self.error_traceback = traceback.format_exc()
                print(f"ERROR RECORDED: {error_msg}")
                print(f"TRACEBACK: {self.error_traceback}")
                # Signal shutdown
                self.shutdown_event.set()
    
    def check_errors(self):
        """Check if any errors occurred"""
        with self.error_lock:
            return self.error_occurred, self.error_message, self.error_traceback

    def init_results_file(self):
        """Initialize results CSV file with headers"""
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                f.write("datetime,user,splits,loss,auc,auprc,accuracy,precision,recall,iteration\n")
    
    def save_metrics(self, user: str, splits: float, metrics: Dict[str, float]):
        """Save metrics to results CSV"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.results_file, 'a') as f:
            f.write(f"{current_time},{user},{splits},{metrics['loss']},{metrics['auc']},{metrics['auprc']},{metrics['accuracy']},{metrics['precision']},{metrics['recall']},{self.iteration}\n")
    
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

        eval_results = model.evaluate(x_test, y_test, verbose=0)
        metric_names = ['loss', 'auc', 'auprc', 'accuracy', 'precision', 'recall']

        results_dict = dict(zip(metric_names, eval_results))

        return results_dict

    def create_model(self):
        """Create model instance with correct input dimension"""
        model = self.model_creator(self.input_dim)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["hyperparameters"]["learning_rate"]),
            loss='binary_crossentropy',
            metrics=[
                AUC(name='auc', curve='ROC', num_thresholds=1000),
                AUC(name='auprc', curve='PR', num_thresholds=1000),
                BinaryAccuracy(name='accuracy'),
                Precision(name='precision'),
                Recall(name='recall')
            ]
        )
        return model

    def load_node_data(self, node_id: int):
        """Load dataset for specific node"""
        data_path = f"{self.data_dir}node{node_id+1}_{self.iteration}.csv"
        data = pd.read_csv(data_path)
        
        # Assume last column is target
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        return x, y

    def calculate_node_weight(self, node_id: int) -> float:
        """Calculate weight based on configuration"""
        return self.node_weights_config[str(node_id+1)]

    def orchestrator_process(self):
        """Orchestrator that manages aggregation rounds and selects aggregators"""
        try:
            print("Orchestrator started")
            
            while (self.current_round < self.num_aggregation_rounds and 
                   not self.shutdown_event.is_set()):
                print(f"Starting aggregation round {self.current_round}")
                
                # Check for errors before proceeding
                if self.shutdown_event.is_set():
                    break
                
                # Calculate weights for this round from configuration
                self.node_weights = {}
                total_weight = 0
                
                for node_id in range(self.num_nodes):
                    weight = self.calculate_node_weight(node_id)
                    self.node_weights[node_id] = weight
                    total_weight += weight
                
                # Normalize weights
                for node_id in self.node_weights:
                    self.node_weights[node_id] /= total_weight
                
                # Select aggregator (round-robin)
                self.aggregator_node = self.current_round % self.num_nodes
                print(f"Selected node {self.aggregator_node} as aggregator for round {self.current_round}")
                
                is_final_round = (self.current_round == self.num_aggregation_rounds - 1)
                
                # Notify aggregator
                self.node_queues[self.aggregator_node].put({
                    'type': 'aggregator',
                    'round': self.current_round,
                    'weights': self.node_weights,
                    'final_round': is_final_round
                })
                
                # Notify other nodes
                for node_id in range(self.num_nodes):
                    if node_id != self.aggregator_node:
                        self.node_queues[node_id].put({
                            'type': 'worker',
                            'round': self.current_round,
                            'aggregator': self.aggregator_node,
                            'final_round': is_final_round
                        })
                
                # Wait for aggregation completion with timeout
                try:
                    aggregator_done = self.aggregator_queue.get(timeout=300)  # 5min timeout
                    if aggregator_done['round'] == self.current_round:
                        print(f"Aggregation round {self.current_round} completed")
                        
                        # Check if this is the final round
                        if is_final_round:
                            print("Final aggregation round completed. Starting final evaluation.")
                            # Signal all nodes to do final evaluation
                            for node_id in range(self.num_nodes):
                                self.node_queues[node_id].put({
                                    'type': 'final_evaluation',
                                    'round': self.current_round
                                })
                            
                            # Wait for all nodes to complete final evaluation
                            nodes_completed = 0
                            while nodes_completed < self.num_nodes and not self.shutdown_event.is_set():
                                try:
                                    final_done = self.final_metrics_queue.get(timeout=60)
                                    nodes_completed += 1
                                    print(f"Node {final_done['node_id']} completed final evaluation ({nodes_completed}/{self.num_nodes})")
                                except queue.Empty:
                                    print("Timeout waiting for final evaluation")
                                    break
                            
                            print("All nodes completed final evaluation. Signaling shutdown.")
                            self.shutdown_event.set()
                            break
                            
                        self.current_round += 1
                except queue.Empty:
                    print(f"Timeout waiting for aggregator in round {self.current_round}")
                    self.record_error(f"Orchestrator timeout in round {self.current_round}")
                    break
                
                time.sleep(1)  # Small delay between rounds
                
        except Exception as e:
            self.record_error(f"Orchestrator error: {str(e)}")
        
        print("Orchestrator finished")

    def node_process(self, node_id: int):
        """Process for individual node"""
        try:
            print(f"Node {node_id} started")
            
            model = self.create_model()
            x, y = self.load_node_data(node_id)
            
            # Get node weight for metrics
            node_weight = self.calculate_node_weight(node_id)
            
            current_weights = model.get_weights()
            
            while not self.shutdown_event.is_set():
                try:
                    # Use timeout to periodically check for shutdown
                    message = self.node_queues[node_id].get(timeout=1)
                    
                    if self.shutdown_event.is_set():
                        break
                        
                    if message['type'] == 'aggregator':
                        current_weights = self.aggregator_behavior(node_id, model, x, y, message, current_weights)
                    elif message['type'] == 'worker':
                        current_weights = self.worker_behavior(node_id, model, x, y, message, node_weight, current_weights)
                    elif message['type'] == 'final_evaluation':
                        self.final_evaluation_behavior(node_id, model, x, y, node_weight, message)
                        break
                        
                except queue.Empty:
                    # Check for shutdown and continue
                    continue
                except Exception as e:
                    self.record_error(f"Node {node_id} message processing error: {str(e)}")
                    break
            
        except Exception as e:
            self.record_error(f"Node {node_id} error: {str(e)}")
        
        print(f"Node {node_id} finished")

    def worker_behavior(self, node_id: int, model, x, y, message, node_weight: float, current_weights):
        """Behavior for worker nodes"""
        try:
            round_num = message['round']
            aggregator = message['aggregator']
            is_final_round = message.get('final_round', False)
            
            print(f"Node {node_id} working on round {round_num}")
            
            # Check for shutdown
            if self.shutdown_event.is_set():
                return current_weights
            
            # Train for local epochs
            model.set_weights(current_weights)
            history = model.fit(x, y, 
                              epochs=self.aggregation_per_epoch,
                              batch_size=self.config["hyperparameters"]["batch_size"],
                              verbose=0)
            
            # For final round, save local weights but don't proceed to aggregation
            if is_final_round:
                print(f"Node {node_id} completed final local training in round {round_num}")
                # Keep the locally trained weights for final evaluation
                return model.get_weights()
            
            # Save weights for aggregation (non-final rounds)
            weight_file = f"{self.weights_dir}/{self.experiment_name}_node{node_id+1}.weights.h5"
            model.save_weights(weight_file)
            print(f"Node {node_id} saved weights for round {round_num}")
            
            # Wait for aggregated weights with timeout and shutdown checks
            aggregated_file = f"{self.weights_dir}/{self.experiment_name}_swarm_round{round_num}.weights.h5"
            
            wait_start = time.time()
            while not os.path.exists(aggregated_file) and not self.shutdown_event.is_set():
                if time.time() - wait_start > 300:  # 5min timeout
                    self.record_error(f"Node {node_id} timeout waiting for aggregated weights in round {round_num}")
                    return current_weights
                time.sleep(5)
            
            if self.shutdown_event.is_set():
                return current_weights
            
            # Load aggregated weights
            model.load_weights(aggregated_file)
            current_weights = model.get_weights()
            print(f"Node {node_id} loaded aggregated weights for round {round_num}")
            
            return current_weights
            
        except Exception as e:
            self.record_error(f"Worker {node_id} error in round {message['round']}: {str(e)}")
            return current_weights

    def aggregator_behavior(self, node_id: int, model, x, y, message, current_weights):
        """Behavior for aggregator node"""
        try:
            round_num = message['round']
            weights_map = message['weights']
            is_final_round = message.get('final_round', False)
            
            print(f"Node {node_id} acting as aggregator for round {round_num}")
            
            # Check for shutdown
            if self.shutdown_event.is_set():
                return current_weights
            
            # Train locally first
            model.set_weights(current_weights)
            history = model.fit(x, y,
                              epochs=self.aggregation_per_epoch,
                              batch_size=self.config["hyperparameters"]["batch_size"],
                              verbose=0)
            
            # For final round, don't aggregate - just keep local weights
            if is_final_round:
                print(f"Aggregator {node_id} completed final local training in round {round_num}")
                # Notify orchestrator that final training is done
                self.aggregator_queue.put({
                    'node': node_id,
                    'round': round_num,
                    'completed': True
                })
                return model.get_weights()
            
            # Save local weights (non-final rounds)
            local_weight_file = f"{self.weights_dir}/{self.experiment_name}_node{node_id+1}.weights.h5"
            model.save_weights(local_weight_file)
            
            # Wait for enough nodes to complete with timeout and shutdown checks
            completed_nodes = 0
            wait_start = time.time()
            while completed_nodes < self.min_responses and not self.shutdown_event.is_set():
                if time.time() - wait_start > 300:  # 5min timeout
                    self.record_error(f"Aggregator {node_id} timeout waiting for nodes in round {round_num}")
                    break
                    
                completed_nodes = 0
                for nid in range(self.num_nodes):
                    node_file = f"{self.weights_dir}/{self.experiment_name}_node{nid+1}.weights.h5"
                    if os.path.exists(node_file):
                        completed_nodes += 1
                time.sleep(2)
            
            if self.shutdown_event.is_set():
                return current_weights
            
            print(f"Aggregator found {completed_nodes} completed nodes")
            
            # Perform weighted aggregation
            aggregated_weights = self.aggregate_weights(round_num, weights_map)
            
            # Save aggregated weights
            aggregated_file = f"{self.weights_dir}/{self.experiment_name}_swarm_round{round_num}.weights.h5"
            model.set_weights(aggregated_weights)
            model.save_weights(aggregated_file)
            
            # Notify orchestrator
            self.aggregator_queue.put({
                'node': node_id,
                'round': round_num,
                'completed': True
            })
            
            # Update current weights for next round
            current_weights = aggregated_weights
            print(f"Aggregator completed round {round_num}")
            
            return current_weights
            
        except Exception as e:
            self.record_error(f"Aggregator {node_id} error in round {message['round']}: {str(e)}")
            return current_weights

    def final_evaluation_behavior(self, node_id: int, model, x, y, node_weight: float, message):
        """Final evaluation after last local training"""
        try:
            print(f"Node {node_id} performing final evaluation")
            
            # The model already has the weights from the final local training
            # Evaluate and save metrics
            metrics = self.evaluate_model(model)
            self.save_metrics(f"node_{node_id+1}_final", node_weight, metrics)
            print(f"Node {node_id} saved final metrics")
            
            # Notify orchestrator that final evaluation is complete
            self.final_metrics_queue.put({
                'node_id': node_id,
                'round': message['round']
            })
            
        except Exception as e:
            self.record_error(f"Node {node_id} error in final evaluation: {str(e)}")
            # Still notify orchestrator to avoid deadlock
            self.final_metrics_queue.put({
                'node_id': node_id,
                'round': message['round'],
                'error': str(e)
            })

    def aggregate_weights(self, round_num: int, weights_map: Dict[int, float]):
        """Perform weighted averaging of weights"""
        all_weights = []
        node_weights = []
        
        for node_id, weight in weights_map.items():
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

    def cleanup_weights(self):
        """Remove all .h5 weights files after the process ends"""
        weight_files = glob.glob(f"{self.weights_dir}/*.h5")
        for file in weight_files:
            try:
                os.remove(file)
                print(f"Removed weight file: {file}")
            except OSError as e:
                print(f"Error removing file {file}: {e}")
        os.rmdir(self.weights_dir)

    def stop_threads(self):
        """Stop all threads gracefully by setting shutdown event and clearing queues."""
        print("Stopping all threads...")
        self.shutdown_event.set()
        
        # Clear all queues to unblock threads
        for q in self.node_queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        
        while not self.aggregator_queue.empty():
            try:
                self.aggregator_queue.get_nowait()
            except queue.Empty:
                break
            
        while not self.final_metrics_queue.empty():
            try:
                self.final_metrics_queue.get_nowait()
            except queue.Empty:
                break
        
        print("All threads have been signaled to stop.")

    def run(self):
        """Start the swarm learning process"""
        print(f"Starting Swarm Learning Experiment: {self.experiment_name}")

        try:
            # Start orchestrator thread
            orchestrator_thread = threading.Thread(target=self.orchestrator_process)
            orchestrator_thread.daemon = True
            orchestrator_thread.start()
            self.active_threads.append(orchestrator_thread)

            # Start node threads
            node_threads = []
            for node_id in range(self.num_nodes):
                thread = threading.Thread(target=self.node_process, args=(node_id,))
                thread.daemon = True
                thread.start()
                node_threads.append(thread)
                self.active_threads.append(thread)

            # Monitor for completion or errors
            while (any(thread.is_alive() for thread in self.active_threads) and 
                   not self.shutdown_event.is_set()):
                time.sleep(1)
                
                # Check for errors
                error_occurred, error_msg, error_tb = self.check_errors()
                if error_occurred:
                    print(f"Error detected: {error_msg}")
                    break

            # Wait a bit for threads to finish
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("Experiment interrupted by user")
        except Exception as e:
            print(f"An error occurred in main thread: {e}")
            traceback.print_exc()
        finally:
            # Ensure cleanup happens regardless of errors
            self.stop_threads()
            self.cleanup_weights()
            
            # Print error info if any
            error_occurred, error_msg, error_tb = self.check_errors()
            if error_occurred:
                print(f"\n=== ERROR SUMMARY ===")
                print(f"Error: {error_msg}")
                if error_tb:
                    print(f"Traceback: {error_tb}")
            
            print("Swarm learning process ended.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python swarm_learner.py <config_json>")
        sys.exit(1)
    
    learner = SwarmLearner(sys.argv[1])
    learner.run()