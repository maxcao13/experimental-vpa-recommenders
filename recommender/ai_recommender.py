import recommender.recommender_config as recommender_config
import numpy as np
from datetime import datetime, timedelta
from utils import resource2str, get_target_containers, get_metric_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import threading
from queue import Queue
import queue
from utils import bound_var, str2resource
from recommender.plotting import plot_resource_usage, PLOT_TIME_RANGE

# TODO: hardcoded
LOWER_BOUND_MULTIPLIER = 0.90
UPPER_BOUND_MULTIPLIER = 1.10

def calculate_maximum_value_resource_recommendation(values: list[float]) -> float:
    """
    Calculate resource recommendation based on the values
    which is a list of floats of different containers and their
    most recent usage. Takes the max of the values.
    """
    if not values:
        return None
        
    values = np.array(values)
    return np.max(values)
    
def prepare_time_series_data(values, lookback=6):
    """
    Convert time series into supervised learning problem
    lookback: number of previous time steps to use as input variables
    """
    if len(values) < lookback + 1:
        print(f"Warning: Not enough data points. Need at least {lookback + 1}, got {len(values)}")
        return np.array([]), np.array([])

    X, y = [], []
    for i in range(len(values) - lookback):
        X.append(values[i:(i + lookback)])
        y.append(values[i + lookback])
    return np.array(X), np.array(y)

class BackgroundModelTrainer:
    def __init__(self):
        self.cpu_model = None
        self.memory_model = None
        self.cpu_scaler = None
        self.memory_scaler = None
        self.last_training_time = None
        self.training_interval = 300  # Force train every 5 minutes
        self.data_queue = Queue()
        self.is_running = True
        self.lock = threading.Lock()
        self.predictions = {
            'cpu': {'timestamps': [], 'values': []},
            'memory': {'timestamps': [], 'values': []}
        }
        self.min_data_points = 10  # Minimum data points before training
        self.recent_weight_factor = 2.0  # How much more weight recent values get (1.0 = equal weights)
        
        # Start background thread
        self.thread = threading.Thread(target=self._training_loop, daemon=True)
        self.thread.start()
    
    def _should_retrain(self, metric_type, data, current_time):
        """Check if we should retrain based on recent data"""
        if len(data) < self.min_data_points:
            print(f"Not enough data to train {metric_type} model: {len(data)} < {self.min_data_points}")
            return False
    
        if metric_type == 'cpu' and self.cpu_model is None:
            return True
        if metric_type == 'memory' and self.memory_model is None:
            return True
        
        time_to_train = (self.last_training_time is None or 
            (current_time - self.last_training_time).total_seconds() >= self.training_interval)
                
        if not time_to_train:
            print(f"Not enough time to train {metric_type} model")
            return False
            
        recent_data = data[-5:]
        historical_data = data[:-5]

        recent_mean = np.mean(recent_data)
        historical_mean = np.mean(historical_data)
        historical_std = np.std(historical_data)
        
        # Calculate how many standard deviations away the recent mean is
        z_score = abs(recent_mean - historical_mean) / (historical_std + 1e-10)
        
        print(f"\nChange Detection Stats for {metric_type.upper()}:")
        print(f"  Recent mean: {recent_mean:.4f}")
        print(f"  Historical mean: {historical_mean:.4f}")
        print(f"  Historical std: {historical_std:.4f}")
        print(f"  Z-score: {z_score:.4f}")
        
        # Retrain if recent mean is more than 1 standard deviation away from historical mean
        return z_score > 1.0
    
    def _training_loop(self):
        while self.is_running:
            try:
                data = self.data_queue.get(timeout=1)
                if data is None:
                    break
                    
                metric_type, X, y = data
                
                current_time = datetime.now()

                if self._should_retrain(metric_type, y, current_time):
                    print(f"\nRe-training {metric_type} model...")
                    model = self._train_model(X, y, metric_type)
                    if model is not None:
                        with self.lock:
                            if metric_type == 'cpu':
                                self.cpu_model = model
                            else:
                                self.memory_model = model
                            self.last_training_time = current_time

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in training loop: {e}")
                continue
    
    def _train_model(self, X, y, metric_type):
        if len(X) == 0 or len(y) == 0:
            print("No data to train model")
            return None
            
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store the scaler
        with self.lock:
            if metric_type == 'cpu':
                self.cpu_scaler = scaler
            else:
                self.memory_scaler = scaler
        
        # Calculate sample weights - more recent samples get higher weights
        n_samples = len(y)
        weights = np.exp(np.linspace(-1, 0, n_samples))  # Exponential decay from 1 to 0.37
        weights = weights * (self.recent_weight_factor - 1) + 1  # Scale to [1, recent_weight_factor]
        
        print(f"\nSample weights for {metric_type}:")
        print(f"  Most recent weight: {weights[-1]:.3f}")
        print(f"  Oldest weight: {weights[0]:.3f}")
        print(f"  Weight ratio: {weights[-1]/weights[0]:.3f}")
        
        # Use TimeSeriesSplit for proper time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model = LinearRegression()
        
        r2_scores = []
        mae_scores = []
        mse_scores = []
        rmse_scores = []
        baseline_rmse_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            # Get weights for current split
            train_weights = weights[train_idx]
            test_weights = weights[test_idx]
            
            # Fit with sample weights
            model.fit(X_scaled[train_idx], y[train_idx], sample_weight=train_weights)
            y_pred = model.predict(X_scaled[test_idx])
            y_true = y[test_idx]
            
            # Calculate weighted metrics
            r2 = model.score(X_scaled[test_idx], y_true, sample_weight=test_weights)
            mae = np.average(np.abs(y_true - y_pred), weights=test_weights)
            mse = np.average((y_true - y_pred) ** 2, weights=test_weights)
            rmse = np.sqrt(mse)
            
            # Calculate weighted baseline RMSE
            y_baseline = y[test_idx - 1]
            baseline_mse = np.average((y_true - y_baseline) ** 2, weights=test_weights)
            baseline_rmse = np.sqrt(baseline_mse)
            
            r2_scores.append(r2)
            mae_scores.append(mae)
            mse_scores.append(mse)
            rmse_scores.append(rmse)
            baseline_rmse_scores.append(baseline_rmse)
        
        avg_rmse = np.mean(rmse_scores)
        avg_baseline_rmse = np.mean(baseline_rmse_scores)
        
        print(f"  Model RMSE: {avg_rmse:.4f}")
        print(f"  Baseline RMSE (using last value): {avg_baseline_rmse:.4f}")
        print(f"  R² Score: {np.mean(r2_scores):.4f}")
        
        # Only return the model if it performs better than the naive baseline
        if avg_rmse < avg_baseline_rmse:
            print(f"Model performs better than baseline by {((avg_baseline_rmse - avg_rmse) / avg_baseline_rmse * 100):.1f}%")
            return model
        else:
            print(f"Model performs worse than baseline by {((avg_rmse - avg_baseline_rmse) / avg_baseline_rmse * 100):.1f}%")
            return None
    
    def add_training_data(self, metric_type, X, y):
        """Add new data to the training queue"""
        self.data_queue.put((metric_type, X, y))
    
    def get_latest_model(self, metric_type):
        """Get the latest trained model for the specified metric type"""
        with self.lock:
            return self.cpu_model if metric_type == 'cpu' else self.memory_model
    
    def stop(self):
        """Stop the background thread"""
        self.is_running = False
        self.data_queue.put(None)
        self.thread.join()

    def add_prediction(self, metric_type, timestamp, value):
        """Track a new prediction"""
        with self.lock:
            self.predictions[metric_type]['timestamps'].append(timestamp)
            self.predictions[metric_type]['values'].append(value)
            
            # Keep only last 1000 predictions
            if len(self.predictions[metric_type]['timestamps']) > 1000:
                self.predictions[metric_type]['timestamps'] = self.predictions[metric_type]['timestamps'][-1000:]
                self.predictions[metric_type]['values'] = self.predictions[metric_type]['values'][-1000:]
    
    def get_predictions(self, metric_type):
        """Get past predictions"""
        with self.lock:
            return {
                'timestamps': self.predictions[metric_type]['timestamps'].copy(),
                'values': self.predictions[metric_type]['values'].copy()
            }

    def predict(self, metric_type, values):
        """Make a prediction using the appropriate model and scaler"""
        with self.lock:
            model = self.cpu_model if metric_type == 'cpu' else self.memory_model
            scaler = self.cpu_scaler if metric_type == 'cpu' else self.memory_scaler
            
            if model is None or scaler is None:
                return None
                
            # Scale the input values
            X_scaled = scaler.transform([values])
            return model.predict(X_scaled)[0]

    def set_recent_weight_factor(self, factor):
        """Set how much more weight recent values get during training"""
        with self.lock:
            self.recent_weight_factor = max(1.0, factor)  # Ensure at least 1.0

model_trainer = BackgroundModelTrainer()

# TODO: hardcoded. should be based on the VPA spec
minCPU = "10m"
minMemory = "10Mi"
maxCPU = "10"
maxMemory = "10Gi"

min_allowed_cpu = str2resource("cpu", minCPU)
max_allowed_cpu = str2resource("cpu", maxCPU)
min_allowed_memory = str2resource("memory", minMemory)
max_allowed_memory = str2resource("memory", maxMemory)

# TODO: configurable
interval_step_size = 1  # in minutes

# TODO: configurable
model_trainer.set_recent_weight_factor(2.0)  # Make recent values x times more important

def ai_get_recommendation(vpa, corev1, prom_client):
    """
    Returns recommendations based on statistical analysis of Prometheus metrics
    """
    if not prom_client:
        return None

    vpa_spec = vpa["spec"]
    target_ref = vpa_spec["targetRef"]
    
    target_namespace = target_ref.get("namespace", recommender_config.DEFAULT_NAMESPACE)
    
    target_containers, existing_pods = get_target_containers(corev1, target_namespace, target_ref)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    recommendations = []
    for container_name in target_containers:
        pod_filter = '|'.join(existing_pods)
        
        cpu_query = f'rate(container_cpu_usage_seconds_total{{namespace="{target_namespace}",container="{container_name}",pod=~"{pod_filter}"}}[5m])'
        memory_query = f'container_memory_working_set_bytes{{namespace="{target_namespace}",container="{container_name}",pod=~"{pod_filter}"}}'
        
        try:
            # get the latest trained models
            cpu_model = model_trainer.get_latest_model('cpu')
            memory_model = model_trainer.get_latest_model('memory')
            
            # get the latest metrics
            cpu_data = get_metric_data(prom_client, cpu_query, start_time, end_time, step=f"{interval_step_size}m")
            memory_data = get_metric_data(prom_client, memory_query, start_time, end_time, step=f"{interval_step_size}m")
            
            cpu_values = [float(value[1]) for metric in cpu_data for value in metric['values']]
            memory_values = [float(value[1]) for metric in memory_data for value in metric['values']]

            current_time = datetime.now()
            prediction_time = current_time + timedelta(minutes=interval_step_size)  # should match recommender loop interval
            
            if cpu_model is not None:
                # make a prediction using the latest trained model
                cpu_prediction = model_trainer.predict('cpu', cpu_values[-6:])
                cpu_recommendation = cpu_prediction
                model_trainer.add_prediction('cpu', prediction_time, cpu_prediction)  # Store with future timestamp
            else:
                print(f"Warning: No trained CPU model available for {container_name}, falling back to max-based recommendation")
                start_time = end_time - timedelta(minutes=5)
                latest_cpu_data = get_metric_data(prom_client, cpu_query, start_time, end_time, step=f"{interval_step_size}m")
                latest_cpu_values = [float(value[1]) for metric in latest_cpu_data for value in metric['values']]
                cpu_recommendation = calculate_maximum_value_resource_recommendation(latest_cpu_values)
                # don't add prediction here, because we don't have a model
            
            if memory_model is not None:
                memory_prediction = model_trainer.predict('memory', memory_values[-6:])
                memory_recommendation = memory_prediction
                model_trainer.add_prediction('memory', prediction_time, memory_prediction)  # Store with future timestamp
            else:
                print(f"Warning: No trained memory model available for {container_name}, falling back to max-based recommendation")
                start_time = end_time - timedelta(minutes=5)
                latest_memory_data = get_metric_data(prom_client, memory_query, start_time, end_time, step=f"{interval_step_size}m")
                latest_memory_values = [float(value[1]) for metric in latest_memory_data for value in metric['values']]
                memory_recommendation = calculate_maximum_value_resource_recommendation(latest_memory_values)
                # don't add prediction here, because we don't have a model

            if cpu_recommendation is None or memory_recommendation is None:
                continue

            # bound the recommendation to the allowed min and max
            cpu_recommendation = bound_var(cpu_recommendation, min_allowed_cpu, max_allowed_cpu)
            memory_recommendation = bound_var(memory_recommendation, min_allowed_memory, max_allowed_memory)

            lower_cpu_recommendation = cpu_recommendation * LOWER_BOUND_MULTIPLIER
            lower_memory_recommendation = memory_recommendation * LOWER_BOUND_MULTIPLIER
            upper_cpu_recommendation = cpu_recommendation * UPPER_BOUND_MULTIPLIER
            upper_memory_recommendation = memory_recommendation * UPPER_BOUND_MULTIPLIER

            string_cpu_recommendation = resource2str("cpu", cpu_recommendation)
            string_memory_recommendation = resource2str("memory", memory_recommendation)
            string_lower_cpu_recommendation = resource2str("cpu", lower_cpu_recommendation)
            string_lower_memory_recommendation = resource2str("memory", lower_memory_recommendation)
            string_upper_cpu_recommendation = resource2str("cpu", upper_cpu_recommendation)
            string_upper_memory_recommendation = resource2str("memory", upper_memory_recommendation)

            recommendation = {
                "containerName": container_name,
                "target": {
                    "cpu": string_cpu_recommendation,
                    "memory": string_memory_recommendation,
                },
                "lowerBound": {
                    "cpu": string_lower_cpu_recommendation,
                    "memory": string_lower_memory_recommendation,
                },
                "upperBound": {
                    "cpu": string_upper_cpu_recommendation,
                    "memory": string_upper_memory_recommendation,
                },
                "uncappedTarget": {
                    "cpu": string_cpu_recommendation,
                    "memory": string_memory_recommendation,
                }
            }

            # prepare the data for training the model
            cpu_X, cpu_y = prepare_time_series_data(cpu_values)
            memory_X, memory_y = prepare_time_series_data(memory_values)
            
            if len(cpu_X) > 0:
                model_trainer.add_training_data('cpu', cpu_X, cpu_y)
            if len(memory_X) > 0:
                model_trainer.add_training_data('memory', memory_X, memory_y)

            if recommender_config.ENABLED_PLOTTING:
                plot_resource_usage(cpu_data, memory_data, container_name, model_trainer, PLOT_TIME_RANGE)

            recommendations.append(recommendation)
            
        except Exception as e:
            print(f"Error getting metrics for container {container_name}: {e}")
            continue
    
    return recommendations
