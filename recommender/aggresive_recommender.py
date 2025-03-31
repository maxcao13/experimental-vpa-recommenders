import numpy as np
from datetime import datetime, timedelta
from utils import resource2str, get_target_containers, get_metric_data, bound_var
from recommender.recommender_config import DEFAULT_NAMESPACE

# TODO: hardcoded
LOWER_BOUND_MULTIPLIER = 0.99 # very small margins since we want to be very aggressive
UPPER_BOUND_MULTIPLIER = 1.01

def calculate_resource_recommendation(values: list[float]) -> float:
    """
    Calculate resource recommendation based on the values
    which is a list of floats of different containers and their
    most recent usage. Takes the max of the values.
    """
    if not values:
        return None
        
    values = np.array(values)
    return np.max(values)
    
def aggressive_get_recommendation(vpa, corev1, prom_client):
    """
    Returns recommendations based on statistical analysis of Prometheus metrics
    """
    if not prom_client:
        return None

    vpa_spec = vpa["spec"]
    target_ref = vpa_spec["targetRef"]
    
    target_namespace = target_ref.get("namespace", DEFAULT_NAMESPACE)
    
    target_containers, existing_pods = get_target_containers(corev1, target_namespace, target_ref)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=1) # small since we only care about the latest data
    
    recommendations = []
    for container_name in target_containers:
        # Only get metrics from running containers
        pod_filter = '|'.join(existing_pods)
        
        # Query for CPU usage (returns cores)
        cpu_query = f'rate(container_cpu_usage_seconds_total{{namespace="{target_namespace}",container="{container_name}",pod=~"{pod_filter}"}}[5m])'
        # Query for memory usage (returns bytes)
        memory_query = f'container_memory_working_set_bytes{{namespace="{target_namespace}",container="{container_name}",pod=~"{pod_filter}"}}'
        
        try:
            cpu_data = get_metric_data(prom_client, cpu_query, start_time, end_time)
            memory_data = get_metric_data(prom_client, memory_query, start_time, end_time)
            
            cpu_values = [float(metric['values'][-1][1]) for metric in cpu_data]
            memory_values = [float(metric['values'][-1][1]) for metric in memory_data]
            
            cpu_recommendation = calculate_resource_recommendation(cpu_values)
            memory_recommendation = calculate_resource_recommendation(memory_values)

            lower_cpu_recommendation = cpu_recommendation * LOWER_BOUND_MULTIPLIER
            lower_memory_recommendation = memory_recommendation * LOWER_BOUND_MULTIPLIER
            upper_cpu_recommendation = cpu_recommendation * UPPER_BOUND_MULTIPLIER
            upper_memory_recommendation = memory_recommendation * UPPER_BOUND_MULTIPLIER

            bound_cpu_recommendation = bound_var(cpu_recommendation, lower_cpu_recommendation, upper_cpu_recommendation)
            bound_memory_recommendation = bound_var(memory_recommendation, lower_memory_recommendation, upper_memory_recommendation)
            bound_lower_cpu_recommendation = bound_var(lower_cpu_recommendation, lower_cpu_recommendation, upper_cpu_recommendation)
            bound_lower_memory_recommendation = bound_var(lower_memory_recommendation, lower_memory_recommendation, upper_memory_recommendation)
            bound_upper_cpu_recommendation = bound_var(upper_cpu_recommendation, lower_cpu_recommendation, upper_cpu_recommendation)
            bound_upper_memory_recommendation = bound_var(upper_memory_recommendation, lower_memory_recommendation, upper_memory_recommendation)

            string_cpu_recommendation = resource2str("cpu", bound_cpu_recommendation)
            string_memory_recommendation = resource2str("memory", bound_memory_recommendation)
            string_lower_cpu_recommendation = resource2str("cpu", bound_lower_cpu_recommendation)
            string_lower_memory_recommendation = resource2str("memory", bound_lower_memory_recommendation)
            string_upper_cpu_recommendation = resource2str("cpu", bound_upper_cpu_recommendation)
            string_upper_memory_recommendation = resource2str("memory", bound_upper_memory_recommendation)
            string_uncapped_cpu_recommendation = resource2str("cpu", cpu_recommendation)
            string_uncapped_memory_recommendation = resource2str("memory", memory_recommendation)

            if cpu_recommendation is None or memory_recommendation is None:
                continue
                
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
                    "cpu": string_uncapped_cpu_recommendation,
                    "memory": string_uncapped_memory_recommendation,
                }
            }
            recommendations.append(recommendation)
            
        except Exception as e:
            print(f"Error getting metrics for container {container_name}: {e}")
            continue
    
    return recommendations