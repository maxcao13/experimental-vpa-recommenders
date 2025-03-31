# many functions from https://github.com/openshift/predictive-vpa-recommenders

AI_RECOMMENDER_NAME = "ai"
AGGRESSIVE_RECOMMENDER_NAME = "aggressive"

def selectsRecommender(vpas):
    selected_vpas = []
    for vpa in vpas["items"]:
        vpa_spec = vpa["spec"]
        if "recommenders" not in vpa_spec.keys():
            continue
        else:
            for recommender in vpa_spec["recommenders"]:
                if recommender["name"] == AI_RECOMMENDER_NAME or recommender["name"] == AGGRESSIVE_RECOMMENDER_NAME:
                    selected_vpas.append(vpa)

    return selected_vpas

# resource2str converts a resource (CPU, Memory) value to a string
def resource2str(resource, value):
    if resource.lower() == "cpu":
        if value < 1:
            return str(int(value * 1000)) + "m"
        else:
            return str(value)
    else:
        if value < 1024:
            return str(value) + "B"
        elif value < 1024 * 1024:
            return str(int(value / 1024)) + "k"
        elif value < 1024 * 1024 * 1024:
            return str(int(value / 1024 / 1024)) + "Mi"
        else:
            return str(int(value / 1024 / 1024 / 1024)) + "Gi"

# Convert a resource (CPU, Memory) string to a float value
def str2resource(resource, value):
    if type(value) is str:
        if resource.lower() == "cpu":
            if value[-1] == "m":
                return float(value[:-1]) / 1000
            else:
                return float(value)
        else:
            if value[-1].lower() == "b":
                return float(value[:-1])
            elif value[-1].lower() == "k":
                return float(value[:-1]) * 1024
            elif value[-2:].lower() == "mi":
                return float(value[:-2]) * 1024 * 1024
            elif value[-2:].lower() == "gi":
                return float(value[:-2]) * 1024 * 1024 * 1024
            else:
                return float(value)
    else:
        return value
        
def get_target_containers(corev1_client, target_namespace, target_ref):
    target_pods = corev1_client.list_namespaced_pod(namespace=target_namespace, label_selector="app=" + target_ref["name"])

    # Retrieve the target containers and pod names
    target_containers = []
    existing_pods = []
    for pod in target_pods.items:
        existing_pods.append(pod.metadata.name)
        for container in pod.spec.containers:
            if container.name not in target_containers:
                target_containers.append(container.name)

    return target_containers, existing_pods

def get_metric_data(prom_client, query, start_time, end_time, step='30s'):
    """
    Get metric data from Prometheus
    """
    try:
        result = prom_client.custom_query_range(
            query,
            start_time=start_time,
            end_time=end_time,
            step=step,
            timeout=30
        )
    except Exception as e:
        print(f"Error getting metric data: {e}")
        return None
    return result

def bound_var(var, min_value, max_value):
    if var < min_value:
        return min_value
    elif var > max_value:
        return max_value
    else:
        return var