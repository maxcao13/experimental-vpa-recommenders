import yaml

try:
    config_file = 'config/recommender_config.yaml'
    config = yaml.load(open(config_file,"r"), Loader=yaml.FullLoader)
except Exception as e:
    config = {
        "DEFAULT_NAMESPACE": "default",
        "RECOMMENDER_INTERVAL": 60,
        "PLOT_TIME_RANGE": "1h",
        "PROMETHEUS_URL": "http://prometheus-server.monitoring.svc.cluster.local:80",
        "ENABLED_PLOTTING": True
    }

# Retrieve the configuration for the recommender
DEFAULT_NAMESPACE = config['DEFAULT_NAMESPACE']
RECOMMENDER_INTERVAL = config['RECOMMENDER_INTERVAL']
PLOT_TIME_RANGE = config['PLOT_TIME_RANGE']
PROMETHEUS_URL = config['PROMETHEUS_URL']
ENABLED_PLOTTING = config['ENABLED_PLOTTING']