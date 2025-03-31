from kubernetes import client, config
from kubernetes.client.rest import ApiException
import time
import os
import traceback
from prometheus_api_client import PrometheusConnect
from recommender.aggresive_recommender import aggressive_get_recommendation
from recommender.ai_recommender import ai_get_recommendation
from recommender.recommender_config import RECOMMENDER_INTERVAL, PROMETHEUS_URL
from utils import selectsRecommender, AGGRESSIVE_RECOMMENDER_NAME, AI_RECOMMENDER_NAME

DOMAIN = "autoscaling.k8s.io"
VPA_NAME = "verticalpodautoscaler"
VPA_PLURAL = "verticalpodautoscalers"

def get_recommendation(recommender_name, vpa, corev1, prom):
    if recommender_name == AGGRESSIVE_RECOMMENDER_NAME:
        return aggressive_get_recommendation(vpa, corev1, prom)
    elif recommender_name == AI_RECOMMENDER_NAME:
        return ai_get_recommendation(vpa, corev1, prom)
    else:
        return None

def main():
    print("Starting the recommender...")
    # Load kubernetes configuration
    if 'KUBERNETES_PORT' in os.environ:
        print("Loading in-cluster config")
        config.load_incluster_config()
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

    else:
        print("Loading kube config")
        config.load_kube_config()
        prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)

    print(f"Prometheus URL: {prom.url}")

    # Get the api instance
    api_client = client.api_client.ApiClient()
    v1 = client.ApiextensionsV1Api(api_client)
    corev1 = client.CoreV1Api(api_client)
    crds = client.CustomObjectsApi(api_client)

    # Check if VPA CRD exists
    current_crds = [x['spec']['names']['kind'].lower() for x in v1.list_custom_resource_definition().to_dict()['items']]
    if VPA_NAME not in current_crds:
        print("VerticalPodAutoscaler CRD is not created!")
        exit(-1)

    runs = 1

    while True:
        try:
            print(f"Recommender Run {runs}")

            # LoadVPAs
            vpas = crds.list_cluster_custom_object(group=DOMAIN, version="v1", plural=VPA_PLURAL)

            observed_vpas = selectsRecommender(vpas)
            
            # UpdateVPAs
            for vpa in observed_vpas:
                vpa_name = vpa["metadata"]["name"]
                vpa_namespace = vpa["metadata"]["namespace"]
                recommender_name = vpa["spec"]["recommenders"][0]["name"]
                recommendations = get_recommendation(recommender_name,vpa, corev1, prom)

                if not recommendations:
                    print("No new recommendations obtained, so skip updating the vpa object {}".format(vpa_name))
                    continue

                patched_vpa = {
                    "recommendation": {
                        "containerRecommendations": recommendations
                    },
                    "conditions": [
                        {
                            "type": "RecommendationProvided",
                            "status": "True",
                        }
                    ]
                }
                body = {"status": patched_vpa}
                try:
                    print(f"Patching VPA {vpa_name} with recommendation: {recommendations}")
                    crds.patch_namespaced_custom_object_status(
                        group=DOMAIN,
                        version="v1",
                        plural=VPA_PLURAL,
                        namespace=vpa_namespace,
                        name=vpa_name,
                        body=body
                    )
                except ApiException:
                    print(f"Error patching VPA {vpa_name}:")
                    traceback.print_exc()

            time.sleep(RECOMMENDER_INTERVAL)
            runs += 1
        except Exception:
            print(f"Error in main loop:")
            traceback.print_exc()
            time.sleep(RECOMMENDER_INTERVAL)

if __name__ == "__main__":
    main() 
