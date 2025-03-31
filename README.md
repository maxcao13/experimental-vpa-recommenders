# Shift Week Vertical Pod Autoscaler Recommenders

This repo contains some Vertical Pod Autoscaler (VPA) recommenders that were created during OpenShift "Shift Week" March 24 - 28, 2025 in celebration of the 4.18 OCP GA release.
Inspired by the work from [openshift/predictive-vpa-recommenders](https://github.com/openshift/predictive-vpa-recommenders)

These recommenders are meant to be run alongside VPA on vanilla Kubernetes.

<sub>It probably works on OpenShift with some tweaks as well, but I haven't tried it since you would have to enable feature-gates and stuff...</sub>

## Limitations

The `LinearRegression` Recommender currently does not work with multiple VPAs.

## Recommenders

### `Aggresive Recommender`

Path: `recommender/aggresive_recommender.py`

The Aggresive Recommender is designed to proactively optimize resource allocation by continuously analyzing the most recent real-time data and providing recommendations to precisely match resource usage.
The recommender leverages the upcoming `InPlaceOrRecreate` VPA feature gate which takes advantage of the Kubernetes feature gate `InPlacePodVerticalScaling`.
This allows the VPA updater to scale pod resource requests and limits while limiting disruptive updates as much as possible.

This strategy is best for environments with fluctuating workloads, where resource costs can be minimized by aligning resource allocation with demand.

### `LinearRegression Recommender`

Path: `recommender/ai_recommender.py`

The `LinearRegression Recommender` utilizes linear regression, a method that fits a linear equation to observed data, to try to predict future resource demands based on historical usage.
The algorithm processes time series data, using a specified lookback period to capture variations in resource consumption.
The recommender trains a linear model that attempts to predict the next value of resource usage, and then applies it as a recommendation for the VPA.
This models are re-trained every 5 minutes using the most recent historical data from Prometheus with a focus on giving more weight to recent data.

This is suited for workloads with stable or linear growth patterns, where it can anticipate future resource usage and potentially reduce costs.
Again, this recommender is best used with `InPlaceOrRecreate` mode to minimize disruptions during small resource updates.

## Prerequisites

* Kubernetes 1.32.0+ with `InPlacePodVerticalScaling` feature gate enabled
* Kubernetes Vertical Pod Autoscaler with `InPlaceOrRecreate` feature gate enabled
* Prometheus
* Python 3.x

Install the python dependencies in a virtual environment if you have one:

```bash
pip install -r requirements.txt
```

## Deployment

### Kubernetes 1.32+ (using [kind](https://kind.sigs.k8s.io/))

Simplest way to try this out is to install a `kind` cluster with `InPlacePodVerticalScaling`.

1. Install kind and prerequisites using the instructions: https://kind.sigs.k8s.io/
2. Create a kind config file like this and use it to create a cluster:

    ```yaml
    kind: Cluster
    apiVersion: kind.x-k8s.io/v1alpha4
    featureGates:
    InPlacePodVerticalScaling: true
    ```

    In a terminal:

    ```bash
    kind create cluster --config kind-config.yaml"
    ```

3. Install VPA with `InPlaceOrRecreate` feature gate alongside metrics server.

4. Install prometheus. The easiest way to do this is by using helm.

    ```bash
    $ helm upgrade --install -f prom_values.yaml prometheus prometheus-community/prometheus -n monitoring
    Release "prometheus" has been upgraded. Happy Helming!
    ...output omitted...
    ```

    This will install Prometheus server inside he cluster using helm with the `prom_values.yaml` overrides which disable some unneeded components.

5. You can run the recommenders locally, or within the cluster:

    **LOCAL:**

    Expose the prometheus server endpoint outside of the cluster in order for the recommender to query it:

    ```bash
    $ export POD_NAME=$(kubectl get pods --namespace monitoring -l "app.kubernetes.io/name=prometheus,app.kubernetes.io/instance=prometheus" -o jsonpath="{.items[0].metadata.name}")
    $ kubectl --namespace monitoring port-forward $POD_NAME 9090
    Forwarding from 127.0.0.1:9090 -> 9090
    ```

    Then in a separate terminal, you can run the recommender:

    ```bash
    $ python app.py
    ...output omitted...
    ```

    The recommender will attempt to use your local kubeconfig.

    **IN CLUSTER:**

    To run the recommender within the cluster, we need to build and push the image to an image repository. Or if you trust me enough, you can use the image here: `quay.io/macao/shift-week-03-24-25:latest`

    Then we need to install some Kubernetes RBAC in order for the recommenders to query Prometheus within the cluster and mutate VPA objects.

    We can do that by:

    ```bash
    $ kubectl apply -f deploy.yaml
    serviceaccount/pod-viewer unchanged
    clusterrole.rbac.authorization.k8s.io/pod-viewer-role unchanged
    clusterrolebinding.rbac.authorization.k8s.io/pod-viewer-binding unchanged
    deployment.apps/shift-week created
    ```

6. Last thing to do is to deploy some workloads!

    There are some examples within this repo you can try in `manifests/`

    To apply a workload for the regression recommender to use:

    ```bash
    kubectl apply -f manifests/ai-workload.yaml
    ```

    To apply an aggresive recommender workload and VPA:

    ```bash
    kubectl apply -f manifests/aggresive-workload.yaml
    ```

7. The last thing is that if you run the recommender locally, you are able to view plots of resource usage vs. the regression recommender's predictions and compare them. You can also view logs and statistics from running the python file that can show some interesting information as well.

    There is an example plot in `example_plots/resoruce_usage_hamster_1h.png`.

### OpenShift 4.x

TODO. Not tested as of now.

## Configuration

There is a `ConfigMap` manifest called `recommender-config` in `manifests/deploy.yaml` called that allow you to configure some details about the recommender.
In order for config changes to act, you must apply the new `recommender-config` and restart the `macao-recommender` pod.

Full list of configurable options should be in `recommender/recommender_config.py`. Note that not everything has been configurable due to time constraints.
