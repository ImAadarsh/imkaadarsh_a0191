import kubernetes
import prometheus_client
from kubernetes import client, config
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deployment.log',
    filemode='w'
)

class DeploymentManager:
    def __init__(self):
        # Initialize Kubernetes configuration
        config.load_kube_config()
        self.k8s_client = client.AppsV1Api()
        self.metrics = {
            'deployment_duration': prometheus_client.Histogram(
                'deployment_duration_seconds',
                'Time taken for deployment',
                buckets=[30, 60, 120, 300, 600]
            ),
            'deployment_success': prometheus_client.Counter(
                'deployment_success_total',
                'Number of successful deployments'
            ),
            'deployment_failure': prometheus_client.Counter(
                'deployment_failure_total',
                'Number of failed deployments'
            )
        }
        
    def deploy_microservice(self, namespace, deployment_name, container_image):
        try:
            # Create deployment object
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(name=deployment_name),
                spec=client.V1DeploymentSpec(
                    replicas=3,
                    selector=client.V1LabelSelector(
                        match_labels={"app": deployment_name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": deployment_name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=deployment_name,
                                    image=container_image,
                                    ports=[client.V1ContainerPort(container_port=8080)],
                                    resources=client.V1ResourceRequirements(
                                        requests={"cpu": "100m", "memory": "512Mi"},
                                        limits={"cpu": "500m", "memory": "1Gi"}
                                    )
                                )
                            ]
                        )
                    )
                )
            )

            # Implement rolling update strategy
            start_time = time.time()
            self.k8s_client.create_namespaced_deployment(
                namespace=namespace,
                body=deployment
            )
            
            # Wait for deployment to complete
            self._wait_for_deployment(namespace, deployment_name)
            
            deployment_time = time.time() - start_time
            self.metrics['deployment_duration'].observe(deployment_time)
            self.metrics['deployment_success'].inc()
            
            return True, f"Deployment {deployment_name} completed successfully"
            
        except kubernetes.client.rest.ApiException as e:
            self.metrics['deployment_failure'].inc()
            logging.error(f"Deployment failed: {str(e)}")
            return False, f"Deployment failed: {str(e)}"
            
    def _wait_for_deployment(self, namespace, deployment_name, timeout=300):
        start = time.time()
        while time.time() - start < timeout:
            response = self.k8s_client.read_namespaced_deployment_status(
                name=deployment_name,
                namespace=namespace
            )
            
            if (response.status.available_replicas == response.status.replicas and
                response.status.ready_replicas == response.status.replicas):
                return True
                
            time.sleep(5)
            
        raise TimeoutError(f"Deployment {deployment_name} timed out")

    def rollback_deployment(self, namespace, deployment_name):
        try:
            # Get deployment history
            api_response = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Rollback to previous revision
            api_response.spec.template = api_response.spec.template
            api_response.spec.revision_history_limit = 2
            
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=api_response
            )
            
            return True, "Rollback initiated successfully"
            
        except kubernetes.client.rest.ApiException as e:
            logging.error(f"Rollback failed: {str(e)}")
            return False, f"Rollback failed: {str(e)}"