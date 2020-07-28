from prometheus_client import Gauge, generate_latest, CollectorRegistry
import psutil


class SystemResourceMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.system_usage = Gauge('prediction_api_system_resources', 'Hold current system resource usage', ['resource_type'],
                                  registry=self.registry)

    def collect_metrics(self):
        process = psutil.Process()
        cpu = process.cpu_percent(interval=1)
        memory = process.memory_percent()
        self.system_usage.labels('CPU').set(cpu)
        self.system_usage.labels('Memory').set(memory)

        return generate_latest(self.registry)


system_resource_metrics = SystemResourceMetrics()
