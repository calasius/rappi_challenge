import time
import psutil
import threading
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import log

logger = log.setup_custom_logger(__name__)


class SystemUsageProfiler:

    def __init__(self, process_name, update_period=300):
        self.process_name = process_name
        self.update_period = update_period
        self.stop = False
        self.registry = CollectorRegistry()
        self.system_usage = Gauge(self.process_name, 'Hold current system resource usage', ['resource_type'],
                                  registry=self.registry)
        thread = threading.Thread(target=self.profile, args=())
        thread.daemon = True
        thread.start()
        self.thread = thread

    def profile(self):
        while not self.stop:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()[2]
            self.push_data(cpu, memory)
            time.sleep(self.update_period)

    def push_data(self, cpu, memory):
        self.system_usage.labels('CPU').set(cpu)
        self.system_usage.labels('Memory').set(memory)
        push_to_gateway('localhost:9091', job=self.process_name, registry=self.registry)

    def finish(self):
        self.stop = True
        self.thread.join()
        self.push_data(0, 0)
