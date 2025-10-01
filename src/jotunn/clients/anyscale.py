import anyscale
from anyscale.compute_config.models import (
    ComputeConfig,
    HeadNodeConfig,
    MarketType,
    WorkerNodeGroupConfig,
)
from anyscale.job.models import JobConfig


class AnyscaleCluster:
    def __init__(
        self,
        cloud="anyscale",
        head_node_type="m4.xlarge",
        worker_node_type="g4dn.xlarge",
        min_worker_nodes=2,
        max_worker_nodes=2,
        market_type=MarketType.SPOT,
        job_name="my-job",
        entrypoint="python run.py",
        working_dir=".",
        max_retries=1,
        requirements="requirements.txt",
    ):
        self.cloud = cloud
        self.head_node_type = head_node_type
        self.worker_node_type = worker_node_type
        self.min_worker_nodes = min_worker_nodes
        self.max_worker_nodes = max_worker_nodes
        self.market_type = market_type
        self.job_name = job_name
        self.entrypoint = entrypoint
        self.working_dir = working_dir
        self.max_retries = max_retries
        self.requirements = requirements

    def compute_config(self):
        return ComputeConfig(
            cloud=self.cloud,
            head_node=HeadNodeConfig(
                instance_type=self.head_node_type,
            ),
            worker_nodes=[
                WorkerNodeGroupConfig(
                    instance_type=self.worker_node_type,
                    min_nodes=self.min_worker_nodes,
                    max_nodes=self.max_worker_nodes,
                    market_type=self.market_type,
                ),
            ],
        )

    def job_config(self):
        return JobConfig(
            name=self.job_name,
            entrypoint=self.entrypoint,
            working_dir=self.working_dir,
            max_retries=self.max_retries,
            requirements=self.requirements,
            compute_config=self.compute_config(),
        )

    def submit_job(self):
        anyscale.job.submit(self.config())
