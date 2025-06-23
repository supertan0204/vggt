import os
import time
import torch

def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ["LOCAL_RANK"]) if os.environ.get("LOCAL_RANK") is not None else None
    distributed_rank = int(os.environ["RANK"]) if os.environ.get("RANK") is not None else None
    assert (
        local_rank is not None and distributed_rank is not None
    ), "Please the set the RANK and LOCAL_RANK environment variables."
    return local_rank, distributed_rank
