"""
This file is responsible for the following:
    starting the flask API server
    Initializing the cache
    Providing the function call for updating the cache
    Sending the shutdown signal to the inferencing processes
"""
import os
import time
import torch.distributed.rpc as rpc
from vast.tools import logger as vastlogger
from FeatureServer.api_plus_cacher import cache
from FeatureServer.api_plus_cacher import api_interface


class starter:
    cache_obj = cache()
    logger = None
    args = None

    def __init__(self, rank, args):
        starter.logger = vastlogger.get_logger(
            level=args.verbose,
            output=args.logs_output_dir / "cacher.log",
            distributed_rank=rank,
            world_size=args.no_of_processes,
        )
        starter.args = args
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.mp_port_no
        rpc.init_rpc(
            "saver",
            rank=rank,
            world_size=args.no_of_processes,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                rpc_timeout=0, init_method="env://"
            ),
        )
        starter.logger.info(f"Started cacher with PID {os.getpid()} and rank {rank}")
        api_interface(starter.cache_obj, starter.logger)
        starter.shutdown()

    @staticmethod
    def reset_q_size(max_len):
        starter.cache_obj.reset_cache_size(starter.args.max_cache_size_factor * max_len)

    @staticmethod
    def shutdown():
        from FeatureServer.extractor_interface import inference_process

        for i in range(starter.args.world_size):
            starter.logger.info(f"Sending shutdown signal to {i}")
            _ = rpc.remote(f"{i}", inference_process.shutdown, timeout=0)
        starter.logger.info("Sleeping for five minutes to let dataloaders shutdown")
        starter.logger.info(
            "Please note while this seems to work in some cases if you specified --workers > 1"
            "on the command line, there is a high probability you will be shown error messages"
            "now even though inferences processes have been shutdown successfully. "
            "Donot panic. Kindly use `killall python` to kill zombies."
        )
        time.sleep(300)
        rpc.shutdown()
        return

    @staticmethod
    def cacher(data):
        starter.cache_obj(data)
