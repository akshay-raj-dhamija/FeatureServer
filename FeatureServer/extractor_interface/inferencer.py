import os
import time
import datetime
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.rpc as rpc
from vast import data_prep

# from vast.tools import model_operations
from vast.tools import logger as vastlogger
from FeatureServer.api_plus_cacher import starter


class inference_process:
    keep_running = True
    logger = None

    def __init__(self, rank, args):
        inference_process.logger = vastlogger.get_logger(
            level=args.verbose,
            output=args.logs_output_dir / "inferencer.log",
            distributed_rank=rank,
            world_size=args.no_of_processes,
        )
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.mp_port_no
        rpc.init_rpc(
            f"{rank}",
            rank=rank,
            world_size=args.no_of_processes,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                rpc_timeout=0, init_method="env://"
            ),
        )
        inference_process.logger.info(
            f"Started inferencer with PID {os.getpid()} and rank {rank}"
        )
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank}"
        # Sleep to give the saver process sometime to initialize
        time.sleep(5)
        """
        modelObj = model_operations(args, gpu)
        modelObj.model.eval()
        modelObj.model.cuda()
        if args.world_size > 1:
            modelObj.model = DistributedDataParallel(modelObj.model, device_ids=[gpu])

            data_loader = data_prep.ImageNetPytorch()
            for i, data in enumerate(data_loader, start=0):
                features, Logit = modelObj(x)
        """
        self.run(rank)
        inference_process.logger.info(f"Shutting down inferencer with PID {os.getpid()}")
        rpc.shutdown()
        return

    @classmethod
    def shutdown(cls):
        cls.keep_running = False

    def run(self, rank):
        inference_process.logger.info("Running inferencer")
        while inference_process.keep_running:
            _ = rpc.remote(
                "saver",
                starter.cacher,
                timeout=0,
                args=(
                    dict(
                        time=str(datetime.datetime.now()),
                        data=(torch.ones(10) * rank).tolist(),
                    ),
                ),
            )
            time.sleep(2)
        return
