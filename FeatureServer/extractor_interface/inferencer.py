import os
import time
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed.rpc as rpc
from vast import data_prep
from vast.tools import model_operations
from vast.tools import logger as vastlogger
from FeatureServer.api_plus_cacher import starter
from FeatureServer.extractor_interface import custom_sampler

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

        # Dataset initialization
        dataset = data_prep.ImageNetPytorch(
            csv_file=args.csv_path, images_path=args.dataset_path
        )

        # Reset cache queue size according to dataset size
        if rank == 0:
            _ = rpc.remote(
                "saver",
                starter.reset_q_size,
                timeout=0,
                args=(len(dataset),),
            )

        # Dataset sampler and loader
        sampler = custom_sampler(dataset,
                                 num_replicas = args.world_size,
                                      rank = rank,
                                      shuffle = True,
                                      seed = 0,
                                      drop_last = False)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
        )

        # Model initialization
        self.modelObj = model_operations(
            args.network_architecture, args.network_weights, args.layer_names
        )
        self.modelObj.model.eval()
        self.modelObj.model.cuda()

        # Start inferencing
        self.run(rank)

        # Inferencing done, exiting
        inference_process.logger.info("Sleeping for five minutes to let dataloaders shutdown")
        time.sleep(300)
        inference_process.logger.info(f"Shutting down inferencer with PID {os.getpid()}")
        rpc.shutdown()
        return

    @classmethod
    def shutdown(cls):
        cls.keep_running = False

    def run(self, rank):
        inference_process.logger.info("Running inferencer")
        for i, data in enumerate(self.dataloader):
            images, labels, class_names, image_ids = data
            inference_process.logger.debug(
                f"{images.shape}, {labels.shape}, {len(class_names)}, {len(image_ids)}"
            )
            with torch.no_grad():
                features_dict = self.modelObj(images.cuda())
            for k in features_dict:
                features_dict[k] = torch.squeeze(features_dict[k]).cpu().tolist()
            _ = rpc.remote(
                "saver",
                starter.cacher,
                timeout=0,
                args=(features_dict,),
            )
            if not inference_process.keep_running:
                break
        return
