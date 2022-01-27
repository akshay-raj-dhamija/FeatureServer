"""
This file is the entry point for running the feature server.
It is responsible for starting the two types of processes.
The first process runs the cache manager and the flask server.
The rest of the processes have one gpu dedicated to each of them to run the feature extraction.
All processes communicate with each other using RPC.
"""
import resource

# Removing ulimit constraint, needed for multiprocessing
resource.setrlimit(resource.RLIMIT_OFILE, (11111, 11111))
import argparse
import pathlib
import requests
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from vast.tools import logger as vastlogger
from FeatureServer import api_plus_cacher
from FeatureServer import extractor_interface


def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
        description="This is the entry script to running the feature server",
    )

    parser.add_argument(
        "-v", "--verbose", help="To decrease verbosity increase", action="count", default=0
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="debugging flag"
    )
    parser.add_argument(
        "--mp_port_no",
        default="5555",
        type=str,
        help="port number for multiprocessing\ndefault: %(default)s",
    )
    parser.add_argument(
        "--api_port_no",
        default="9999",
        type=str,
        help="port number on which api server will run\ndefault: %(default)s",
    )
    parser.add_argument(
        "--api_ip_address",
        default="0.0.0.0",
        type=str,
        help="IP address at which to run the flask server\ndefault: %(default)s",
    )
    parser.add_argument(
        "--logs_output_dir",
        type=pathlib.Path,
        default=pathlib.Path("/scratch/adhamija/feature_server/logs"),
        help="Logs directory",
    )
    parser.add_argument(
        "--network_architecture",
        help="The network architecture for which a layer needs to be extracted from the network",
        default="resnet50",
    )
    parser.add_argument(
        "--layer_names",
        nargs="+",
        help="The layers to extract from the network\ndefault: %(default)s",
        default=["avgpool"],
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size on each GPU\ndefault: %(default)s",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        help="number of workers for dataloading used by a single inference process\ndefault: %(default)s",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--csv_path",
        help="The path to the csv file containing the list of images to load\ndefault: %(default)s",
        type=pathlib.Path,
        default="/home/adhamija/The/protocol/train_knowns.csv",
    )
    parser.add_argument(
        "--dataset_path",
        help="The path to the dataset containing all the images to be loaded\ndefault: %(default)s",
        type=pathlib.Path,
        default="/scratch/datasets/ImageNet/ILSVRC_2012",
    )
    # TODO - only initializes, not yet used
    parser.add_argument(
        "--network_weights",
        help="The weights used to initialize the network architecture with\ndefault: %(default)s",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--augmentations_per_image",
        help="number of augmentations to perform per image\ndefault: %(default)s",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = command_line_options()

    args.logs_output_dir.mkdir(parents=True, exist_ok=True)
    logger = vastlogger.setup_logger(
        level=args.verbose, output=args.logs_output_dir / "main.log"
    )

    args.world_size = torch.cuda.device_count()
    logger.info(f"Found {args.world_size} GPU's would be running inferencing on all")
    args.no_of_processes = args.world_size + 1

    logger.info("Starting processes")
    mp.set_start_method("forkserver", force=True)

    logger.info("Starting API & caching process")
    api_and_caching_process = mp.Process(
        target=api_plus_cacher.starter, args=(args.world_size, args)
    )
    api_and_caching_process.start()

    logger.info("Starting feature extraction processes")
    feature_extraction_processes = mp.spawn(
        extractor_interface.inference_process,
        args=(args,),
        nprocs=args.world_size,
        join=False,
    )
    logger.info("Joining all processes")
    logger.critical(" IMPORTANT: For a graceful shutdown enter [y/Y] ".center(90, "-"))
    s = input()
    logger.critical(f"Input recieved {s}")
    if s == "y" or s == "Y":
        logger.info("Calling shutdown API")
        requests.post("http://localhost:9999/shutdown")
    feature_extraction_processes.join()
    api_and_caching_process.join()
    logger.info("Processes joined ... Ending")
