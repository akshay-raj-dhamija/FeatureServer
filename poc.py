import os
import time
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from collections import deque
import datetime

no_of_saver_processes = 1
world_size = 2  # torch.cuda.device_count()


class feature_cache:
    def __init__(self):
        self.data = deque([])

    def __call__(self, t):
        self.data.append(t)

    def get(self):
        if len(self.data) > 0:
            return dict(l=len(self.data), d=self.data.popleft())
        else:
            return dict(l=len(self.data), d="EMPTY")


def get_helper():
    return feature_cache_obj.get()


def flask_processing():
    print("Starting flask_processing")
    from flask import Flask
    from flask import request
    import logging

    app = Flask(__name__)
    # Suppressing flask logs only to error because we want the input to be accepted by main process
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    global feature_cache_obj
    feature_cache_obj = feature_cache()

    @app.route("/data")
    def hello():
        return feature_cache_obj.get()

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()
        return "Server shutting down..."

    app.run(host="0.0.0.0", port="9999", debug=False)
    return


def helper(t):
    global feature_cache_obj
    feature_cache_obj(t)


class cpu_process:
    def __init__(self, rank):
        print(f"cpu_process_initialization with rank {rank}")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "5555"
        rpc.init_rpc(
            "saver",
            rank=rank,
            world_size=world_size + no_of_saver_processes,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                rpc_timeout=0, init_method="env://"
            ),
        )
        print(f"Started CPU process {rank}")
        cpu_process.run()

    @staticmethod
    def run():
        flask_processing()
        cpu_process.shutdown()

    @staticmethod
    def shutdown():
        for i in range(world_size):
            _ = rpc.remote(f"{i}", cuda_process.shutdown, timeout=0)
        rpc.shutdown()
        return


class cuda_process:
    keep_running = True

    def __init__(self, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "5555"
        rpc.init_rpc(
            f"{rank}",
            rank=rank,
            world_size=world_size + no_of_saver_processes,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                rpc_timeout=0, init_method="env://"
            ),
        )
        print(f"Started CUDA process on gpu {rank}")
        # Sleep to give the saver process sometime to initialize
        time.sleep(5)
        self.run(rank)
        rpc.shutdown()
        return

    @classmethod
    def shutdown(cls):
        cls.keep_running = False

    def run(self, rank):
        print(f"Started CUDA process on gpu {rank}")
        while cuda_process.keep_running:
            _ = rpc.remote(
                "saver",
                helper,
                timeout=0,
                args=(
                    dict(
                        time=str(datetime.datetime.now()),
                        data=(torch.ones(10) * rank).tolist(),
                    ),
                ),
            )
            time.sleep(5)
        return


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    print("Starting CPU processes")
    p = mp.Process(target=cpu_process, args=(world_size,))
    p.start()
    print("Starting CUDA processes")
    trainer_processes = mp.spawn(
        cuda_process, nprocs=world_size, join=False  # cuda_process_initialization,
    )
    print("Joining all processes")
    print(" IMPORTANT: For a graceful shutdown enter [y/Y] ".center(90, "-"))
    s = input()
    print(f"Registered {s}")
    if s == "y" or s == "Y":
        import requests

        requests.post("http://localhost:9999/shutdown")
    trainer_processes.join()
    p.join()
    print("Processes joined ... Ending")
