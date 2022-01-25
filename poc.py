import os
import time
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from collections import deque
import datetime

no_of_saver_processes = 1
world_size = 2#torch.cuda.device_count()

class feature_cache:
    def __init__(self):
        self.data = deque([])
    def __call__(self, t):
        self.data.append(t)
        print("Inserted into cache")
    def get(self):
        if len(self.data)>0:
            return dict(l=len(self.data),d=self.data.popleft())
        else:
            return dict(l=len(self.data),d="EMPTY")

def get_helper():
    return feature_cache_obj.get()

def flask_processing():
    print(f"Starting flask_processing")
    from flask import Flask
    app = Flask(__name__)
    global feature_cache_obj
    feature_cache_obj = feature_cache()
    @app.route('/')
    def hello():
        return feature_cache_obj.get()
    app.run(host='0.0.0.0',port='9999',debug=False)

def helper(t):
    global feature_cache_obj
    feature_cache_obj(t)
def cpu_process_initialization(rank):
    print(f"cpu_process_initialization with rank {rank}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5555'
    rpc.init_rpc(f"saver",
                 rank=rank,
                 world_size = world_size + no_of_saver_processes,
                 backend=rpc.BackendType.TENSORPIPE,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                     init_method='env://')
                 )
    print(f"Started CPU process {rank}")
    flask_processing()
    rpc.shutdown()
    return

def cuda_process_initialization(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5555'
    rpc.init_rpc(f"{rank}",
                 rank=rank,
                 world_size=world_size + no_of_saver_processes,
                 backend=rpc.BackendType.TENSORPIPE,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=0,
                                                                     init_method='env://')
                 )
    print(f"Started CUDA process on gpu {rank}")
    # Sleep to give the saver process sometime to initialize
    time.sleep(5)
    while True:
        print(f"{rank} Trying to insert into cache")
        _ = rpc.remote(f"saver",
                       helper,
                       timeout=0,
                       args=(dict(time=str(datetime.datetime.now()),
                                  data=(torch.ones(10)*rank).tolist()),
                             ))
        time.sleep(5)
    rpc.shutdown()
    return

if __name__ == "__main__":
    mp.set_start_method('forkserver', force=True)
    print("Starting CPU processes")
    p = mp.Process(target=cpu_process_initialization,
                   args=(world_size,))
    p.start()
    print("Starting CUDA processes")
    trainer_processes = mp.spawn(cuda_process_initialization,
                                 nprocs=world_size,
                                 join=False)
    print("Joining all processes")
    time.sleep(30)
    print("TERMINATING SAVER")
    p.terminate()
    trainer_processes.join()
    p.join()
    print("Processes joined ... Ending")