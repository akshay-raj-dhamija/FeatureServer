import itertools
import bisect
import datetime
from collections import deque

# import zlib, json
# from base64 import b64encode, b64decode


class cache:
    def __init__(self):
        self.data = deque([])
        self.time_stamp = deque([])

    def __call__(self, data):
        # compressed_data = zlib.compress(json.dumps(t).encode('utf-8'))
        # compressed_str = b64encode(compressed_data).decode('ascii')
        # self.data.append(compressed_str)
        self.data.append(data)
        self.time_stamp.append(datetime.datetime.now().timestamp())

    def reset_cache_size(self, max_len):
        assert (
            len(self.data) == 0
        ), f"Q is not empty, you will loose {len(self.data)}  entries"
        self.data = deque([], maxlen=int(max_len))
        self.time_stamp = deque([], maxlen=int(max_len))

    def get(self):
        if len(self.data) > 0:
            return dict(len=len(self.data), z=self.data[0])
        else:
            return dict(len=len(self.data), z="EMPTY")

    def get_after(self, get_after_timestamp):
        idx = bisect.bisect(self.time_stamp, get_after_timestamp)
        z = list(itertools.islice(self.data, idx + 1, len(self.data)))
        return dict(len=len(self.data), z=z)
