from collections import deque


class cache:
    def __init__(self):
        self.data = deque([])

    def __call__(self, t):
        self.data.append(t)

    def get(self):
        if len(self.data) > 0:
            return dict(len=len(self.data), z=self.data.popleft())
        else:
            return dict(len=len(self.data), z="EMPTY")
