class cache:
    def __init__(self):
        self.data = []

    def __call__(self, t):
        self.data.append(t)

    def get(self):
        if len(self.data) > 0:
            return dict(l=len(self.data), d=self.data.pop(0))
        else:
            return dict(l=len(self.data), d="EMPTY")
