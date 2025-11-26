class MockEvent:
    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass


class MockStream:
    def wait_stream(self, other):
        pass

    def wait_event(self, event):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockStreamManager:
    """Drop-in replacement for your StreamManager"""

    def __init__(self, device):
        self.device = device
        self.compute_stream = MockStream()
        self.comm_stream = MockStream()

    def wait_comm(self):
        pass

    def wait_compute(self):
        pass
