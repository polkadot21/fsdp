import torch


class StreamManager:
    """
    The 'Manager's Office'.
    Keeps track of the Assembly Line (Compute) and Logistics (Comm).
    """

    def __init__(self, device: torch.device):
        self.device = device

        # Priority 0 (Default): Compute / Assembly Line
        self.compute_stream = torch.cuda.current_stream(device)

        # Priority -1 (High): Communication / Logistics
        # We give Comm high priority so NCCL kernels launch immediately.
        self.comm_stream = torch.cuda.Stream(device=device, priority=-1)

    def wait_comm(self):
        """Compute waits for Comm (Assembly waits for parts)."""
        self.compute_stream.wait_stream(self.comm_stream)

    def wait_compute(self):
        """Comm waits for Compute (Logistics waits for Assembly to finish with the buffer)."""
        self.comm_stream.wait_stream(self.compute_stream)
