from torch.quantization import default_weight_observer, default_histogram_observer
from torch.quantization import RecordingObserver as _RecordingObserver

__all__ = ["RecordingObserver", "default_weight_observer", "default_histogram_observer"]


class RecordingObserver(_RecordingObserver):
    """
    A extended version of PyTorch's RecordingObserver, used to record gpu tensor
    """

    def forward(self, x):
        val = x.cpu()
        super().forward(val)
        return x
