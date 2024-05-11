import numpy as np

class WFSFilter:
    def __init__(self, n, a):
        self.n = n
        self.a = a
        self.reset()
        
    def reset(self):
        self.last_out = np.zeros(self.n)
        self.last_in = np.zeros(self.n)

class HighPassFilter(WFSFilter):
    """
    Class to handle the high-pass filter after the first-stage WFS.
    """
    def __call__(self, inp):
        out = self.a * self.last_out + self.a * (inp - self.last_in)
        self.last_in, self.last_out = np.copy(inp), np.copy(out)
        return out

class LowPassFilter(WFSFilter):
    """
    Class to handle the low-pass filter after the second-stage WFS.
    """
    def __call__(self, inp):
        out = (1 - self.a) * inp + self.a * self.last_out
        self.last_out = np.copy(out)
        return out
