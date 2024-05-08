import numpy as np

class WFSFilter:
    """
    Class to handle the HPF/LPF wavefront sensor output filtering.
    """
    def __init__(self, n, a):
        self.n = n
        self.a = a
        self.last_hpf = np.zeros(n)
        self.last_lpf = np.zeros(n)
        self.last_inp = np.zeros(n)
        
    def __call__(self, inp):
        hpf = inp - (self.a * self.last_hpf + (1 - self.a) * self.last_inp)
        lpf = self.a * self.last_lpf + (1 - self.a) * self.last_inp
        self.last_hpf, self.last_lpf, self.last_inp = hpf, lpf, inp
        return hpf, lpf
        
    