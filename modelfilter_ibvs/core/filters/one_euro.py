# modelfilter_ibvs/core/filters/one_euro.py
from .base import BaseFilter

class OneEuroFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        # 1-Euro spezifische Parameter
        self.min_cutoff = 1.0
        self.beta = 0.0

    def set_params(self, min_cutoff, beta):
        self.min_cutoff = min_cutoff
        self.beta = beta
        print(f"[1-Euro] Parameter aktualisiert: min_cutoff={min_cutoff}, beta={beta}")
        
    # ... restliche Methoden (reset, initialize, predict, update, get_projected_points) ...