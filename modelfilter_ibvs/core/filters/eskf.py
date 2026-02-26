# modelfilter_ibvs/core/filters/ekf.py
from .base import BaseFilter
import numpy as np

class ErrorStateKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        # Standard-Parameter
        self.Q_scalar = 0.01
        self.R_scalar = 1.0

    def set_Q_R(self, q_val, r_val):
        """Wird von der GUI aufgerufen, um Parameter live zu ändern."""
        self.Q_scalar = q_val
        self.R_scalar = r_val
        print(f"[EKF] Parameter aktualisiert: Q={q_val}, R={r_val}")

    def reset(self):
        super().reset()
        # Hier Zustandsvektor P und x zurücksetzen

    def initialize(self, current_pixels, desired_pixels):
        self.initialized = True
        print("[EKF] Initialisiert!")

    def predict(self, v_cam, dt):
        pass # Hier kommt deine EKF-Prädiktion hin

    def update(self, current_pixels, desired_pixels):
        pass # Hier kommt dein EKF-Update hin

    def get_projected_points(self, desired_pixels):
        return desired_pixels # Hier Rückprojektion einbauen