# modelfilter_ibvs/core/filters/base.py

class BaseFilter:
    """Interface für alle Keypoint-Filter."""
    def __init__(self, K):
        self.K = K
        self.initialized = False

    def reset(self):
        self.initialized = False

    def initialize(self, current_pixels, desired_pixels):
        raise NotImplementedError("Muss implementiert werden.")

    def predict(self, v_cam, dt):
        raise NotImplementedError("Muss implementiert werden.")

    def update(self, current_pixels, desired_pixels):
        raise NotImplementedError("Muss implementiert werden.")

    def get_projected_points(self, desired_pixels):
        raise NotImplementedError("Muss implementiert werden.")