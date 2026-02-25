# modelfilter_ibvs/core/detectors/base.py
import numpy as np

class BaseExtractor:
    """Interface für alle Keypoint-Detektoren."""
    def __init__(self):
        self.ref_kp = None
        self.ref_des = None

    def set_reference_image(self, img):
        raise NotImplementedError("Muss implementiert werden.")

    def match_current_image(self, img):
        raise NotImplementedError("Muss implementiert werden.")

    def get_reference_coordinates(self):
        raise NotImplementedError("Muss implementiert werden.")