# modelfilter_ibvs/core/detectors/superpoint.py
from .base import BaseExtractor

class AlikedExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        print("Aliked initialisiert (Achtung: Logik muss noch implementiert werden!)")

    def set_reference_image(self, img):
        pass # Hier kommt später dein PyTorch Code

    def match_current_image(self, img):
        return None, None 

    def get_reference_coordinates(self):
        return None