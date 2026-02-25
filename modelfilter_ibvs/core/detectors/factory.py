# modelfilter_ibvs/core/detectors/factory.py

# Imports der einzelnen Implementierungen
from .orb import OrbExtractor
from .sift import SiftExtractor
from .akaze import AkazeExtractor
from .xfeat import XFeatExtractor
from .superpoint import SuperPointExtractor
from .aliked import AlikedExtractor

class ExtractorFactory:
    """Factory zur dynamischen Erstellung von Feature Extractors."""
    
    # Hier registrieren wir alle Detektoren mit ihrem Anzeigenamen für die GUI
    _registry = {
        "ORB (OpenCV)": OrbExtractor,
        "SIFT (OpenCV)": SiftExtractor,
        "AKAZE (OpenCV)": AkazeExtractor,
        "XFeat (CNN)": XFeatExtractor,
        "SuperPoint (CNN)": SuperPointExtractor,
        "ALIKED (CNN)": AlikedExtractor
    }

    @classmethod
    def get_available_extractors(cls):
        """Gibt eine Liste der Namen für das Dropdown-Menü zurück."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name):
        """Instanziiert den ausgewählten Extractor."""
        if name not in cls._registry:
            raise ValueError(f"Extractor '{name}' ist nicht registriert.")
        return cls._registry[name]()