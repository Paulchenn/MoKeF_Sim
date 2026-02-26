# modelfilter_ibvs/core/filters/factory.py

# from .homography import HomographyFilter
from .skf import StandardKalmanFilter
from .ekf import ExtendedKalmanFilter
from .iekf import IteratedExtendedKalmanFilter
from .eskf import ErrorStateKalmanFilter
from .ukf import UnscentedKalmanFilter
from .one_euro import OneEuroFilter

class FilterFactory:
    _registry = {
        "- (Kein Filter)": None,
        #"Homography Model (SL3)": HomographyFilter,
        "Standard Kalman Filter": StandardKalmanFilter,
        "Extended Kalman Filter": ExtendedKalmanFilter,
        "Iterated Extended Kalman Filter": IteratedExtendedKalmanFilter,
        "Error-State Kalman Filter": ErrorStateKalmanFilter,
        "Unscented Kalman Filter": UnscentedKalmanFilter,
        "1-Euro Filter": OneEuroFilter
    }

    @classmethod
    def get_available_filters(cls):
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name, K):
        if name not in cls._registry:
            raise ValueError(f"Filter '{name}' ist nicht registriert.")
            
        FilterClass = cls._registry[name]
        if FilterClass is None:
            return None # Fallback für "Kein Filter"
            
        return FilterClass(K)