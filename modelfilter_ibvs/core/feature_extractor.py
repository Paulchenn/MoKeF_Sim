import cv2
import numpy as np
import torch

class BaseExtractor:
    """Basisklasse (Interface) für alle Feature Extractors."""
    def __init__(self):
        self.ref_kp = None
        self.ref_des = None

    def set_reference_image(self, img):
        raise NotImplementedError("Muss in der Unterklasse implementiert werden.")

    def match_current_image(self, img):
        raise NotImplementedError("Muss in der Unterklasse implementiert werden.")
    
    def get_reference_coordinates(self):
        """Gibt die 2xN Koordinaten aller gespeicherten Soll-Punkte zurück."""
        raise NotImplementedError("Muss in der Unterklasse implementiert werden.")


class OrbExtractor(BaseExtractor):
    """ORB Implementierung."""
    def __init__(self):
        super().__init__()
        self.detector = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def set_reference_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.detector.detectAndCompute(gray, None)
        print(f"[ORB] Reference Image set! Found {len(self.ref_kp)} features.")

    def match_current_image(self, img):
        if self.ref_des is None:
            return None, None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = self.detector.detectAndCompute(gray, None)
        
        if curr_des is None or len(curr_des) < 4:
            return None, None
            
        matches = self.bf.match(self.ref_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) < 4:
            return None, None
            
        pts_ref = np.float32([self.ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2).T
        pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2).T
        
        return pts_curr, pts_ref

    def get_reference_coordinates(self):
        if self.ref_kp is None: return None
        # Liste von KeyPoint-Objekten in ein 2xN Numpy Array wandeln
        return np.float32([kp.pt for kp in self.ref_kp]).T


class XFeatExtractor(BaseExtractor):
    """XFeat Implementierung."""
    def __init__(self):
        super().__init__()
        print("Lade XFeat Modell...")
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        print("XFeat bereit.")

    def set_reference_image(self, img):
        # Das Modell gibt eine Liste zurück, wir nehmen das erste Element [0]
        output = self.xfeat.detectAndCompute(img, top_k=4096)[0]
        self.ref_kp, self.ref_des = output['keypoints'], output['descriptors']
        print(f"[XFeat] Reference Image set! Found {len(self.ref_kp)} features.")

    def match_current_image(self, img):
        if self.ref_des is None:
            return None, None
            
        output = self.xfeat.detectAndCompute(img, top_k=4096)[0]
        curr_kp, curr_des = output['keypoints'], output['descriptors']

        if curr_des is None or len(curr_des) < 4:
            return None, None
        
        # XFeat Matching (hier nutzt du die Funktion deines Wrappers)
        idx0, idx1 = self.xfeat.match(self.ref_des, curr_des, 0.82)
        idx0, idx1 = idx0[:50], idx1[:50]  # Top 50 Matches
        
        if len(idx0) < 4:
            return None, None
        
        pts_ref = self.ref_kp[idx0].cpu().numpy().T
        pts_curr = curr_kp[idx1].cpu().numpy().T
        
        return pts_curr, pts_ref
    
    def get_reference_coordinates(self):
        if self.ref_kp is None: return None
        # PyTorch Tensor in ein 2xN Numpy Array wandeln
        return self.ref_kp.cpu().numpy().T

# Ein Dictionary, um in der GUI einfach über Strings darauf zuzugreifen
AVAILABLE_EXTRACTORS = {
    "XFeat (CNN/Robust)": XFeatExtractor,
    "ORB (Fast/Fallback)": OrbExtractor
}