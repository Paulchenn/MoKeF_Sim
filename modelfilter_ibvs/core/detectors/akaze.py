import cv2
import numpy as np
from .base import BaseExtractor

class AkazeExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.detector = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def set_reference_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.detector.detectAndCompute(gray, None)
        print(f"[AKAZE] Reference Image set! Found {len(self.ref_kp)} features.")

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