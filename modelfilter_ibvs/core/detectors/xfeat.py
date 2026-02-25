from .base import BaseExtractor
import torch

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