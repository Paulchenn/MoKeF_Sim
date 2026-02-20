import numpy as np
import cv2
from modelfilter_ibvs.core.core import skew, flatten_H, unflatten_H

class BaseFilter:
    """Basisklasse für alle Keypoint-Filter."""
    def __init__(self, K):
        self.K = K
        self.initialized = False
        self.status = "INIT"

    def reset(self):
        self.status = "INIT"
        self.initialized = False

    def initialize(self, current_pixels, desired_pixels):
        pass

    def predict(self, v_cam, dt):
        pass

    def update(self, current_pixels, desired_pixels):
        pass

    def get_projected_points(self, desired_pixels):
        return desired_pixels
    

class HomographyFilter(BaseFilter):
    """Dein bestehender lie-gruppen basierter Homographie-Filter."""
    def __init__(self, K, alpha=0.9):
        super().__init__(K)
        self.K_inv = np.linalg.inv(K)
        self.H_est = np.eye(3) 
        self.n_star = np.array([0, 0, 1])
        self.alpha = alpha # Glättungsfaktor für die Messung (0.1 = 10% Messung, 90% Modell)

    def reset(self):
        super().reset()
        self.H_est = np.eye(3)
        print("HomographyFilter: Zurückgesetzt!")

    def pixel2norm(self, uv):
        """Hilfsfunktion: Pixel (2xN) -> Normalisiert (3xN homogen)"""
        num = uv.shape[1]
        uv_hom = np.vstack((uv, np.ones((1, num))))
        return self.K_inv @ uv_hom

    def initialize(self, current_pixels, desired_pixels):
        """Initialisiert H basierend auf den ersten sauberen Punkten"""
        # Wir berechnen H im normalisierten Raum!
        p_curr = self.pixel2norm(current_pixels) # 3xN
        p_des = self.pixel2norm(desired_pixels)  # 3xN
        
        # cv2 braucht Nx2 (x,y)
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        # findHomography: src = H * dst
        # Also p_curr = H * p_des
        H_meas, _ = cv2.findHomography(dst, src)
        
        if H_meas is not None:
            self.H_est = H_meas
            self.initialized = True
            print("Filter: Initialisiert (Normalized Space)!")

    def predict(self, v_cam, d_star, dt):
        """
        Der "Blindflug"-Schritt.
        v_cam: Twist der Kamera [v, w]
        """
        if not self.initialized: return

        v = v_cam[:3]
        w = v_cam[3:]
        
        # Skew-Symmetric Matrix für Winkelgeschwindigkeit [w]x
        w_x = np.array([
            [0,     -w[2],  w[1]],
            [w[2],   0,    -w[0]],
            [-w[1],  w[0],  0   ]
        ])
        
        # Term für lineare Geschwindigkeit: (1/d) * v * n^T
        n_vec = self.n_star.reshape(3, 1)
        v_vec = v.reshape(3, 1)
        
        T_v = (1.0 / d_star) * (v_vec @ n_vec.T)
        
        # Die Differentialgleichung für H (Motion composition)
        # WICHTIG: Das Vorzeichen! 
        # Wenn Kamera sich bewegt, ändert sich H entgegengesetzt zur Bildbewegung?
        # Standard Literatur (Benhimane & Malis): H_dot = - (w_x * H + T_v * H)
        # Wir probieren hier das Minuszeichen für den gesamten Term.
        
        H_dot = - (w_x + T_v) @ self.H_est
        
        # Integration
        self.H_est = self.H_est + H_dot * dt
        
        # Normalisierung (Determinante = 1)
        det = np.linalg.det(self.H_est)
        if det > 0:
            self.H_est = self.H_est / np.cbrt(det)
        self.status = "PREDICT"

    def update(self, current_pixels, desired_pixels):
        """Fusion mit Messung"""
        if not self.initialized:
            self.initialize(current_pixels, desired_pixels)
            return
        
        if current_pixels is None or desired_pixels is None:
            self.status = "MISSING"
            return

        # 1. Messung im normalisierten Raum
        p_curr = self.pixel2norm(current_pixels)
        p_des = self.pixel2norm(desired_pixels)
        
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        H_meas, _ = cv2.findHomography(dst, src)
        
        if H_meas is None:
            self.status = "MISSING"
            return 
            
        # 2. Komplementärfilter auf Lie-Gruppe (vereinfacht linear)
        self.H_est = self.alpha * H_meas + (1.0 - self.alpha) * self.H_est
        
        # Normalisieren
        det = np.linalg.det(self.H_est)
        if det > 0:
            self.H_est = self.H_est / np.cbrt(det)
        self.status = "UPDATE"

    def get_projected_points(self, desired_pixels):
        """Liefert Pixel-Koordinaten zurück"""
        if not self.initialized:
            return desired_pixels
            
        # 1. Soll-Pixel -> Normalisiert
        p_des_norm = self.pixel2norm(desired_pixels)
        
        # 2. H anwenden
        p_curr_est_norm = self.H_est @ p_des_norm
        
        # 3. Zurück in Pixel: p_pixel = K * p_norm
        p_curr_pixel_hom = self.K @ p_curr_est_norm
        
        # 4. De-Homogenisieren
        u = p_curr_pixel_hom[0, :] / p_curr_pixel_hom[2, :]
        v = p_curr_pixel_hom[1, :] / p_curr_pixel_hom[2, :]
        
        return np.vstack((u, v))
    
class HomographyEKF(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.debug_pints = False

        self.K_inv = np.linalg.inv(K)

        self.x = np.eye(3).reshape(9)
        self.P = np.eye(9) * 0.01

    def reset(self):
        """Setzt den Filter auf den Ursprungszustand zurück."""
        super().reset()
        self.x = np.eye(3).reshape(9)
        self.P = np.eye(9) * 0.01
        print("Filter: Zurückgesetzt!")

    def set_Q_R_gate(self, q_scalar, r_scalar, gate_threshold):
        self.Q = np.eye(9) * q_scalar
        self.R = np.eye(9) * r_scalar
        self.gate_threshold = gate_threshold
        print(f"EKF Parameter geupdatet: Q={q_scalar}, R={r_scalar}")

    def pixel2norm(self, uv):
        """Hilfsfunktion: Pixel (2xN) -> Normalisiert (3xN homogen)"""
        num = uv.shape[1]
        uv_hom = np.vstack((uv, np.ones((1, num))))
        return self.K_inv @ uv_hom

    def initialize(self, current_pixels, desired_pixels):
        """Initialisiert H basierend auf den ersten sauberen Punkten"""
        # Wir berechnen H im normalisierten Raum!
        p_curr = self.pixel2norm(current_pixels) # 3xN
        p_des = self.pixel2norm(desired_pixels)  # 3xN
        
        # cv2 braucht Nx2 (x,y)
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        # findHomography: src = H * dst
        # Also p_curr = H * p_des
        H_meas, _ = cv2.findHomography(dst, src)
        
        if H_meas is not None:
            self.H_est = H_meas
            self.initialized = True
            print("Filter: Initialisiert (Normalized Space)!")

    def predict(self, vw_cam, d_star, dt):
        if not self.initialized: return

        v_cam = vw_cam[:3]
        w_cam = vw_cam[3:]

        H_curr = unflatten_H(self.x)
        n_star = np.array([0, 0, 1]) 
        d_safe = d_star #max(0.1, d_star)
        
        M = skew(w_cam) + (1.0/d_safe) * np.outer(v_cam, n_star)
        H_pred = H_curr + (M @ H_curr) * dt
        
        # Safety
        if np.abs(H_pred).max() > 1e3: H_pred = np.eye(3)
        H_pred /= (np.linalg.det(H_pred) + 1e-9)**(1/3)
        
        self.x = flatten_H(H_pred)
        self.P += self.Q
        self.status = "PREDICT"
        self.H_est = H_pred

    def update(self, current_pixels, desired_pixels):

        if not self.initialized:
            self.initialize(current_pixels, desired_pixels)
            return #unflatten_H(self.x)
        
        if current_pixels is None or desired_pixels is None:
            self.status = "MISSING"
            return #unflatten_H(self.x)

        # 1. Messung im normalisierten Raum
        p_curr = self.pixel2norm(current_pixels)
        p_des = self.pixel2norm(desired_pixels)
        
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        H_meas, _ = cv2.findHomography(dst, src)

        if H_meas is None:
            self.status = "MISSING"
            return #unflatten_H(self.x)
        
        z = flatten_H(H_meas)
        y = z - self.x
        S = self.P + self.R
        
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis = y.T @ S_inv @ y
            
            # GATING CHECK
            if mahalanobis > self.gate_threshold:
                self.status = "REJECT (GLARE)"
                # Wir ignorieren die Messung (kein Update von x und P)
                return #unflatten_H(self.x)
            
        except np.linalg.LinAlgError:
            print("Singuläre Matrix im Filter")

        K = self.P @ S_inv
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K) @ self.P

        H_upd = unflatten_H(self.x)
        H_upd /= (np.linalg.det(H_upd) + 1e-9)**(1/3)
        self.x = flatten_H(H_upd)
        self.status = "UPDATE"
        self.H_est = H_upd
        return #H_upd

    def get_projected_points(self, desired_pixels):
        """Liefert Pixel-Koordinaten zurück"""
        if not self.initialized:
            return desired_pixels
            
        # 1. Soll-Pixel -> Normalisiert
        p_des_norm = self.pixel2norm(desired_pixels)
        
        # 2. H anwenden
        if self.debug_pints:
            print("[HomographyEKF][get_projected_points] H:\n", self.H_est)
            print("[HomographyEKF][get_projected_points] p_des_norm:\n", p_des_norm)
        p_curr_est_norm = self.H_est @ p_des_norm
        
        # 3. Zurück in Pixel: p_pixel = K * p_norm
        p_curr_pixel_hom = self.K @ p_curr_est_norm
        
        # 4. De-Homogenisieren
        u = p_curr_pixel_hom[0, :] / p_curr_pixel_hom[2, :]
        v = p_curr_pixel_hom[1, :] / p_curr_pixel_hom[2, :]
        
        return np.vstack((u, v))


# Für später: Hier kommt z.B. der OneEuroFilter(BaseFilter) hin

AVAILABLE_FILTERS = {
    "- (Kein Filter)": None,
    "Extended Kalman Filter": HomographyEKF,
    "Homography Model (SL3)": HomographyFilter
}