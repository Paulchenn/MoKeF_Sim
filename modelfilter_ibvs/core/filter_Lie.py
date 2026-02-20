import numpy as np
import cv2

class HomographyFilter:
    def __init__(self, K):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        
        # H speichert die Beziehung im NORMALISIERTEN Raum (Metrisch)
        # p_curr_norm = H * p_des_norm
        self.H_est = np.eye(3) 
        self.initialized = False
        
        # Parameter für das Modell (Ebene)
        # Wir nehmen an, die Ebene ist parallel zur Kamera im Soll-Zustand
        self.n_star = np.array([0, 0, 1]) 
        # self.d_star = 0.5 # Geschätzte Distanz zur Ebene (in Metern)
        
        # Alpha: Wie stark vertrauen wir der Messung?
        # 0.1 = 10% Messung, 90% Modell (Glättung)
        self.alpha = 0.9

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

    def update(self, current_pixels, desired_pixels):
        """Fusion mit Messung"""
        if not self.initialized:
            self.initialize(current_pixels, desired_pixels)
            return

        # 1. Messung im normalisierten Raum
        p_curr = self.pixel2norm(current_pixels)
        p_des = self.pixel2norm(desired_pixels)
        
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        H_meas, _ = cv2.findHomography(dst, src)
        
        if H_meas is None:
            return 
            
        # 2. Komplementärfilter auf Lie-Gruppe (vereinfacht linear)
        self.H_est = self.alpha * H_meas + (1.0 - self.alpha) * self.H_est
        
        # Normalisieren
        det = np.linalg.det(self.H_est)
        if det > 0:
            self.H_est = self.H_est / np.cbrt(det)

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