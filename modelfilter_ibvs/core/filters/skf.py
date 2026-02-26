import numpy as np
import cv2
from .base import BaseFilter

class StandardKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.K_inv = np.linalg.inv(K)
        
        # Zustand x ist die flachgedrückte Homographie-Matrix (9x1 Vektor)
        self.x = np.zeros((9, 1))
        # Kovarianzmatrix P (9x9)
        self.P = np.eye(9)
        
        # Standard-Parameter
        self.Q_scalar = 0.01
        self.R_scalar = 0.1
        self.gate_threshold = 20
        
        # Modell-Annahmen
        self.n_star = np.array([0, 0, 1])

        self.status = "INIT"
        
    def set_Q_R(self, q_val, r_val, gate_threshold):
        """Wird von der GUI aufgerufen, um Parameter live zu ändern."""
        self.Q_scalar = q_val
        self.R_scalar = r_val
        self.gate_threshold = gate_threshold
        print(f"[SKF] Parameter aktualisiert: Q={q_val}, R={r_val}, Gate={gate_threshold}")
        
    def reset(self):
        super().reset()
        self.x = np.zeros((9, 1))
        self.P = np.eye(9)
        self.status = "INIT"
        self.initialized = False
        print("SKF: Zurückgesetzt!")
        
    def pixel2norm(self, uv):
        """Wandelt Pixelkoordinaten in die normalisierte Bildebene um."""
        num = uv.shape[1]
        uv_hom = np.vstack((uv, np.ones((1, num))))
        return self.K_inv @ uv_hom

    def get_H_meas(self, current_pixels, desired_pixels):
        """Hilfsfunktion: Berechnet die rohe Homographie aus den XFeat-Matches."""
        p_curr = self.pixel2norm(current_pixels)
        p_des = self.pixel2norm(desired_pixels)
        
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        H_meas, _ = cv2.findHomography(dst, src)
        return H_meas

    def initialize(self, current_pixels, desired_pixels):
        H_meas = self.get_H_meas(current_pixels, desired_pixels)
        
        if H_meas is not None:
            # Determinante auf 1 normalisieren (SL(3) Eigenschaft)
            det = np.linalg.det(H_meas)
            if det > 0:
                H_meas = H_meas / np.cbrt(det)
                
            # 'F' steht für Fortran-Style (Spaltenweise flachdrücken).
            # Das ist wichtig, damit die Mathematik bei der Kronecker-Multiplikation unten stimmt!
            self.x = H_meas.flatten('F').reshape(9, 1)
            self.P = np.eye(9) * 1.0 # Initiale Unsicherheit
            self.initialized = True
            self.status = "INIT"
            print("[SKF] Initialisiert!")

    def predict(self, v_cam, d_star, dt):
        if not self.initialized: return
        
        v = v_cam[:3]
        w = v_cam[3:]
        
        # 1. Kinematische Matrix A berechnen
        w_x = np.array([
            [0,     -w[2],  w[1]],
            [w[2],   0,    -w[0]],
            [-w[1],  w[0],  0   ]
        ])
        
        n_vec = self.n_star.reshape(3, 1)
        v_vec = v.reshape(3, 1)
        T_v = (1.0 / d_star) * (v_vec @ n_vec.T)
        
        A = w_x + T_v
        
        # 2. Zustandsübergangsmatrix F (für 3x3 Matrix)
        # Euler-Integration: H_neu = H_alt - A * H_alt * dt = (I - A*dt) * H_alt
        F_3 = np.eye(3) - A * dt
        
        # 3. Zustandsübergangsmatrix F_9 (für den 9x1 Vektor x)
        # Mit dem Kronecker-Produkt blasen wir die 3x3 Logik auf unseren 9x1 Vektor auf
        F_9 = np.kron(np.eye(3), F_3)
        
        # 4. Prozessrauschen Q
        Q = np.eye(9) * self.Q_scalar
        
        # --- KALMAN PREDICT SCHRITT ---
        self.x = F_9 @ self.x
        self.P = F_9 @ self.P @ F_9.T + Q
        self.status = "PREDICT"

    def update(self, current_pixels, desired_pixels):
        if not self.initialized:
            self.initialize(current_pixels, desired_pixels)
            return

        if current_pixels is None or desired_pixels is None:
            self.status = "MISSING"
            return
        
        # 1. Messung einholen
        H_meas = self.get_H_meas(current_pixels, desired_pixels)
        if H_meas is None:
            self.status = "MISSING"
            return # Blindflug fortsetzen
        
        det = np.linalg.det(H_meas)
        if det > 0:
            H_meas = H_meas / np.cbrt(det)
            
        z = H_meas.flatten('F').reshape(9, 1)
        
        # Messrauschen R
        R = np.eye(9) * self.R_scalar
        
        # --- KALMAN UPDATE SCHRITT ---
        # S = P + R (Da Messmatrix C die Einheitsmatrix ist, fällt sie in den Formeln weg)
        S = self.P + R

        y = z - self.x

        try:
            S_inv = np.linalg.inv(S)
            mahalanobis = y.T @ S_inv @ y
            
            # GATING CHECK
            if mahalanobis > self.gate_threshold:
                self.status = "REJECT (GLARE)"
                # Wir ignorieren die Messung (kein Update von x und P)
                return
        except np.linalg.LinAlgError:
            print("[EKF] Singularität in S-Matrix erkannt! Überspringe Update.")
            return
        
        # Kalman Gain K = P * S^-1
        K = self.P @ np.linalg.inv(S)
        
        # Zustand korrigieren: x = x + K * (z - x)
        self.x = self.x + K @ y
        
        # Kovarianz updaten: P = (I - K) * P
        self.P = (np.eye(9) - K) @ self.P
        
        # SL(3) Normalisierung nach dem Update, um Verzerrungen zu vermeiden
        H_temp = self.x.reshape(3, 3, order='F')
        det = np.linalg.det(H_temp)
        if det > 0:
            self.x = (H_temp / np.cbrt(det)).flatten('F').reshape(9, 1)
        self.status = "UPDATE"

    def get_projected_points(self, desired_pixels):
        if not self.initialized: return desired_pixels
        
        # Matrix aus Vektor wiederherstellen
        H_est = self.x.reshape(3, 3, order='F')
        
        # Vorwärtsprojektion
        p_des_norm = self.pixel2norm(desired_pixels)
        p_curr_est_norm = H_est @ p_des_norm
        p_curr_pixel_hom = self.K @ p_curr_est_norm
        
        u = p_curr_pixel_hom[0, :] / p_curr_pixel_hom[2, :]
        v = p_curr_pixel_hom[1, :] / p_curr_pixel_hom[2, :]
        
        return np.vstack((u, v))