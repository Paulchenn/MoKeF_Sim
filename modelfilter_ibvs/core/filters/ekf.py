import numpy as np
import cv2
from .base import BaseFilter

class ExtendedKalmanFilter(BaseFilter):
    def __init__(self, K):
        super().__init__(K)
        self.K_inv = np.linalg.inv(K)
        
        # Zustand x ist die flachgedrückte Homographie-Matrix (9x1 Vektor)
        # Ordnung 'F' (Fortran/Column-major): [h11, h21, h31, h12, h22, h32, h13, h23, h33]^T
        self.x = np.zeros((9, 1))
        self.P = np.eye(9) * 0.01
        
        # Standard-Parameter
        self.Q_scalar = 0.01
        self.R_scalar = 0.005 # Messrauschen in normalisierten Koordinaten (sehr klein!)
        self.gate_threshold = 20
        
        # Modell-Annahmen für Prediction
        self.n_star = np.array([0, 0, 1])
        
    def set_Q_R(self, q_val, r_val, gate_threshold):
        self.Q_scalar = q_val
        self.R_scalar = r_val
        self.gate_threshold = gate_threshold
        print(f"[EKF] Parameter aktualisiert: Q={q_val}, R={r_val}, Gate={gate_threshold}")
        
    def reset(self):
        super().reset()
        self.x = np.zeros((9, 1))
        self.P = np.eye(9) * 0.01
        self.status = "INIT"
        self.initialized = False
        
    def pixel2norm(self, uv):
        num = uv.shape[1]
        uv_hom = np.vstack((uv, np.ones((1, num))))
        return self.K_inv @ uv_hom

    def initialize(self, current_pixels, desired_pixels):
        # Für die allererste Schätzung nutzen wir OpenCV (DLT Algorithmus)
        p_curr = self.pixel2norm(current_pixels)
        p_des = self.pixel2norm(desired_pixels)
        
        src = p_curr[:2, :].T
        dst = p_des[:2, :].T
        
        H_meas, _ = cv2.findHomography(dst, src)
        
        if H_meas is not None:
            det = np.linalg.det(H_meas)
            if det > 0:
                H_meas = H_meas / np.cbrt(det)
                
            self.x = H_meas.flatten('F').reshape(9, 1)
            self.P = np.eye(9) * 0.01
            self.initialized = True
            print("[EKF] Initialisiert!")

    def predict(self, v_cam, d_star, dt):
        """Lineare Prädiktion (Motion Model) - Identisch zum SKF"""
        if not self.initialized: return
        
        v = v_cam[:3]; w = v_cam[3:]
        w_x = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        
        n_vec = self.n_star.reshape(3, 1)
        v_vec = v.reshape(3, 1)
        T_v = (1.0 / d_star) * (v_vec @ n_vec.T)
        
        A = w_x + T_v
        
        F_3 = np.eye(3) - A * dt
        F_9 = np.kron(np.eye(3), F_3)
        
        Q = np.eye(9) * self.Q_scalar
        
        self.x = F_9 @ self.x
        self.P = F_9 @ self.P @ F_9.T + Q
        self.status = "PREDICT"

    def update(self, current_pixels, desired_pixels):
        """NICHT-lineares Update mit den rohen Keypoint-Koordinaten"""
        if not self.initialized: 
            self.initialize(current_pixels, desired_pixels)
            return
        
        if current_pixels is None or desired_pixels is None:
            self.status = "MISSING"
            return
        
        num_points = current_pixels.shape[1]
        if num_points < 4: return # Zu wenig Punkte für sinnvolles Update
        
        # 1. Alles in normalisierte Koordinaten wandeln
        p_curr = self.pixel2norm(current_pixels)
        p_des = self.pixel2norm(desired_pixels)
        
        # 2. Reale Messung z formattieren: [u1, v1, u2, v2, ...]^T
        z = p_curr[:2, :].T.flatten().reshape(-1, 1)
        
        # 3. Vorhersage der Messung (z_hat) und Berechnung der Jacobi-Matrix (C)
        H_est = self.x.reshape(3, 3, order='F')
        
        # Projiziere die Soll-Punkte mit unserem aktuellen Zustand H
        p_proj = H_est @ p_des
        
        C_list = []
        z_hat_list = []
        
        for i in range(num_points):
            x_i = p_des[0, i]
            y_i = p_des[1, i]
            
            # Nenner der perspektivischen Division
            D = p_proj[2, i] 
            
            # Verhindere Division durch Null
            if abs(D) < 1e-6: D = 1e-6 
            
            u_hat = p_proj[0, i] / D
            v_hat = p_proj[1, i] / D
            z_hat_list.extend([u_hat, v_hat])
            
            # --- Analytische Jacobi-Matrix (Partielle Ableitungen) ---
            # Ableitungen nach x = [h11, h21, h31, h12, h22, h32, h13, h23, h33]
            du_dx = np.array([
                x_i/D, 0, -u_hat*x_i/D, 
                y_i/D, 0, -u_hat*y_i/D, 
                1/D,   0, -u_hat/D
            ])
            
            dv_dx = np.array([
                0, x_i/D, -v_hat*x_i/D, 
                0, y_i/D, -v_hat*y_i/D, 
                0, 1/D,   -v_hat/D
            ])
            
            C_list.append(du_dx)
            C_list.append(dv_dx)
            
        z_hat = np.array(z_hat_list).reshape(-1, 1)
        C = np.vstack(C_list) # Matrix der Größe (2N x 9)
        
        # 4. Kalman Filter Mathematik
        # Innovation (Fehler zwischen Messung und Prädiktion)
        y = z - z_hat
        
        # Messrauschen Matrix R (2N x 2N)
        R = np.eye(2 * num_points) * self.R_scalar
        
        # Innovationskovarianz S
        S = C @ self.P @ C.T + R
        
        # Kalman Gain K = P * C^T * S^-1
        # (Bei sehr vielen Keypoints kann die Inversion von S teuer werden. 
        # Da wir auf 50 Matches limitiert haben, ist max_size 100x100 -> Numpy macht das in Millisekunden)
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
            
        K_gain = self.P @ C.T @ S_inv
        
        # 5. Zustands- und Kovarianz-Update
        self.x = self.x + K_gain @ y
        
        # P = (I - K*C) * P
        I = np.eye(9)
        self.P = (I - K_gain @ C) @ self.P
        
        # 6. SL(3) Normalisierung (H_33 muss Determinante 1 behalten)
        H_temp = self.x.reshape(3, 3, order='F')
        det = np.linalg.det(H_temp)
        if det > 0:
            self.x = (H_temp / np.cbrt(det)).flatten('F').reshape(9, 1)
        self.status = "UPDATE"

    def get_projected_points(self, desired_pixels):
        if not self.initialized: return desired_pixels
        
        H_est = self.x.reshape(3, 3, order='F')
        
        p_des_norm = self.pixel2norm(desired_pixels)
        p_curr_est_norm = H_est @ p_des_norm
        p_curr_pixel_hom = self.K @ p_curr_est_norm
        
        u = p_curr_pixel_hom[0, :] / p_curr_pixel_hom[2, :]
        v = p_curr_pixel_hom[1, :] / p_curr_pixel_hom[2, :]
        
        return np.vstack((u, v))