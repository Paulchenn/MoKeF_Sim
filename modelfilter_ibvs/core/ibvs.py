import numpy as np

class IBVSController:
    def __init__(self, intrinsic_matrix):
        """
        Args:
            intrinsic_matrix (3x3 np.array): Kameramatrix K [[f, 0, cu], [0, f, cv], [0, 0, 1]]
        """
        self.K = intrinsic_matrix
        self.K_inv = np.linalg.inv(intrinsic_matrix)
        self.lambda_gain = 1.0 # Verstärkungsfaktor (Geschwindigkeit)

    def pixel2normalized(self, uv):
        """Wandelt Pixel (u,v) in normalisierte Bildkoordinaten (x,y) um."""
        # uv ist 2xN
        num_points = uv.shape[1]
        # Homogen machen (3xN)
        uv_hom = np.vstack((uv, np.ones((1, num_points))))
        # Mit K_inv multiplizieren
        xy_hom = self.K_inv @ uv_hom
        # Zurückgeben als 2xN (die 3. Zeile ist 1)
        return xy_hom[:2, :]

    def get_interaction_matrix_point(self, x, y, Z):
        """
        Berechnet die 2x6 Interaktionsmatrix Ls für EINEN Punkt.
        x, y: Normalisierte Koordinaten (nicht Pixel!)
        Z: Tiefe des Punktes (in Meter)
        """
        L = np.array([
            [-1/Z,  0,    x/Z,  x*y,      -(1+x**2), y],
            [ 0,   -1/Z,  y/Z,  1+y**2,   -x*y,      -x]
        ])
        return L

    def compute_velocity(self, current_pixels, desired_pixels, Z_est=0.5):
        """
        Berechnet die notwendige Kamerageschwindigkeit (Twist).
        
        Args:
            current_pixels: 2xN Array (Aktuelle Features)
            desired_pixels: 2xN Array (Soll Features)
            Z_est: Geschätzte Tiefe (hier vereinfacht konstant oder vom Sim)
            
        Returns:
            v_cam: 6x1 Vektor [vx, vy, vz, wx, wy, wz]
            error_norm: Skalarer Fehler (zum Plotten)
        """
        # 1. Pixel in normalisierte Koordinaten wandeln
        cur_n = self.pixel2normalized(current_pixels)
        des_n = self.pixel2normalized(desired_pixels)
        
        # 2. Fehler berechnen (e = s - s*)
        error = cur_n - des_n
        # Flatten error für Matrix-Vektor Multiplikation (2N x 1)
        error_vec = error.T.flatten()
        
        # 3. Interaktionsmatrix stapeln
        num_points = current_pixels.shape[1]
        Ls_list = []
        
        for i in range(num_points):
            x = cur_n[0, i]
            y = cur_n[1, i]
            # Ls für diesen Punkt berechnen
            L_p = self.get_interaction_matrix_point(x, y, Z_est)
            Ls_list.append(L_p)
            
        # Gestapelte Matrix (2N x 6)
        Ls = np.vstack(Ls_list)
        
        # # 4. Pseudoinverse berechnen (Moore-Penrose)
        # # Wir wollen v lösen für: Ls * v = -lambda * error
        # try:
        #     Ls_inv = np.linalg.pinv(Ls)
        # except np.linalg.LinAlgError:
        #     print("Singularität in Interaktionsmatrix!")
        #     return np.zeros(6), 0.0
        
        # 4. Damped Least Squares (DLS) berechnen (anstelle reiner Pseudoinverse)
        # Wir wollen v lösen für: Ls * v = -lambda * error
        # Ls ist (2N x 6). DLS für überbestimmte Systeme: (Ls^T Ls + lambda^2 I)^-1 Ls^T
        damp = 0.05 # Dämpfungsfaktor
        try:
            I6 = np.eye(6)
            Ls_inv = np.linalg.inv(Ls.T @ Ls + (damp**2) * I6) @ Ls.T
        except np.linalg.LinAlgError:
            print("Singularität in Interaktionsmatrix (DLS fehlgeschlagen)!")
            return np.zeros(6), 0.0

        # 5. Regelgesetz: v_c = -lambda * Ls_inv * error
        v_cam = -self.lambda_gain * (Ls_inv @ error_vec)
        
        return v_cam, np.linalg.norm(error_vec)