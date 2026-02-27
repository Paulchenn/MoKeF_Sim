# modelfilter_ibvs/core/filters/base.py
import numpy as np
import cv2

class BaseFilter:
    """
    Zentrale Basisklasse für alle Image-Space Filter.
    Übernimmt das RANSAC-Matching, die dynamische Bounding-Box und den Geometrie-Check.
    """
    def __init__(self, K):
        self.K = K
        self.initialized = False
        self.status = "INIT"
        
        # Wird dynamisch bei der Initialisierung gesetzt
        self.proxy_ref = None 
        self.H_filtered = np.eye(3)

    def reset(self):
        """Setzt den internen Zustand des Filters zurück."""
        self.initialized = False
        self.status = "INIT"
        self.proxy_ref = None
        self.H_filtered = np.eye(3)
        
    def force_relocalization(self):
        """
        Zwingt den Filter dazu, seinen Zustand im nächsten Update-Schritt 
        anhand der aktuellen Messung komplett neu zu initialisieren (Tracking Recovery).
        Wichtig für die Erholung nach Teleportation oder langem Tracking-Verlust.
        """
        self.initialized = False
        self.status = "RELOCALIZING"
        
    def _init_dynamic_proxies(self, desired_pixels):
        """Erstellt eine eng anliegende Bounding-Box um das Referenz-Objekt."""
        u_min, u_max = np.min(desired_pixels[0, :]), np.max(desired_pixels[0, :])
        v_min, v_max = np.min(desired_pixels[1, :]), np.max(desired_pixels[1, :])
        
        # Etwas Padding (z.B. 10 Pixel) hinzufügen, damit die Punkte nicht exakt auf der Kante liegen
        pad = 10.0
        self.proxy_ref = np.array([
            [u_min - pad, v_min - pad],
            [u_max + pad, v_min - pad],
            [u_max + pad, v_max + pad],
            [u_min - pad, v_max + pad]
        ], dtype=np.float32)
        
    def _get_raw_measurement(self, current_pixels, desired_pixels):
        """
        Führt RANSAC aus und projiziert die proxy_ref Punkte ins aktuelle Bild.
        Gibt z_k (8x1) und H_raw zurück. Gibt (None, None) bei Fehler zurück.
        """
        if current_pixels is None or current_pixels.shape[1] < 4:
            return None, None, "MISSING"

        src_pts = desired_pixels.T.astype(np.float32)
        dst_pts = current_pixels.T.astype(np.float32)
        H_raw, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H_raw is None:
            return None, None, "REJECT (RANSAC FAIL)"

        # Wenn dies der allererste Durchlauf ist, Proxy-Punkte initialisieren!
        if self.proxy_ref is None:
            self._init_dynamic_proxies(desired_pixels)

        # proxy_ref mit H_raw projizieren
        proxy_hom = np.hstack((self.proxy_ref, np.ones((4, 1))))
        z_raw_hom = (H_raw @ proxy_hom.T).T
        
        z_k = np.zeros((8, 1))
        for i in range(4):
            z_k[i*2, 0] = z_raw_hom[i, 0] / z_raw_hom[i, 2]
            z_k[i*2 + 1, 0] = z_raw_hom[i, 1] / z_raw_hom[i, 2]
            
        return z_k, H_raw, self.status

    def _check_geometry_and_update_H(self, state_8d):
        """
        Prüft auf Topology Flip. Wenn konvex, wird H_filtered aktualisiert.
        Rückgabe: True wenn Geometrie OK, sonst False.
        """
        proxy_filtered = state_8d.reshape(4, 2).astype(np.float32)
        
        def cross_z(A, B, C):
            return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
        
        c1 = cross_z(proxy_filtered[0], proxy_filtered[1], proxy_filtered[2])
        c2 = cross_z(proxy_filtered[1], proxy_filtered[2], proxy_filtered[3])
        c3 = cross_z(proxy_filtered[2], proxy_filtered[3], proxy_filtered[0])
        c4 = cross_z(proxy_filtered[3], proxy_filtered[0], proxy_filtered[1])
        
        signs = [c > 0 for c in [c1, c2, c3, c4]]
        is_convex = all(signs) or not any(signs)
        
        if not is_convex:
            print("\n!!! CRITICAL GEOMETRY ERROR ABGEFANGEN !!!")
            print("Das Proxy-Polygon hat sich zusammengefaltet (Topology Flip).")
            print("Koordinaten der 4 Ecken:")
            print(proxy_filtered)
            print("---------------------------------\n")
            return False
            
        self.H_filtered = cv2.getPerspectiveTransform(self.proxy_ref, proxy_filtered)
        return True

    def predict(self, v_ee, Z_est, dt):
        """
        Prädiktionsschritt (Modell-Update).
        Wird in JEDEM Frame aufgerufen, nutzt die Roboter-Kinematik.
        
        Args:
            v_ee (np.array): 6x1 Twist des Endeffektors (vx, vy, vz, wx, wy, wz)
            Z_est (float): Geschätzte mittlere Tiefe
            dt (float): Zeitdifferenz zum letzten Frame in Sekunden
        """
        raise NotImplementedError("Muss implementiert werden.")

    def update(self, current_pixels, desired_pixels):
        """
        Korrekturschritt (Mess-Update).
        Bekommt die rohen Matches von XFeat/ORB.
        
        Args:
            current_pixels (np.array): 2xN Array der aktuellen Pixel
            desired_pixels (np.array): 2xN Array der Referenz-Pixel
        """
        raise NotImplementedError("Muss implementiert werden.")
    
    def get_projected_points(self, desired_features):
        """
        Projiziert die sauberen Soll-Features mittels der gefilterten Transformation.
        
        Args:
            desired_features (np.array): 2xM Array der idealen Zielkoordinaten
            
        Returns:
            np.array: 2xM Array der gefilterten aktuellen Pixelkoordinaten
        """
        if not self.initialized or self.H_filtered is None:
            return desired_features
        M = desired_features.shape[1]
        hom_pts = np.vstack((desired_features, np.ones((1, M))))
        proj_hom = self.H_filtered @ hom_pts
        return np.vstack((proj_hom[0, :] / proj_hom[2, :], proj_hom[1, :] / proj_hom[2, :]))