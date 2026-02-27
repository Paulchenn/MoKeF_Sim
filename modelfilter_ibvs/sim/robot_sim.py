import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import cv2

class RobotSimulator:
    def __init__(self):
        self.debug_pints = False

        # Wir laden einen Standard 6-Achs-Roboter (UR5)
        self.robot = rtb.models.DH.UR5()
        if self.debug_pints:
            print("[RobotSimulator][__init__] Robot loaded: ", self.robot)
        
        # Startkonfiguration (q = Gelenkwinkel)
        # qn ist eine "schöne" Standardpose, damit er nicht zusammengeklappt startet
        self.robot.myq_start = np.array([np.deg2rad(165), -np.pi/2, np.pi/2, 0, np.pi/2, 0]) # self.robot.qz
        self.q = self.robot.myq_start.copy() # Aktuelle Gelenkwinkel
        if self.debug_pints:
            print("[RobotSimulator][__init__] Initial joint configuration: ", self.q)
        
        # Definition des Objekts im Raum (World Frame)
        # 4 Punkte einer Ebene (z.B. ein Blech)
        # Koordinaten: [X, Y, Z]
        # Das Objekt steht ca. 0.6m vor dem Roboter
        self.world_points = np.array([
            [2, -0.15, 0.2], # Punkt 1 (unten links)
            [2,  0.15, 0.2], # Punkt 2 (unten rechts)
            [2,  0.15, 0.5], # Punkt 3 (oben rechts)
            [2, -0.15, 0.5]  # Punkt 4 (oben links)
        ]).T # Transponieren zu 3x4 Matrix für einfachere Multiplikation

        # Kamera-Offset: Die Kamera sitzt nicht IM Handgelenk, sondern etwas davor
        # Aktuell: keine Verschiebung und Rotation um 180° um Z, damit sie nach vorne schaut
        self.T_ee_cam = SE3(0, 0, 0) * SE3.Rz(np.pi) # Rz dreht Kamera richtig
        
        # Intrinsische Kameramatrix (vereinfacht, f = Brennweite)
        # Wir nehmen an, das Bild ist normalisiert auf +/- 1 oder wir nutzen f=800 px
        self.f = 800.0
        self.principal_point = np.array([320, 240]) # Bildmitte (bei 640x480)

    def update_joints(self, q_new):
        """Setzt die Gelenkwinkel neu"""
        self.q = np.array(q_new)

    def get_camera_view(self):
        """
        Berechnet die Projektion der Welt-Punkte auf den Kamerasensor.
        Rückgabe: 2x4 Array (u, v Koordinaten der 4 Punkte)
        """
        # 1. Vorwärtskinematik: Wo ist der Endeffektor? (T_base_ee)
        T_base_ee = self.robot.fkine(self.q)
        
        # 2. Wo ist die Kamera? (T_base_cam = T_base_ee * T_ee_cam)
        T_base_cam = T_base_ee * self.T_ee_cam
        if T_base_ee != T_base_cam and self.debug_pints:
            print("[RobotSimulator][get_camera_view] T_base_ee:\n", T_base_ee)
            print("[RobotSimulator][get_camera_view] T_base_cam:\n", T_base_cam)
        elif self.debug_pints:
            print("[RobotSimulator][get_camera_view] T_base_ee == T_base_cam (no offset)")
        
        # 3. Wir brauchen die Punkte im KAMERA-Koordinatensystem
        # T_cam_world = Inverse von T_base_cam
        T_cam_base = T_base_cam.inv()
        
        # Homogene Transformation der Weltpunkte in den Kamera-Frame
        # P_cam = T_cam_base * P_world
        # Wir müssen world_points erst homogen machen (1 anhängen)
        num_points = self.world_points.shape[1]
        P_world_hom = np.vstack((self.world_points, np.ones((1, num_points))))
        
        P_cam_hom = T_cam_base.A @ P_world_hom # .A gibt uns die Numpy Matrix
        
        # 4. Pinhole Projektion
        # P_cam_hom ist [X, Y, Z, 1]
        X = P_cam_hom[0, :]
        Y = P_cam_hom[1, :]
        Z = P_cam_hom[2, :]
        
        # Wir erstellen eine Maske für alle Punkte, die strikt VOR der Kamera liegen.
        # Wir nehmen 1e-3 (1 mm) als Mindestabstand zur Linse, um Division durch 0 zu vermeiden.
        valid_mask = Z > 1e-3 
        
        # Bereite leere Arrays voller NaNs vor, die genauso groß sind wie X
        u = np.full_like(X, np.nan)
        v = np.full_like(Y, np.nan)
        
        # Berechne die Projektion NUR für die gültigen Punkte (valid_mask)
        u[valid_mask] = self.f * (X[valid_mask] / Z[valid_mask]) + self.principal_point[0]
        v[valid_mask] = self.f * (Y[valid_mask] / Z[valid_mask]) + self.principal_point[1]
        
        return np.vstack((u, v))

    def get_camera_image(self, texture_img=None, noise_std=0.0):
        """
        Rendert ein 2D-Bild der Szene. 
        Projiziert eine Textur auf das Viereck der Kameraansicht.
        """
        # Weißes/Graues Hintergrundbild (640x480)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200 
        
        # Hole die aktuellen UV-Koordinaten des Objekts
        uv = self.get_camera_view()
        dst_pts = uv.T.astype(np.float32) # 4x2 Array für OpenCV
        
        # Wenn wir keine Textur übergeben, machen wir ein einfaches Schachbrett
        if texture_img is None:
            texture_img = np.zeros((200, 200, 3), dtype=np.uint8)
            texture_img[0:100, 0:100] = 255
            texture_img[100:200, 100:200] = 255
            texture_img[0:100, 100:200] = 100
            texture_img[100:200, 0:100] = 100
            
        h, w = texture_img.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        # Berechne die Transformation von der flachen Textur ins Kamerabild
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Zeichne die Textur in das Kamerabild
        cv2.warpPerspective(texture_img, M, (640, 480), img, borderMode=cv2.BORDER_TRANSPARENT)

        # NEU: Bildrauschen auf die Pixel anwenden
        if noise_std > 0:
            # Rauschen generieren (shape wie img)
            noise = np.random.normal(0, noise_std * 5, img.shape).astype(np.float32) # *5 damit der Slider mehr Wumms hat
            img_noisy = img.astype(np.float32) + noise
            # Werte wieder auf 0-255 clippen
            img = np.clip(img_noisy, 0, 255).astype(np.uint8)
            
        return img

    def get_pose(self):
        """Gibt die aktuelle Pose für den Plotter zurück"""
        return self.robot.fkine(self.q)
    
    def get_jacobian(self):
        """Gibt die Jacobi-Matrix im Endeffektor-Frame zurück (6xN)."""
        # jacobe berechnet Jacobian im End-Effector Frame
        return self.robot.jacobe(self.q)
    
    def get_intrinsic_matrix(self):
        """Gibt K zurück"""
        return np.array([
            [self.f, 0,      self.principal_point[0]],
            [0,      self.f, self.principal_point[1]],
            [0,      0,      1                      ]
        ])
        
    def get_mean_depth(self):
        """
        Cheat-Funktion für die Simulation: 
        Gibt die wahre mittlere Tiefe der Punkte zur Kamera zurück.
        (Im echten System bräuchte man hier einen Schätzer oder konstanten Wert)
        """
        # Pose der Kamera
        T_base_cam = self.robot.fkine(self.q) * self.T_ee_cam
        T_cam_base = T_base_cam.inv()
        
        # Punkte in Kamera-Frame transformieren
        P_world_hom = np.vstack((self.world_points, np.ones((1, 4))))
        P_cam = T_cam_base.A @ P_world_hom
        
        # Z-Werte sind in der 3. Zeile (Index 2)
        return np.mean(P_cam[2, :])
    
    def get_noisy_camera_view(self, noise_level=0.0):
        """
        Gibt verrauschte Punkte zurück.
        noise_level: Standardabweichung in Pixeln
        """
        clean_uv = self.get_camera_view()
        noise = np.random.normal(0, noise_level, clean_uv.shape)
        return clean_uv + noise