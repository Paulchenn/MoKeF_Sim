####################################################################################################
# GUI für die IBVS Simulation
#
# Autor: Paul Glaser
#
# Version 0.1 - Stand 19.02.2026
# 
# Ziel: Ein einfacher Simulator mit einem 6-Achs-Roboter, einer Kamera und einem IBVS-Controller. Inklusive eines EKF-Filters für die Homographie-Schätzung.
#
# Wichtig: Um Pfadfehler zu vermeiden, sollte dieses Skript aus dem Hauptverzeichnis des Projekts gestartet werden (dort, wo auch "sim" und "core" liegen), mit:
# python -m modelfilter_ibvs.sim.gui_main
####################################################################################################
VERSION = "0.2"

import sys, os, cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QComboBox, 
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Imports (Pfade anpassen wie gehabt)
try:
    from modelfilter_ibvs.sim.robot_sim import RobotSimulator
    from modelfilter_ibvs.core.ibvs import IBVSController
    from modelfilter_ibvs.core.filter_2 import HomographyEKF
    # from modelfilter_ibvs.core.feature_extractor import FeatureExtractor
    from modelfilter_ibvs.core.feature_extractor import AVAILABLE_EXTRACTORS
except ImportError:
    path_file = os.path.abspath(__file__)
    path_parent = os.path.dirname(path_file)
    path_grandpaent = os.path.dirname(path_parent)
    sys.path.insert(0, path_parent)
    sys.path.insert(0, path_grandpaent)
    from sim.robot_sim import RobotSimulator
    from core.ibvs import IBVSController
    from core.filter_2 import HomographyEKF
    # from core.feature_extractor import FeatureExtractor
    from core.feature_extractor import AVAILABLE_EXTRACTORS

class SimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.debug_prints = False
        
        self.sim = RobotSimulator()

        self.vision = None # Wird durch Dropdown initialisiert
        
        # Controller initialisieren
        K = self.sim.get_intrinsic_matrix()
        self.controller = IBVSController(K)

        # Filter initialisieren
        self.filter = HomographyEKF(K, Q_scale=0.0001, R_scale=0.005)
        self.use_filter = False
        self.simulate_glare = False # Wenn True, kommen keine Messdaten mehr
        self.noise_level = 0.0
        
        # Speicher für Soll-Features (None = noch nicht eingelernt)
        self.desired_features = None
        self.is_servoing = False
        
        # Timer für Regelungsschleife (ca. 20 Hz)
        self.timer = QTimer()
        self.timer.timeout.connect(self.control_loop)
        self.timer.start(50) 

        self.current_q_dot = np.zeros(6)
        self.debug_filtered_pts = None
        self.current_img = None
        self.current_raw_features = None
        self.current_matched_ref = None
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("IBVS Simulation - Version " + VERSION)
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Layouts
        left_layout = QVBoxLayout() # Plots
        right_layout = QVBoxLayout() # Controls
        
        # --- Matplotlib Canvas ---
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_cam = self.fig.add_subplot(122)
        left_layout.addWidget(self.canvas)
        
        # --- Controls ---
        # 1. IBVS Buttons
        ibvs_group = QGroupBox("IBVS Control")
        ibvs_layout = QVBoxLayout()
        
        self.btn_set_desired = QPushButton("1. Set Current View as Desired")
        self.btn_set_desired.clicked.connect(self.set_desired_view)
        
        self.btn_toggle_servo = QPushButton("2. Start Servoing")
        self.btn_toggle_servo.setCheckable(True)
        self.btn_toggle_servo.clicked.connect(self.toggle_servoing)
        self.btn_toggle_servo.setEnabled(False) # Erst aktiv wenn Desired gesetzt
        
        self.lbl_error = QLabel("Error: -")
        
        ibvs_layout.addWidget(self.btn_set_desired)
        ibvs_layout.addWidget(self.btn_toggle_servo)
        ibvs_layout.addWidget(self.lbl_error)
        ibvs_group.setLayout(ibvs_layout)
        right_layout.addWidget(ibvs_group)

        # --- Vision Settings ---
        vision_group = QGroupBox("Vision System")
        vision_layout = QVBoxLayout()

        self.cb_extractor = QComboBox()
        self.cb_extractor.addItems(list(AVAILABLE_EXTRACTORS.keys()))
        self.cb_extractor.currentTextChanged.connect(self.change_extractor)

        vision_layout.addWidget(QLabel("Feature Extractor:"))
        vision_layout.addWidget(self.cb_extractor)
        vision_group.setLayout(vision_layout)
        right_layout.addWidget(vision_group)

        # 2. Filter & Noise Simulation
        filter_group = QGroupBox("Filter & Noise Simulation")
        filter_layout = QVBoxLayout()
        
        self.cb_use_filter = QPushButton("Enable Filter Prediction")
        self.cb_use_filter.setCheckable(True)
        self.cb_use_filter.clicked.connect(self.toggle_filter)
        
        self.cb_glare = QPushButton("SIMULATE GLARE (Blind)")
        self.cb_glare.setCheckable(True)
        self.cb_glare.setStyleSheet("background-color : red") # Warnfarbe
        self.cb_glare.clicked.connect(self.toggle_glare)
        
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise:"))
        self.sl_noise = QSlider(Qt.Horizontal)
        self.sl_noise.setRange(0, 20) # 0 bis 20 Pixel Rauschen
        self.sl_noise.valueChanged.connect(self.set_noise)
        noise_layout.addWidget(self.sl_noise)
        
        filter_layout.addWidget(self.cb_use_filter)
        filter_layout.addLayout(noise_layout)
        filter_layout.addWidget(self.cb_glare)
        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)

        # 3. Joint Sliders
        slider_group = QGroupBox("Manual Joint Control")
        self.slider_layout = QVBoxLayout()
        self.sliders = []
        self.labels = []
        
        for i in range(6):
            h = QHBoxLayout()
            lbl = QLabel(f"J{i+1}")
            val_lbl = QLabel("0.0")
            self.labels.append(val_lbl)
            
            sl = QSlider(Qt.Horizontal)
            sl.setRange(-180, 180)
            deg = int(np.rad2deg(self.sim.q[i]))
            sl.setValue(deg)
            sl.valueChanged.connect(self.manual_slider_change)
            
            self.sliders.append(sl)
            h.addWidget(lbl)
            h.addWidget(sl)
            h.addWidget(val_lbl)
            self.slider_layout.addLayout(h)
            
        slider_group.setLayout(self.slider_layout)
        right_layout.addWidget(slider_group)
        right_layout.addStretch()

        reset_group = QGroupBox("Reset")
        reset_layout = QVBoxLayout()
        self.btn_reset = QPushButton("Reset Simulation")
        self.btn_reset.setStyleSheet("background-color : #f0ad4e") # Leicht orange zur Warnung
        self.btn_reset.clicked.connect(self.reset_simulation)
        reset_layout.addWidget(self.btn_reset)
        reset_group.setLayout(reset_layout)
        right_layout.addWidget(reset_group)
        
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)
        
        self.change_extractor(self.cb_extractor.currentText())
        self.update_plots()

    def set_desired_view(self):
        # """Speichert die aktuellen Punkte als Soll-Zustand"""
        # self.desired_features = self.sim.get_camera_view()
        # self.btn_toggle_servo.setEnabled(True)
        # print("Desired Features set!")
        # self.update_plots() # Zeichnet die grünen Soll-Punkte ein

        # Rendere das Bild
        img = self.sim.get_camera_image()

        # Speichere es als Referenz im Feature-Extractor
        self.vision.set_reference_image(img)

        # Wir speichern auch die "echten" Ecken für den Filter als Referenz
        # (Da wir hier in der Sim sind, wissen wir die echten Ecken für das IBVS)
        self.desired_features = self.sim.get_camera_view()

        self.btn_toggle_servo.setEnabled(True)
        print("Desired View set in Vision System!")
        self.update_plots()

    def toggle_servoing(self):
        self.is_servoing = self.btn_toggle_servo.isChecked()
        if self.is_servoing:
            self.btn_toggle_servo.setText("Stop Servoing")
            # Deaktiviere Slider während Auto-Mode
            for s in self.sliders: s.setEnabled(False)
        else:
            self.btn_toggle_servo.setText("Start Servoing")
            # Reaktiviere Slider
            for s in self.sliders: s.setEnabled(True)
            # Slider Positionen updaten auf aktuelle Roboter-Pose
            self.update_sliders_from_robot()

    def manual_slider_change(self):
        if self.is_servoing: return # Ignorieren wenn Servo läuft
        
        q_new = []
        for i, s in enumerate(self.sliders):
            rad = np.deg2rad(s.value())
            q_new.append(rad)
            self.labels[i].setText(f"{s.value()}°")
            
        self.sim.update_joints(q_new)
        self.update_plots()

    def update_sliders_from_robot(self):
        """Synchronisiert Slider mit aktueller Roboterposition (nach Servoing)"""
        for i, s in enumerate(self.sliders):
            deg = int(np.rad2deg(self.sim.q[i]))
            # Blockiere Signal, damit kein Loop entsteht
            s.blockSignals(True)
            s.setValue(deg)
            self.labels[i].setText(f"{deg}°")
            s.blockSignals(False)

    def change_extractor(self, extractor_name):
        """Wird aufgerufen, wenn ein anderer Extractor im Dropdown gewählt wird."""
        print(f"Wechsle Feature Extractor zu: {extractor_name}")

        # Instanziiere die gewählte Klasse aus dem Dictionary
        ExtractorClass = AVAILABLE_EXTRACTORS[extractor_name]
        self.vision = ExtractorClass()

        # Da der neue Extractor keine Soll-Bilder kennt, machen wir einen Reset
        # (Um den Aufruf sauber zu halten, checken wir, ob wir schon initialisiert sind)
        if hasattr(self, 'is_servoing'):
            self.reset_simulation()

    def control_loop(self):
        if not self.is_servoing or self.desired_features is None:
            if self.desired_features is None: return

        # 1. Datenbeschaffung (Bilder rendern & Keypoints extrahieren)
        self.current_img = self.sim.get_camera_image()
        
        if self.simulate_glare:
            # Bei Blendung blenden wir das Bild in der GUI weiß aus
            self.current_img = np.ones_like(self.current_img) * 255
            self.current_raw_features = None 
            self.current_matched_ref = None
        else:
            # XFeat / ORB Wrapper aufrufen
            raw_curr, raw_des = self.vision.match_current_image(self.current_img)
            self.current_raw_features = raw_curr
            self.current_matched_ref = raw_des
            
        # 2. Prediction Step (Immer ausführen, wenn Roboter sich bewegt)
        # Wir brauchen die Geschwindigkeit. Wenn der Roboter steht -> 0.
        # Wir nehmen current_q_dot vom letzten Loop.
        J = self.sim.get_jacobian()
        v_ee = J @ self.current_q_dot
        
        # Kleiner Hack: In der Sim ist Kamera starr am EE, aber Koordinatensystem gedreht.
        # Korrekterweise müssten wir v_ee in v_cam transformieren.
        # Für unsere einfachen Tests (Drehen um Z) reicht v_ee ~= v_cam oft aus, 
        # ABER für die Prediction Formel ist die Achsenrichtung wichtig.
        # Wir übergeben den Twist und lassen den Filter arbeiten (dort haben wir das Minuszeichen gesetzt).
        dt = 0.05
        Z_est = self.sim.get_mean_depth()
        if self.debug_prints:
            print("[GUI_MAIN][control_loop] v_ee (Twist): ", v_ee)
            print("[GUI_MAIN][control_loop] Z_est: ", Z_est)
        self.filter.predict(v_ee, Z_est, dt) # Wir nutzen v_ee als Annäherung für v_cam

        # 3. Update Step (Fusion)
        if self.current_raw_features is not None:
            # Wir füttern den Filter mit dem Schwarm an XFeat-Punkten
            H_est = self.filter.update(self.current_raw_features, self.current_matched_ref)


        # # --- ENTSCHEIDUNGSLOGIK ---
        features_for_control = None

        # # Fall A: Wir wollen den Filter nutzen UND er ist bereit
        if self.use_filter and self.filter.initialized:
            # Regler nutzt VIRTUELL projizierte Proxy-Punkte (die 4 Ecken)
            features_for_control = self.filter.get_projected_points(H_est, self.desired_features)
            self.debug_filtered_pts = features_for_control

        # Fall B: Kein Filter gewünscht ODER Filter noch nicht bereit -> Nimm Rohdaten
        elif self.current_raw_features is not None:
            # Ohne Filter nutzen wir (zum Testen) die wahren Ecken aus der Sim, 
            # da der Regler feste Proxy-Punkte braucht. 
            features_for_control = self.sim.get_camera_view()
            self.debug_filtered_pts = None 
            
            if self.use_filter and not self.filter.initialized:
                self.filter.initialize(self.current_raw_features, self.current_matched_ref)
        
        # Fall C: Glare an, aber Filter war noch nie an -> PANIK
        else:
            print("Verloren! Blendung aktiv, aber Filter nicht initialisiert.")
            self.debug_filtered_pts = None
            return

        # 4. Regelung (Nur wenn Servoing aktiv)
        if self.is_servoing:
            v_cam_cmd, error_norm = self.controller.compute_velocity(
                features_for_control, 
                self.desired_features, 
                Z_est=Z_est
            )
            self.lbl_error.setText(f"Error: {error_norm:.4f}")
            
            # ... (Rest der Kinematik Berechnung wie zuvor) ...
            # Transformiere v_cam_cmd zu v_ee_cmd ...
            # v_ee_cmd berechnen ...
            R = self.sim.T_ee_cam.R; t = self.sim.T_ee_cam.t
            v_lin = R @ v_cam_cmd[0:3]; v_ang = R @ v_cam_cmd[3:6]
            v_lin = v_lin + np.cross(t, v_ang)
            v_ee_cmd = np.concatenate((v_lin, v_ang))

            J_pinv = np.linalg.pinv(J)
            q_dot = J_pinv @ v_ee_cmd * 0.5 # Gain etwas runter
            
            # SPEICHERN für nächsten Filter Schritt!
            self.current_q_dot = q_dot 
            
            self.sim.update_joints(self.sim.q + q_dot * dt)

            if error_norm < 0.01:
                print(f"Ziel erreicht! End-Fehler: {error_norm:.4f}")
                
                # Servoing intern beenden
                self.is_servoing = False
                
                # GUI Button visuell zurücksetzen
                self.btn_toggle_servo.setChecked(False)
                self.btn_toggle_servo.setText("2. Start Servoing")
                
                # Slider für manuelle Steuerung wieder freigeben
                for s in self.sliders: 
                    s.setEnabled(True)
                    
                # Slider-Positionen auf die aktuelle (jetzt erreichte) Roboterpose updaten
                self.update_sliders_from_robot()
        else:
            self.current_q_dot = np.zeros(6)

        self.update_plots()

    def update_plots(self):
        # 3D Plot
        self.ax_3d.clear()
        self.ax_3d.set_xlim([-0.5, 2.0]); self.ax_3d.set_ylim([-0.5, 0.5]); self.ax_3d.set_zlim([0.0, 1.0])
    
        # Roboter & Objekt zeichnen (wie vorher)
        T_all = self.sim.robot.fkine_all(self.sim.q)
        x, y, z = [], [], []
        for T in T_all:
            x.append(T.t[0]); y.append(T.t[1]); z.append(T.t[2])
        self.ax_3d.plot(x, y, z, '-o', c='k')
        
        wp = self.sim.world_points
        if self.debug_prints:
            print("[GUI_MAIN][update_plots] Weltpunkte:\n", wp)
        wp_plot = np.hstack((wp, wp[:, [0]])) # Schließe die Punkte für Plotten (ansonsten ein U)
        if self.debug_prints:
            print("[GUI_MAIN][update_plots] Weltpunkte Plot:\n", wp_plot)
        self.ax_3d.plot(wp_plot[0,:], wp_plot[1,:], wp_plot[2,:], 'r-')
        
        # --- 2D Kamera Plot ---
        self.ax_cam.clear()
        self.ax_cam.set_title("Camera View (Image + XFeat + Filter)")
        
        # 1. Zeige das Kamera-Bild
        if self.current_img is not None:
            # cv2 liefert BGR, matplotlib braucht RGB
            img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
            self.ax_cam.imshow(img_rgb)
        else:
            self.ax_cam.set_xlim([0, 640]); self.ax_cam.set_ylim([480, 0])
            self.ax_cam.invert_yaxis()
            
        # 2. Zeichne den Schwarm der rohen XFeat/ORB Keypoints (Gelb)
        if self.current_raw_features is not None:
            self.ax_cam.plot(self.current_raw_features[0,:], self.current_raw_features[1,:], 
                             'y.', markersize=4, alpha=0.7, label='XFeat Matches')
            
            # Optischer Fluss (Verbindungslinien, um Matching zu sehen)
            if self.current_matched_ref is not None:
                for i in range(min(20, self.current_raw_features.shape[1])): # max 20 Linien der Übersichtlichkeit halber
                    self.ax_cam.plot([self.current_raw_features[0,i], self.current_matched_ref[0,i]],
                                     [self.current_raw_features[1,i], self.current_matched_ref[1,i]], 'y-', alpha=0.2)

        # 3. Zeichne die 4 "Proxy-Punkte" für den IBVS Regler 
        # Ziel-Position (Rot)
        if self.desired_features is not None:
            self.ax_cam.plot(self.desired_features[0,:], self.desired_features[1,:], 
                             'ro', markersize=8, alpha=0.5, label='Target Proxy')
            
        # Gefilterte Position (Blaues Kreuz)
        if self.debug_filtered_pts is not None:
            self.ax_cam.plot(self.debug_filtered_pts[0,:], self.debug_filtered_pts[1,:], 
                             'bx', markersize=10, markeredgewidth=2, label='Filtered Proxy')
        
        # Reale Ecken (zur Kontrolle, wo sie WIRKLICH sind)
        true_corners = self.sim.get_camera_view()
        self.ax_cam.plot(true_corners[0,:], true_corners[1,:], 'g+', markersize=8, label='True Corners')

        self.ax_cam.legend(loc='upper right', fontsize='small')
        self.canvas.draw()

    def toggle_filter(self):
        self.use_filter = self.cb_use_filter.isChecked()
        if self.use_filter:
            self.cb_use_filter.setText("Filter ENABLED (Using H_est)")
            self.cb_use_filter.setStyleSheet("background-color : lightblue")
        else:
            self.cb_use_filter.setText("Filter DISABLED (Raw Vision)")
            self.cb_use_filter.setStyleSheet("")

    def toggle_glare(self):
        self.simulate_glare = self.cb_glare.isChecked()

    def set_noise(self):
        self.noise_level = self.sl_noise.value()

    def reset_simulation(self):
        """Setzt die komplette Simulation auf den Startzustand zurück."""
        print("--- Simulation Reset ---")
        
        # 1. IBVS und Regelung stoppen
        self.is_servoing = False
        self.btn_toggle_servo.setChecked(False)
        self.btn_toggle_servo.setText("2. Start Servoing")
        self.btn_toggle_servo.setEnabled(False)
        
        # 2. Statusvariablen löschen
        self.desired_features = None
        self.lbl_error.setText("Error: -")
        self.current_q_dot = np.zeros(6)
        self.debug_filtered_pts = None
        
        # 3. Filter und Störgrößen zurücksetzen
        self.filter.reset()
        self.sl_noise.setValue(0)
        
        self.cb_glare.setChecked(False)
        self.toggle_glare() # Logik ausführen
        
        self.cb_use_filter.setChecked(False)
        self.toggle_filter() # Logik ausführen
        
        # 4. Roboter auf Startpose (qn) zurücksetzen
        initial_q = self.sim.robot.myq_start.copy() # Startpose aus RobotSimulator
        self.sim.update_joints(initial_q)
        
        # 5. Slider wieder aktivieren und auf Startwerte setzen
        for i, s in enumerate(self.sliders):
            s.blockSignals(True) # Verhindert, dass manual_slider_change gefeuert wird
            deg = int(np.rad2deg(initial_q[i]))
            s.setValue(deg)
            self.labels[i].setText(f"{deg}°")
            s.setEnabled(True)
            s.blockSignals(False)
            
        # 6. Alles neu zeichnen
        self.update_plots()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SimulatorGUI()
    gui.show()
    sys.exit(app.exec_())