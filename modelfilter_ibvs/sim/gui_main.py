####################################################################################################
# GUI für die IBVS Simulation
#
# Autor: Paul Glaser
#
# Version 1.0 - Stand 20.02.2026
# 
# Ziel: Ein einfacher Simulator mit einem 6-Achs-Roboter, einer Kamera und einem IBVS-Controller. Inklusive eines EKF-Filters für die Homographie-Schätzung.
#
# Wichtig: Um Pfadfehler zu vermeiden, sollte dieses Skript aus dem Hauptverzeichnis des Projekts gestartet werden (dort, wo auch "sim" und "core" liegen), mit:
# python -m modelfilter_ibvs.sim.gui_main
####################################################################################################
VERSION = "1.0"

import sys, os, cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QPushButton, 
                             QComboBox, QDoubleSpinBox) # <-- QDoubleSpinBox ergänzen
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Imports (Pfade anpassen wie gehabt)
try:
    from modelfilter_ibvs.sim.robot_sim import RobotSimulator
    from modelfilter_ibvs.core.ibvs import IBVSController
    from modelfilter_ibvs.core.detectors.factory import ExtractorFactory
    from modelfilter_ibvs.core.filters.factory import FilterFactory
except ImportError:
    path_file = os.path.abspath(__file__)
    path_parent = os.path.dirname(path_file)
    path_grandpaent = os.path.dirname(path_parent)
    sys.path.insert(0, path_parent)
    sys.path.insert(0, path_grandpaent)
    from sim.robot_sim import RobotSimulator
    from core.ibvs import IBVSController
    from core.detectors.factory import ExtractorFactory
    from core.filters.factory import FilterFactory

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
        self.active_filter = None
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
        self.setGeometry(100, 100, 1400, 800)
        
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
        
        error_layout = QHBoxLayout()
        self.lbl_error = QLabel("Error: -")
        error_layout.addWidget(self.lbl_error)
        error_layout.addWidget(QLabel("Stop Thresh.:"))
        self.sb_threshold = QDoubleSpinBox()
        self.sb_threshold.setRange(0.0001, 10.0) # Min/Max Werte
        self.sb_threshold.setDecimals(4)         # 4 Nachkommastellen
        self.sb_threshold.setSingleStep(0.001)    # Schrittweite bei Pfeilklick
        self.sb_threshold.setValue(0.001)         # Dein bisheriger Standardwert
        error_layout.addWidget(self.sb_threshold)
        
        ibvs_layout.addWidget(self.btn_set_desired)
        ibvs_layout.addWidget(self.btn_toggle_servo)
        ibvs_layout.addLayout(error_layout) # Dem IBVS-Block hinzufügen
        ibvs_group.setLayout(ibvs_layout)
        right_layout.addWidget(ibvs_group)

        # --- Vision Settings ---
        vision_group = QGroupBox("Vision System")
        vision_layout = QVBoxLayout()

        self.cb_extractor = QComboBox()
        self.cb_extractor.addItems(ExtractorFactory.get_available_extractors())
        self.cb_extractor.currentTextChanged.connect(self.change_extractor)

        vision_layout.addWidget(QLabel("Feature Extractor:"))
        vision_layout.addWidget(self.cb_extractor)
        vision_group.setLayout(vision_layout)
        right_layout.addWidget(vision_group)

        # 2. Filter & Noise Simulation
        filter_group = QGroupBox("Filter & Noise Simulation")
        filter_layout = QVBoxLayout()
        
        # Im Block "Filter & Noise Simulation" (Ersetze den bisherigen Button hiermit)
        filter_layout.addWidget(QLabel("Tracking Filter:"))
        
        self.cb_filter_select = QComboBox()
        self.cb_filter_select.addItems(FilterFactory.get_available_filters())
        # Setze Standard auf Homography (Index 1)
        self.cb_filter_select.setCurrentIndex(0)
        self.cb_filter_select.currentTextChanged.connect(self.change_filter)
        
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

        filter_layout.addWidget(self.cb_filter_select)
        
        # NEU: Ein Container für die dynamischen Parameter des aktiven Filters
        self.filter_params_group = QGroupBox("Filter Parameters")
        self.filter_params_layout = QVBoxLayout()
        self.filter_params_group.setLayout(self.filter_params_layout)
        filter_layout.addWidget(self.filter_params_group)
        self.filter_params_group.hide() # Zunächst verstecken, bis ein Filter gewählt ist

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

        # Wir holen uns die echten XFeat/ORB Punkte als Regler-Ziel!
        self.desired_features = self.vision.get_reference_coordinates()

        self.btn_toggle_servo.setEnabled(True)
        print("Desired View set in Vision System!")
        self.update_plots()

    def toggle_servoing(self):
        self.is_servoing = self.btn_toggle_servo.isChecked()
        if self.is_servoing:
            self.btn_toggle_servo.setText("Stop Servoing")
            for s in self.sliders: s.setEnabled(False)
            self.sb_threshold.setEnabled(False) # NEU: Eingabefeld ausgrauen
        else:
            self.btn_toggle_servo.setText("Start Servoing")
            for s in self.sliders: s.setEnabled(True)
            self.sb_threshold.setEnabled(True)  # NEU: Eingabefeld wieder freigeben
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
        print(f"Wechsle Feature Extractor zu: {extractor_name}")
        
        # NEU: Instanziierung über die Factory
        try:
            self.vision = ExtractorFactory.create(extractor_name)
        except Exception as e:
            print(f"Fehler beim Laden von {extractor_name}: {e}")
            return
            
        if hasattr(self, 'is_servoing'):
            self.reset_simulation()

    def change_filter(self, filter_name):
        self.clear_layout(self.filter_params_layout)
        
        try:
            K = self.sim.get_intrinsic_matrix()
            self.active_filter = FilterFactory.create(filter_name, K)
        except Exception as e:
            print(f"Fehler beim Laden von {filter_name}: {e}")
            return
        
        if self.active_filter is None:
            self.filter_params_group.hide()
            print("Filter deaktiviert (-)")
        else:
            self.filter_params_group.show()
            
            # -- PARAMETER FÜR HOMOGRAPHIE --
            if filter_name == "Homography Model (SL3)":
                h_layout = QHBoxLayout()
                h_layout.addWidget(QLabel("Alpha (Meas. Trust):"))
                
                sb_alpha = QDoubleSpinBox()
                sb_alpha.setRange(0.0, 1.0)
                sb_alpha.setSingleStep(0.05)
                # Setze den Default-Wert aus dem frisch erstellten Filter
                sb_alpha.setValue(self.active_filter.alpha) 
                
                # Lambda-Funktion, um den Wert on-the-fly an den Filter zu senden
                sb_alpha.valueChanged.connect(lambda val: setattr(self.active_filter, 'alpha', val))
                
                h_layout.addWidget(sb_alpha)
                self.filter_params_layout.addLayout(h_layout)
                
            # -- PARAMETER FÜR EKF / ESKF / UKF --
            elif filter_name in ["Standard Kalman Filter", "Extended Kalman Filter", "Error-State Kalman Filter", "Unscented Kalman Filter"]:
                # Q-Parameter (Prozessrauschen)
                q_layout = QHBoxLayout()
                q_layout.addWidget(QLabel("Q (Process Noise):"))
                sb_q = QDoubleSpinBox()
                sb_q.setRange(0.0001, 10.0)
                sb_q.setDecimals(4)
                sb_q.setSingleStep(0.01)
                sb_q.setValue(0.0001) # Dein gewünschter Default
                
                # R-Parameter (Messrauschen)
                r_layout = QHBoxLayout()
                r_layout.addWidget(QLabel("R (Meas. Noise):"))
                sb_r = QDoubleSpinBox()
                sb_r.setRange(0.0001, 10.0)
                sb_r.setDecimals(4)
                sb_r.setSingleStep(0.01)
                sb_r.setValue(0.005) # Dein gewünschter Default
                
                # R-Parameter (Messrauschen)
                gate_thresh_layout = QHBoxLayout()
                gate_thresh_layout.addWidget(QLabel("gate thresh:"))
                sb_gate_thresh = QDoubleSpinBox()
                sb_gate_thresh.setRange(1, 100.0)
                sb_gate_thresh.setDecimals(1)
                sb_gate_thresh.setSingleStep(1)
                sb_gate_thresh.setValue(20) # Dein gewünschter Default
                
                # Update-Funktion für den EKF
                def update_ekf_params():
                    # Hier setzt du die Skalare auf die Matrizen deines EKF um
                    q_val = sb_q.value()
                    r_val = sb_r.value()
                    gate_thresh_val = sb_gate_thresh.value()
                    # Angenommen, dein EKF hat set_Q und set_R Methoden:
                    if hasattr(self.active_filter, 'set_Q_R_gate'):
                        self.active_filter.set_Q_R_gate(q_val, r_val, gate_thresh_val)
                
                sb_q.valueChanged.connect(update_ekf_params)
                sb_r.valueChanged.connect(update_ekf_params)
                sb_gate_thresh.valueChanged.connect(update_ekf_params)
                
                # Einmal initial aufrufen
                update_ekf_params()
                
                q_layout.addWidget(sb_q)
                r_layout.addWidget(sb_r)
                gate_thresh_layout.addWidget(sb_gate_thresh)
                self.filter_params_layout.addLayout(q_layout)
                self.filter_params_layout.addLayout(r_layout)
                self.filter_params_layout.addLayout(gate_thresh_layout)

            # -- PARAMETER FÜR 1-EURO --
            elif filter_name == "1-Euro Filter":
                # Baue hier zwei SpinBoxen für min_cutoff und beta
                # und verbinde sie mit self.active_filter.set_params
                
                pass
            
            print(f"Filter gewechselt zu: {filter_name}")
            
        if hasattr(self, 'is_servoing'):
            self.reset_simulation()

    def control_loop(self):
        if not self.is_servoing or self.desired_features is None:
            if self.desired_features is None: return

        # 1. Datenbeschaffung
        # Übergebe den Noise Level an das gerenderte Bild (Bilder rendern & Keypoints extrahieren)
        self.current_img = self.sim.get_camera_image(noise_std=self.noise_level)
        
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

        # 2. Filter Update (Nur wenn ein Filter aktiv ist!)
        Z_est = self.sim.get_mean_depth()
        J = self.sim.get_jacobian()
        dt = 0.05
        if self.active_filter is not None:
            v_ee = J @ self.current_q_dot
            self.active_filter.predict(v_ee, Z_est, dt=dt) 
            if self.debug_prints:
                print("[GUI_MAIN][control_loop] v_ee (Twist): ", v_ee)
                print("[GUI_MAIN][control_loop] Z_est: ", Z_est)

            self.active_filter.update(self.current_raw_features, self.current_matched_ref)

        # 3. ENTSCHEIDUNGSLOGIK FÜR DEN REGLER
        features_for_control = None
        target_for_control = None
        
        # Fall A: Wir wollen den Filter nutzen UND er ist bereit
        if self.active_filter is not None and self.active_filter.initialized:
            # MIT FILTER: Wir nutzen ALLE initialen Keypoints (z.B. 4000 Stück).
            features_for_control = self.active_filter.get_projected_points(self.desired_features)
            target_for_control = self.desired_features
            self.debug_filtered_pts = features_for_control
            
        # Fall B: Kein Filter gewünscht ODER Filter noch nicht bereit -> Nimm Rohdaten
        elif self.current_raw_features is not None:
            # Kein Filter (oder noch nicht init) -> Nutze Rohdaten
            features_for_control = self.current_raw_features
            target_for_control = self.current_matched_ref
            self.debug_filtered_pts = None 
            
        else:
            print("Verloren! Blendung aktiv, aber kein Filter vorhanden.")
            self.debug_filtered_pts = None
            return
            
        # 4. Regelung (Nur wenn Servoing aktiv)
        if self.is_servoing:
            # Wir übergeben das dynamische target_for_control!
            v_cam_cmd, error_norm = self.controller.compute_velocity(
                features_for_control, 
                target_for_control, 
                Z_est=Z_est
            )
            
            # Da wir jetzt viel mehr Punkte haben, ist der Fehler (Norm) absolut 
            # gesehen viel größer. Wir mitteln den Fehler für die Anzeige/Stopp.
            avg_error = error_norm / features_for_control.shape[1]
            self.lbl_error.setText(f"Avg Error: {avg_error:.4f}")
            
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

            # NEU: Lese den aktuellen Threshold aus der GUI
            current_threshold = self.sb_threshold.value()
            
            # Passe die Stopp-Bedingung an
            if avg_error < current_threshold:
                print(f"Ziel erreicht! End-Fehler (Avg): {avg_error:.4f}")
                self.is_servoing = False
                self.btn_toggle_servo.setChecked(False)
                self.btn_toggle_servo.setText("2. Start Servoing")
                
                self.sb_threshold.setEnabled(True) # NEU: Eingabefeld wieder freigeben
                
                for s in self.sliders: 
                    s.setEnabled(True)
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
                             'r.', markersize=2, alpha=0.3, label='Target Keypoints')
            
        # Gefilterte Position (Blaues Kreuz)
        if self.debug_filtered_pts is not None:
            self.ax_cam.plot(self.debug_filtered_pts[0,:], self.debug_filtered_pts[1,:], 
                             'b.', markersize=2, alpha=0.5, label='Filtered Keypoints')
        
        # Reale Ecken (zur Kontrolle, wo sie WIRKLICH sind)
        true_corners = self.sim.get_camera_view()
        self.ax_cam.plot(true_corners[0,:], true_corners[1,:], 'g+', markersize=8, label='True Corners')

        # Status anzeigen
        x_pos = 0.05; y_pos = 0.9
        if self.active_filter is not None:
            status_filter = self.active_filter.status
            if status_filter == "INIT":
                self.ax_cam.text(x_pos, y_pos, "Filter: INIT (Not Ready)", fontsize=10, bbox=dict(facecolor='yellow'), transform=self.ax_cam.transAxes)
            elif status_filter == "PREDICT":
                self.ax_cam.text(x_pos, y_pos, "Filter: PREDICTING", fontsize=10, bbox=dict(facecolor='lightblue'), transform=self.ax_cam.transAxes)
            elif status_filter == "MISSING":
                self.ax_cam.text(x_pos, y_pos, "Filter: MISSING (No Data)", fontsize=10, bbox=dict(facecolor='red'), transform=self.ax_cam.transAxes)
            elif status_filter == "REJECT (GLARE)":
                self.ax_cam.text(x_pos, y_pos, "Filter: REJECT (Glare)", fontsize=10, bbox=dict(facecolor='red'), transform=self.ax_cam.transAxes)
            elif status_filter == "UPDATE":
                self.ax_cam.text(x_pos, y_pos, "Filter: UPDATE", fontsize=10, bbox=dict(facecolor='lightgreen'), transform=self.ax_cam.transAxes)
            else:
                self.ax_cam.text(x_pos, y_pos, f"Filter: {status_filter} (UNKNOWN)", fontsize=10, bbox=dict(facecolor='gray'), transform=self.ax_cam.transAxes)
        else:
            self.ax_cam.text(x_pos, y_pos, f"Filter: NONE", fontsize=10, bbox=dict(facecolor='lightgray'), transform=self.ax_cam.transAxes)

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
        
        self.sb_threshold.setEnabled(True) # NEU: Feld entsperren
        
        # 2. Statusvariablen löschen
        self.desired_features = None
        self.lbl_error.setText("Error: -")
        self.current_q_dot = np.zeros(6)
        self.debug_filtered_pts = None
        
        # 3. Filter und Störgrößen zurücksetzen
        if self.active_filter is not None:
            self.active_filter.reset()
        self.sl_noise.setValue(0)
        
        self.cb_glare.setChecked(False)
        self.toggle_glare() # Logik ausführen
        
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

    def clear_layout(self, layout):
        """Löscht alle Widgets aus einem Layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SimulatorGUI()
    gui.show()
    sys.exit(app.exec_())