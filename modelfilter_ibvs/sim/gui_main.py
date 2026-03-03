####################################################################################################
# GUI für die IBVS Simulation
#
# Autor: Paul Glaser
#
# Version 2.0 - Stand 03.03.2026
# 
# Ziel: Ein einfacher Simulator mit einem 6-Achs-Roboter, einer Kamera und einem IBVS-Controller. Inklusive eines EKF-Filters für die Homographie-Schätzung.
#
# Wichtig: Um Pfadfehler zu vermeiden, sollte dieses Skript aus dem Hauptverzeichnis des Projekts gestartet werden (dort, wo auch "sim" und "core" liegen), mit:
# python -m modelfilter_ibvs.sim.gui_main
####################################################################################################
VERSION = "2.0"

import sys, os, cv2, csv, json
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QGroupBox, QPushButton, 
                             QComboBox, QDoubleSpinBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pyqtgraph as pg

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

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
        self.noise_level_initial = 2   # kein float
        self.noise_level = self.noise_level_initial
        
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

        # Lade ein beliebiges Bild (muss im gleichen Ordner liegen oder Pfad anpassen)
        try:
            self.custom_texture = cv2.imread("modelfilter_ibvs/sim/images/ATP-DDR3-2G-AL56M72B8BJH9S.jpg") 
        except Exception as e:
            print(f"Fehler beim Laden des Bildes, nutze Schachbrett: {e}")
            self.custom_texture = None

        self.benchmark_mode = False
        self.benchmark_scenarios = []
        self.current_scenario_idx = 0
        self.benchmark_summary_path = ""
        self.current_scenario = None
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("IBVS Simulation - Version " + VERSION)
        self.setGeometry(100, 100, 1400, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Layouts
        top_layout = QHBoxLayout() # Für Statusanzeigen
        left_layout = QVBoxLayout() # Plots (Simulation + Graphen)
        right_layout = QVBoxLayout() # Controls
        
        # --- Matplotlib Canvas (3D & Camera) ---
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_cam = self.fig.add_subplot(122)
        left_layout.addWidget(self.canvas)
        
        # --- NEU: PyQtGraph Live Plots ---
        bottom_plot_layout = QHBoxLayout()
        
        # Plot 1: Feature Error
        self.plot_widget_error = pg.PlotWidget(title="Feature Error (von IBVS Controller)")
        self.plot_widget_error.setLabel('left', 'Fehler', units='px')
        self.plot_widget_error.setLabel('bottom', 'Zeit', units='Frames')
        self.plot_widget_error.showGrid(x=True, y=True, alpha=0.3)
        self.curve_error = self.plot_widget_error.plot(pen=pg.mkPen(color='r', width=2))
        
        # Plot 2: Control Effort (Geschwindigkeit)
        self.plot_widget_vel = pg.PlotWidget(title="Control Effort (Norm v_cam)")
        self.plot_widget_vel.setLabel('left', 'Geschw.', units='m/s')
        self.plot_widget_vel.setLabel('bottom', 'Zeit', units='Frames')
        self.plot_widget_vel.showGrid(x=True, y=True, alpha=0.3)
        self.curve_vel = self.plot_widget_vel.plot(pen=pg.mkPen(color='b', width=2))

        # Feste Höhe für die Plots, damit sie die Kameraansicht nicht verdrängen
        self.plot_widget_error.setFixedHeight(220)
        self.plot_widget_vel.setFixedHeight(220)

        bottom_plot_layout.addWidget(self.plot_widget_error)
        bottom_plot_layout.addWidget(self.plot_widget_vel)
        
        # Graphen unter das Matplotlib Canvas einfügen
        left_layout.addLayout(bottom_plot_layout)

        # Daten-Speicher für die Plots initialisieren
        self.plot_data_time = []
        self.plot_data_error = []
        self.plot_data_vel = []
        self.frame_counter = 0
        # ---------------------------------
        
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
        self.sb_threshold.setRange(0.0001, 10.0)
        self.sb_threshold.setDecimals(4)
        self.sb_threshold.setSingleStep(0.001)
        self.sb_threshold.setValue(0.001)
        error_layout.addWidget(self.sb_threshold)
        
        ibvs_layout.addWidget(self.btn_set_desired)
        ibvs_layout.addWidget(self.btn_toggle_servo)
        ibvs_layout.addLayout(error_layout)
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
        
        filter_layout.addWidget(QLabel("Tracking Filter:"))
        
        self.cb_filter_select = QComboBox()
        self.cb_filter_select.addItems(FilterFactory.get_available_filters())
        self.cb_filter_select.setCurrentIndex(0)
        self.cb_filter_select.currentTextChanged.connect(self.change_filter)
        
        self.cb_glare = QPushButton("SIMULATE GLARE (Blind)")
        self.cb_glare.setCheckable(True)
        self.cb_glare.setStyleSheet("background-color : red")
        self.cb_glare.clicked.connect(self.toggle_glare)
        
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Noise:"))
        self.sl_noise = QSlider(Qt.Horizontal)
        self.sl_noise.setRange(0, 20)
        self.sl_noise.valueChanged.connect(self.set_noise)
        self.sl_noise.setValue(self.noise_level_initial)
        noise_layout.addWidget(self.sl_noise)

        filter_layout.addWidget(self.cb_filter_select)
        
        self.filter_params_group = QGroupBox("Filter Parameters")
        self.filter_params_layout = QVBoxLayout()
        self.filter_params_group.setLayout(self.filter_params_layout)
        filter_layout.addWidget(self.filter_params_group)
        self.filter_params_group.hide()

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

        reset_group = QGroupBox("Reset & Benchmark")
        reset_layout = QVBoxLayout()

        self.btn_reset = QPushButton("Reset Simulation")
        self.btn_reset.setStyleSheet("background-color : #f0ad4e")
        self.btn_reset.clicked.connect(self.reset_simulation)
        reset_layout.addWidget(self.btn_reset)

        self.btn_benchmark = QPushButton("Run JSON Benchmark")
        self.btn_benchmark.setStyleSheet("background-color : #5bc0de") # Schönes Blau
        self.btn_benchmark.clicked.connect(self.start_benchmark)
        reset_layout.addWidget(self.btn_benchmark)

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
        img = self.sim.get_camera_image(texture_img=self.custom_texture)

        # Speichere es als Referenz im Feature-Extractor
        self.vision.set_reference_image(img)

        # Wir holen uns die echten XFeat/ORB Punkte als Regler-Ziel!
        self.desired_features = self.vision.get_reference_coordinates()

        self.btn_toggle_servo.setEnabled(True)
        print("Desired View set in Vision System!")
        self.update_plots()

    def start_benchmark(self):
        # 1. JSON auswählen
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Benchmark JSON", "", "JSON Files (*.json)", options=options)
        if not filepath: return

        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.benchmark_scenarios = data.get("scenarios", [])
        if not self.benchmark_scenarios:
            print("Keine Szenarien gefunden!")
            return

        self.benchmark_mode = True
        self.current_scenario_idx = 0
        
        # 2. Master-Summary CSV vorbereiten
        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark_summary_path = os.path.join(downloads_path, f"benchmark_summary_{timestamp}.csv")
        
        with open(self.benchmark_summary_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Scenario', 'Extractor', 'Filter', 'Success', 'Frames', 'Final_Avg_Error', 'Control_Effort_Integral'])
            
        # 3. Den ersten Test anstoßen!
        self.run_next_scenario()

    def run_next_scenario(self):
        if not self.benchmark_mode: return

        # Prüfen ob wir fertig sind
        if self.current_scenario_idx >= len(self.benchmark_scenarios):
            self.benchmark_mode = False
            QMessageBox.information(self, "Benchmark Beendet", f"Alle Tests abgeschlossen!\nSummary gespeichert unter:\n{self.benchmark_summary_path}")
            return

        self.current_scenario = self.benchmark_scenarios[self.current_scenario_idx]
        print(f"\n--- Starte Benchmark: {self.current_scenario['name']} ---")

        # Schritt A: Extractor & Filter wechseln (Achtung: das ruft intern reset_simulation auf!)
        self.cb_extractor.setCurrentText(self.current_scenario["extractor_name"])
        self.cb_filter_select.setCurrentText(self.current_scenario["filter_name"])
        
        # Wir geben der GUI 200ms Zeit, den Reset durchzuführen, bevor wir die Posen anfahren
        QTimer.singleShot(200, self._benchmark_step_2)

    def _benchmark_step_2(self):
        # Schritt B: Target anfahren und Referenzbild aufnehmen
        target_q = np.deg2rad(self.current_scenario["target_joints"])
        # self.sim.q = target_q
        self.sim.update_joints(target_q)
        self.set_desired_view() # Virtueller Klick auf "Set Desired"

        # Kurz warten, bis RANSAC und die Zielfeatures extrahiert wurden
        QTimer.singleShot(200, self._benchmark_step_3)

    def _benchmark_step_3(self):
        # Schritt C: Startpose anfahren, Noise/Threshold setzen und GO!
        start_q = np.deg2rad(self.current_scenario["start_joints"])
        # self.sim.q = start_q
        self.sim.update_joints(start_q)
        self.update_sliders_from_robot()

        self.sl_noise.setValue(int(self.current_scenario["noise_std"]))
        self.sb_threshold.setValue(self.current_scenario["stop_error"])

        # Start Servoing!
        self.btn_toggle_servo.setChecked(True)
        self.toggle_servoing()

    def toggle_servoing(self):
        self.is_servoing = self.btn_toggle_servo.isChecked()
        if self.is_servoing:
            self.btn_toggle_servo.setText("Stop Servoing")
            for s in self.sliders: s.setEnabled(False)
            self.sb_threshold.setEnabled(False) # NEU: Eingabefeld ausgrauen

            # Zwinge den aktiven Filter, auf die aktuelle Roboterpose zu springen,
            # bevor die erste Regler-Geschwindigkeit berechnet wird.
            if self.active_filter is not None:
                self.active_filter.force_relocalization()

            # --- Plot-Daten zurücksetzen ---
            self.plot_data_time = []
            self.plot_data_error = []
            self.plot_data_vel = []
            self.frame_counter = 0
        else:
            self.btn_toggle_servo.setText("Start Servoing")
            for s in self.sliders: s.setEnabled(True)
            self.sb_threshold.setEnabled(True)  # NEU: Eingabefeld wieder freigeben
            self.update_sliders_from_robot()

    def target_reached(self, final_error, timeout=False):
        self.btn_toggle_servo.setChecked(False)
        self.toggle_servoing()

        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        
        # Dateiname je nach Modus
        if self.benchmark_mode:
            name = self.current_scenario['name']
            filename = f"run_detail_{name}.csv"
        else:
            filter_name = self.cb_filter_select.currentText().replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ibvs_run_{filter_name}_{timestamp}.csv"
            
        filepath = os.path.join(downloads_path, filename)

        # Metrik: Control Effort berechnen (Integral der Geschwindigkeitsquadrate)
        control_effort = sum([v**2 for v in self.plot_data_vel])

        try:
            with open(filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Frame', 'avg_error', 'Velocity_Norm'])
                for t, e, v in zip(self.plot_data_time, self.plot_data_error, self.plot_data_vel):
                    writer.writerow([t, e, v])
                    
            if self.benchmark_mode:
                # In die Master-Summary schreiben
                with open(self.benchmark_summary_path, mode='a', newline='') as f_sum:
                    w_sum = csv.writer(f_sum)
                    success = "No (Timeout)" if timeout else "Yes"
                    w_sum.writerow([
                        self.current_scenario['name'],
                        self.current_scenario['extractor_name'],
                        self.current_scenario['filter_name'],
                        success,
                        self.frame_counter,
                        round(final_error, 5),
                        round(control_effort, 4)
                    ])
                
                # Das nächste Szenario triggern (Kurze Pause, damit die GUI atmen kann)
                self.current_scenario_idx += 1
                QTimer.singleShot(1000, self.run_next_scenario) 
            else:
                # Manueller Modus -> Pop-Up anzeigen
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning if timeout else QMessageBox.Information)
                msg.setWindowTitle("Timeout!" if timeout else "Ziel erreicht!")
                msg.setText("Zeit abgelaufen." if timeout else "Die visuelle Regelung war erfolgreich.")
                msg.setInformativeText(f"End-Fehler: {final_error:.5f}\n\nCSV gespeichert unter:\n{filepath}")
                msg.exec_()
                
        except Exception as e:
            print(f"Fehler beim Speichern der CSV: {e}")

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
                if filter_name == "Standard Kalman Filter":
                    self.active_filter.Q_scalar = 500.0
                    self.active_filter.R_scalar = 100.0
                    self.active_filter.gate_thresh = 20.0
                elif filter_name == "Extended Kalman Filter":
                    self.active_filter.Q_scalar = 1.0
                    self.active_filter.R_scalar = 50.0
                    self.active_filter.gate_thresh = 20.0
                elif filter_name == "Unscented Kalman Filter":
                    self.active_filter.Q_scalar = 1.0
                    self.active_filter.R_scalar = 50.0
                    self.active_filter.gate_thresh = 20.0
                elif filter_name == "Error-State Kalman Filter":
                    self.active_filter.Q_scalar = 1.0
                    self.active_filter.R_scalar = 50.0
                    self.active_filter.gate_thresh = 20.0

                # Q-Parameter (Prozessrauschen)
                q_layout = QHBoxLayout()
                q_layout.addWidget(QLabel("Q (Process Noise):"))
                sb_q = QDoubleSpinBox()
                sb_q.setRange(0.0001, 1000.0)
                sb_q.setDecimals(4)
                sb_q.setSingleStep(0.01)
                sb_q.setValue(self.active_filter.Q_scalar) # Dein gewünschter Default
                
                # R-Parameter (Messrauschen)
                r_layout = QHBoxLayout()
                r_layout.addWidget(QLabel("R (Meas. Noise):"))
                sb_r = QDoubleSpinBox()
                sb_r.setRange(0.0001, 1000.0)
                sb_r.setDecimals(4)
                sb_r.setSingleStep(0.01)
                sb_r.setValue(self.active_filter.R_scalar) # Dein gewünschter Default
                
                # Gate Threshold (Outlier Detection)
                gate_thresh_layout = QHBoxLayout()
                gate_thresh_layout.addWidget(QLabel("gate thresh:"))
                sb_gate_thresh = QDoubleSpinBox()
                sb_gate_thresh.setRange(1, 100.0)
                sb_gate_thresh.setDecimals(1)
                sb_gate_thresh.setSingleStep(1)
                sb_gate_thresh.setValue(self.active_filter.gate_thresh) # Dein gewünschter Default

                # Relocalization erzwingen (beim druecken des Buttons)
                relocate_layout = QHBoxLayout()
                btn_relocate = QPushButton("Force Relocalization")
                btn_relocate.clicked.connect(lambda: self.active_filter.force_relocalization())
                relocate_layout.addWidget(btn_relocate)

                
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
                self.filter_params_layout.addLayout(relocate_layout)

            # -- PARAMETER FÜR 1-EURO --
            elif filter_name == "1-Euro Filter":
                # Baue hier zwei SpinBoxen für min_cutoff und beta
                # und verbinde sie mit self.active_filter.set_params
                
                pass
            
            print(f"Filter gewechselt zu: {filter_name}")
            
        if hasattr(self, 'is_servoing'):
            self.reset_simulation()

    def control_loop(self):

        # 1. Datenbeschaffung
        # Übergebe den Noise Level an das gerenderte Bild (Bilder rendern & Keypoints extrahieren)
        self.current_img = self.sim.get_camera_image(noise_std=self.noise_level, texture_img=self.custom_texture)
        
        if self.benchmark_mode and self.current_scenario is not None:
            blind_intervals = self.current_scenario.get("blind_intervals", [])
            is_blind = False
            
            # Prüfe, ob der aktuelle Frame in IRGENDEINEM der Intervalle liegt
            for interval in blind_intervals:
                if len(interval) == 2 and interval[0] <= self.frame_counter <= interval[1]:
                    self.simulate_glare = True
                    break # Wir haben ein Treffer-Intervall gefunden, müssen nicht weiter suchen
                else:
                    self.simulate_glare = False

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

        if not self.is_servoing or self.desired_features is None:
            self.frame_counter += 1
            self.update_plots() # Damit die Zeitachse weiterläuft, auch wenn wir nicht servoen
            return

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
            print(f"Frame: {self.frame_counter} - Verloren! Blendung aktiv, aber kein Filter vorhanden.")
            self.debug_filtered_pts = None
            self.frame_counter += 1
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
            R = self.sim.T_ee_cam.R; t = self.sim.T_ee_cam.t
            v_lin = R @ v_cam_cmd[0:3]; v_ang = R @ v_cam_cmd[3:6]
            v_lin = v_lin + np.cross(t, v_ang)
            v_ee_cmd = np.concatenate((v_lin, v_ang))

            # --- Damped Least Squares (DLS) für die Roboter-Kinematik ---
            # Verhindert, dass der Roboter an kinematischen Singularitäten ausrastet
            damp_robot = 0.1 
            I6 = np.eye(6)
            # Für 6x6 Matrix J: J^T * (J * J^T + lambda^2 * I)^-1
            J_dls_inv = J.T @ np.linalg.inv(J @ J.T + (damp_robot**2) * I6)
            
            q_dot = J_dls_inv @ v_ee_cmd * 0.5 # Gain etwas runter, damit es nicht zu ruckartig wird
            
            # SPEICHERN für nächsten Filter Schritt!
            self.current_q_dot = q_dot 
            
            self.sim.update_joints(self.sim.q + q_dot * dt)

            # --- Live Plotting Updates ---
            self.frame_counter += 1
            # Wir müssen sicherstellen, dass wir gleich viele Punkte vergleichen.
            # In IBVS nutzt der Regler oft den Fehlervektor 'error' direkt.
            # Wenn dein Regler 'error' zurückgibt, nimm einfach dessen Norm!
            # Alternativ manueller RMSE der getrackten Punkte.
            
            # Nimm einfach die L2-Norm der berechneten Kamerageschwindigkeit
            vel_norm = np.linalg.norm(v_cam_cmd)

            self.plot_data_time.append(self.frame_counter)
            self.plot_data_error.append(avg_error)
            self.plot_data_vel.append(vel_norm)

            print(f"Frame: {self.frame_counter}, Avg Error: {avg_error:.4f}, Vel Norm: {vel_norm:.4f}")

            # Zeichnen! (Wir übergeben nur die letzten 200 Punkte für eine flüssige GUI)
            show_points = 200
            self.curve_error.setData(self.plot_data_time[-show_points:], self.plot_data_error[-show_points:])
            self.curve_vel.setData(self.plot_data_time[-show_points:], self.plot_data_vel[-show_points:])

            # Lese den aktuellen Threshold aus der GUI
            current_threshold = self.sb_threshold.value()

            timeout = False
            if self.benchmark_mode and self.current_scenario is not None:
                if self.frame_counter >= self.current_scenario.get("max_frames", 500):
                    timeout = True

            # Wir reagieren auf Ziel-Erreichung ODER auf Timeout
            if (avg_error < current_threshold and self.frame_counter > 5) or timeout:
                # print(f"Ziel erreicht! End-Fehler (Avg): {avg_error:.4f}")
                self.is_servoing = False
                self.btn_toggle_servo.setChecked(False)
                self.btn_toggle_servo.setText("2. Start Servoing")
                
                self.sb_threshold.setEnabled(True) # NEU: Eingabefeld wieder freigeben
                
                for s in self.sliders: 
                    s.setEnabled(True)
                self.update_sliders_from_robot()
                self.target_reached(avg_error, timeout=timeout)
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
        if self.debug_prints:
            print("[GUI_MAIN][update_plots] Aktuelles Kamerabild Shape: ", self.current_img.shape if self.current_img is not None else "None")
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
            if "INIT" in status_filter:
                self.ax_cam.text(x_pos, y_pos, f"Filter: {status_filter}", fontsize=10, bbox=dict(facecolor='yellow'), transform=self.ax_cam.transAxes)
            elif "PREDICT" in status_filter:
                self.ax_cam.text(x_pos, y_pos, f"Filter: {status_filter}", fontsize=10, bbox=dict(facecolor='lightblue'), transform=self.ax_cam.transAxes)
            elif "MISSING" in status_filter:
                self.ax_cam.text(x_pos, y_pos, f"Filter: {status_filter}", fontsize=10, bbox=dict(facecolor='red'), transform=self.ax_cam.transAxes)
            elif "REJECT" in status_filter:
                self.ax_cam.text(x_pos, y_pos, f"Filter: {status_filter}", fontsize=10, bbox=dict(facecolor='red'), transform=self.ax_cam.transAxes)
            elif "UPDATE" in status_filter:
                self.ax_cam.text(x_pos, y_pos, f"Filter: {status_filter}", fontsize=10, bbox=dict(facecolor='lightgreen'), transform=self.ax_cam.transAxes)
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
        self.sl_noise.setValue(self.noise_level_initial)
        
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