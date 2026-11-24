import os
import json
import traceback
import datetime
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog

from CalibrationModel import CalibrationModel
from CalibrationView import CalibrationView


class CalibrationController(QObject):
    def __init__(self, model: CalibrationModel, view: CalibrationView):
        super().__init__()
        self._model = model
        self._view = view
        self._connect_view_signals()
        self._connect_model_signals()
        self._apply_config()

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------
    def _connect_view_signals(self):
        self._view.load_requested.connect(self._on_load_images)
        self._view.capture_requested.connect(self._on_capture_image)
        self._view.recognize_requested.connect(self._on_recognize)
        self._view.recog_calib_browse_requested.connect(self._on_recog_calib_browse)
        self._view.calibrate_requested.connect(self._on_run_calibration)
        self._view.save_requested.connect(self._on_save)
        self._view.clear_all_requested.connect(self._on_clear_all)
        self._view.clear_selected_requested.connect(self._on_clear_selected)
        self._view.image_selected.connect(self._on_image_selected)

    def _connect_model_signals(self):
        self._model.log_message.connect(print)
        self._model.images_loaded.connect(self._on_images_loaded)
        self._model.calibration_done.connect(self._on_calibration_done)
        self._model.save_done.connect(self._on_save_done)
        self._model.error_occurred.connect(self._view.show_error)

    def _apply_config(self):
        cfg = self._model.load_config()
        self._view.set_defaults(cfg)

    # ------------------------------------------------------------------
    # Slots for View signals
    # ------------------------------------------------------------------
    def _on_load_images(self):
        try:
            squares_x = int(self._view.get_squares_x())
            squares_y = int(self._view.get_squares_y())
            square_length = float(self._view.get_square_length())
            marker_length = float(self._view.get_marker_length())
        except ValueError:
            self._view.show_error("Error",
                "Invalid numeric input. Please enter integer values for number of squares "
                "and numeric values for lengths.")
            return

        if marker_length >= square_length:
            self._view.show_error("Error", "Marker Length must be smaller than Square Length.")
            return

        if not self._model.initialize_board(squares_x, squares_y, square_length, marker_length,
                                             self._view.get_dict_name(), self._view.get_is_legacy()):
            return  # model already emitted error_occurred

        file_dialog = QFileDialog(self._view, "Select Image Files")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not file_dialog.exec_():
            return
        selected_files = file_dialog.selectedFiles()
        if not selected_files:
            return

        self._model.load_images(selected_files)

    def _on_capture_image(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self._view.show_error("Capture Error",
                    "Unable to open the local camera. Please check the connection and try again.")
                return
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                self._view.show_error("Capture Error", "Failed to grab a frame from the camera.")
                return

            images_dir = os.path.join(self._project_root(), "images")
            os.makedirs(images_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(images_dir, f"capture_{timestamp}.png")
            cv2.imwrite(out_path, frame)
            print(f"Capture saved to {out_path}")

            self._view.show_image(frame, panel=0, scale_already_applied=False)
        except Exception as e:
            tb = traceback.format_exc()
            self._view.show_error("Capture Error",
                                  f"An error occurred while capturing:\n{e}\n\nTraceback:\n{tb}")

    def _on_run_calibration(self):
        self._model.run_calibration()

    # ------------------------------------------------------------------
    # Recognition: estimate single ArUco marker 6D poses
    # ------------------------------------------------------------------
    def _project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _on_recog_calib_browse(self):
        """Let the user pick a calibration result JSON for recognition."""
        start_dir = self._model.last_image_dir or os.path.join(self._project_root(), "images")
        file_path, _ = QFileDialog.getOpenFileName(
            self._view, "Select Calibration JSON", start_dir,
            "JSON files (*.json)")
        if file_path:
            self._view.set_recog_calib_json(os.path.abspath(file_path))

    def _load_intrinsics_from_json(self, path=None):
        """Read camera_matrix / dist_coeffs from a calibration JSON file.

        If ``path`` is None, defaults to the project images/calibration.json.
        Returns (camera_matrix, dist_coeffs) or (None, None)."""
        if path is None:
            path = os.path.join(self._project_root(), "images", "calibration.json")
        if not os.path.isfile(path):
            return None, None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            intr = data["intrinsics"]
            dist_l = data["lensDistortions"]
            cam = np.array([
                [intr["focalLengthX"], 0.0, intr["principalPointX"]],
                [0.0, intr["focalLengthY"], intr["principalPointY"]],
                [0.0, 0.0, 1.0]], dtype=np.float64)
            k1 = dist_l["radial"]["k1"]
            k2 = dist_l["radial"]["k2"]
            k3 = dist_l["radial"].get("k3", 0.0)
            p1 = dist_l["tangential"]["p1"]
            p2 = dist_l["tangential"]["p2"]
            dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
            print(f"Loaded intrinsics from {path}")
            return cam, dist
        except Exception as e:
            print(f"Warning: failed to load intrinsics from {path}: {e}")
            return None, None

    def _get_recognition_dictionary(self):
        """Build the dictionary selected in the Recognition Settings group."""
        try:
            return self._model._get_aruco_dictionary(self._view.get_recog_dict_name())
        except Exception as e:
            print(f"Warning: failed to create dictionary for recognition: {e}")
            return None

    def _save_recognition_json(self, image_path, marker_length, dict_name, source, per_marker):
        """Save recognition results to <project>/results/<image_stem>.json. Returns path or None."""
        try:
            results_dir = os.path.join(self._project_root(), "results")
            os.makedirs(results_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(results_dir, stem + ".json")
            payload = {
                "sourceImage": os.path.abspath(image_path),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "arucoDictionary": dict_name,
                "markerLength": marker_length,
                "markerLengthUnit": "m",
                "markerLengthMm": marker_length * 1000.0,
                "intrinsicsSource": source,
                "markers": per_marker,
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=4)
            print(f"Recognition JSON saved to {out_path}")
            return out_path
        except Exception as e:
            print(f"Warning: failed to save recognition JSON: {e}")
            return None

    def _on_recognize(self):
        try:
            start_dir = self._model.last_image_dir or os.path.join(self._project_root(), "images")
            file_path, _ = QFileDialog.getOpenFileName(
                self._view, "Select Image to Recognize", start_dir,
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
            if not file_path:
                return

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img is None:
                self._view.show_error("Recognize", f"Could not read image:\n{file_path}")
                return
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cam = None
            dist = None
            source = ""
            calib_json = self._view.get_recog_calib_json()
            if calib_json:
                cam, dist = self._load_intrinsics_from_json(calib_json)
                source = calib_json
            if cam is None or dist is None:
                cam = self._model.camera_matrix
                dist = self._model.dist_coeffs
                source = "calibration_run"
            if cam is None or dist is None:
                cam, dist = self._load_intrinsics_from_json()
                source = "images/calibration.json"
            if cam is None or dist is None:
                self._view.show_error(
                    "Recognize",
                    "No camera intrinsics available.\n\n"
                    "Please select a calibration JSON, run calibration first, or place "
                    "a calibration.json in the project images/ directory.")
                return
            cam = np.asarray(cam, dtype=np.float64).reshape(3, 3)
            dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

            dict_obj = self._get_recognition_dictionary()
            if dict_obj is None:
                self._view.show_error("Recognize", "Could not create the ArUco dictionary.")
                return
            try:
                dict_obj.setLegacyPattern(self._view.get_recog_is_legacy())
            except AttributeError:
                pass

            corners, ids, _ = cv2.aruco.detectMarkers(gray, dict_obj)
            if ids is None or len(ids) == 0:
                self._view.clear_recognition_table()
                self._view.show_warning("Recognize", "No ArUco markers detected in the image.")
                return

            if not (0 <= cam[0, 2] <= img.shape[1] and 0 <= cam[1, 2] <= img.shape[0]):
                print("Warning: principal point is outside the image; intrinsics may come "
                      "from a different resolution.")

            try:
                marker_length_mm = float(self._view.get_recog_marker_length_mm())
            except (TypeError, ValueError):
                marker_length_mm = None
            if not marker_length_mm or marker_length_mm <= 0:
                self._view.show_error("Recognize", "Invalid Marker Length (mm) value.")
                return
            marker_length = marker_length_mm / 1000.0
            dict_name = self._view.get_recog_dict_name()

            half = marker_length / 2.0
            obj_pts = np.array([
                [-half,  half, 0.0],
                [ half,  half, 0.0],
                [ half, -half, 0.0],
                [-half, -half, 0.0]], dtype=np.float64)

            annotated = img.copy()
            rows = []
            per_marker = []
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                image_pts = marker_corners.reshape(-1, 2).astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj_pts, image_pts, cam, dist)
                if not ok:
                    continue
                cv2.drawFrameAxes(annotated, cam, dist, rvec, tvec, marker_length * 0.5, 3)

                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam, dist)
                reproj = float(np.mean(np.linalg.norm(
                    proj_pts.reshape(-1, 2) - image_pts, axis=1)))

                rvec_f = rvec.flatten()
                tvec_f = tvec.flatten()
                tvec_mm = tvec_f * 1000.0

                rows.append([
                    int(marker_id),
                    f"({rvec_f[0]:.4f}, {rvec_f[1]:.4f}, {rvec_f[2]:.4f})",
                    f"({tvec_f[0]:.4f}, {tvec_f[1]:.4f}, {tvec_f[2]:.4f})",
                    f"({tvec_mm[0]:.1f}, {tvec_mm[1]:.1f}, {tvec_mm[2]:.1f})",
                    f"{reproj:.3f}",
                ])
                per_marker.append({
                    "id": int(marker_id),
                    "rvec": [float(x) for x in rvec_f],
                    "tvec_m": [float(x) for x in tvec_f],
                    "tvec_mm": [float(x) for x in tvec_mm],
                    "reprojectionErrorPx": reproj,
                })

            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

            self._view.show_image(annotated, panel=0, scale_already_applied=False)
            self._view.show_recognition_table(rows)
            self._save_recognition_json(file_path, marker_length, dict_name, source, per_marker)

        except Exception as e:
            tb = traceback.format_exc()
            self._view.show_error("Recognition Error",
                                  f"An error occurred during recognition:\n{e}\n\nTraceback:\n{tb}")

    def _on_save(self):
        self._model.save_calibration()

    def _on_clear_all(self):
        self._model.clear_all()
        self._view.clear_image_list()
        self._view.clear_results()
        self._view.clear_image_panels()

    def _on_clear_selected(self):
        rows = self._view.get_selected_rows()
        if not rows:
            print("No image selected for removal.")
            self._view.show_error("No Image Selected", "No image selected for removal.")
            return
        self._model.remove_images(rows)
        self._view.clear_image_list()
        for name in self._model.image_names:
            self._view.append_image_item(name)
        self._view.clear_image_panels()
        self._render_all_corners_plot()

    def _on_image_selected(self, index):
        if index < 0 or index >= len(self._model.loaded_images):
            return
        try:
            img = self._model.loaded_images[index]
            if img is None:
                return

            orig_h, orig_w = img.shape[:2]
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            target_width = max(1, self._view.get_panel_size(0)[0])

            if orig_w != target_width:
                scale = target_width / float(orig_w)
                target_height = int(round(orig_h * scale))
                scaled_img = cv2.resize(display_img, (target_width, target_height),
                                        interpolation=cv2.INTER_LINEAR)
            else:
                scale = 1.0
                target_height = orig_h  # bug fix: always define target_height
                scaled_img = display_img.copy()

            # Draw detected ChArUco corners
            if index < len(self._model.detected_corners):
                charuco_corners, _ = self._model.detected_corners[index]
                if charuco_corners is not None and len(charuco_corners) > 0:
                    for pt in charuco_corners:
                        if hasattr(pt, "shape") and pt.shape[-1] == 2:
                            x_f = float(pt[0][0]) if pt.ndim == 2 else float(pt[0])
                            y_f = float(pt[0][1]) if pt.ndim == 2 else float(pt[1])
                        else:
                            try:
                                x_f, y_f = float(pt[0][0]), float(pt[0][1])
                            except Exception:
                                continue
                        cv2.circle(scaled_img,
                                   (int(round(x_f * scale)), int(round(y_f * scale))),
                                   4, (255, 0, 0), -1)

            self._view.show_image(scaled_img, panel=0, scale_already_applied=True)

            # Show undistorted image if calibration is available
            if self._model.camera_matrix is not None and self._model.dist_coeffs is not None:
                try:
                    undistorted = cv2.undistort(img, self._model.camera_matrix, self._model.dist_coeffs)
                    scaled_undist = cv2.resize(undistorted, (target_width, target_height),
                                               interpolation=cv2.INTER_LINEAR)
                    self._view.show_image(cv2.cvtColor(scaled_undist, cv2.COLOR_GRAY2BGR),
                                          panel=1, scale_already_applied=True)
                except Exception as e:
                    print(f"Undistortion failed for image {index}: {e}")

        except Exception as e:
            self._view.show_warning("Display Error", f"Failed to display image:\n{e}")

    # ------------------------------------------------------------------
    # Slots for Model signals
    # ------------------------------------------------------------------
    def _on_images_loaded(self, filenames):
        self._view.clear_image_list()
        for name in filenames:
            self._view.append_image_item(name)
        self._render_all_corners_plot()

    def _on_calibration_done(self, results):
        self._view.show_results(results)
        rms = self._model.reprojection_error
        n = len(self._model.rvecs) if self._model.rvecs is not None else 0
        self._view.show_info("Calibration Complete",
            f"Calibration successful!\n\nReprojection Error: {rms:.3f}\nImages used: {n}")

    def _on_save_done(self, path):
        self._view.set_log_save_dir(os.path.dirname(path))
        # Point recognition at the newly saved calibration JSON.
        self._view.set_recog_calib_json(os.path.abspath(path))
        self._view.show_info("Save Calibration", f"Calibration saved successfully to:\n{path}")

    # ------------------------------------------------------------------
    # Matplotlib scatter plot of all detected corners
    # ------------------------------------------------------------------
    def _render_all_corners_plot(self):
        try:
            if not self._model.loaded_images or not self._model.detected_corners:
                return

            h, w = self._model.loaded_images[0].shape[:2]
            gv_w, gv_h = self._view.get_panel_size(1)
            gv_w = max(200, gv_w)
            gv_h = max(200, gv_h)

            dpi = 100
            fig, ax = plt.subplots(figsize=(gv_w / dpi, gv_h / dpi), dpi=dpi)
            ax.set_facecolor("white")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.add_patch(plt.Rectangle((0, 0), w, h, linewidth=2,
                                        edgecolor="black", facecolor="none"))

            cmap = plt.get_cmap("tab10", len(self._model.detected_corners))
            for i, (corners, _) in enumerate(self._model.detected_corners):
                if corners is None or len(corners) == 0:
                    continue
                pts = corners.reshape(-1, 2)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, color=cmap(i), label=f"Image {i + 1}")

            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout(pad=0.0)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf_w, buf_h = canvas.get_width_height()
            image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(buf_h, buf_w, 4)[..., :3]
            plt.close(fig)

            self._view.show_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                  panel=1, scale_already_applied=True)

        except Exception as e:
            self._view.show_error("Corner Plot Error", f"Failed to plot detected corners:\n{e}")
            traceback.print_exc()
