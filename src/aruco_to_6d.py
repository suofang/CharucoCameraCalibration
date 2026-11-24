"""Standalone ArUco -> 6D pose (position + orientation) recognition tool.

This module detects ArUco markers in an image and estimates each marker's full
6-DoF pose (translation + rotation) relative to the camera. It is independent of
the PyQt calibration GUI and can be imported or run directly.

The recognition logic is encapsulated in the :class:`ArucoTo6D` class. All
tunable inputs (camera intrinsics, ArUco parameters, image options) are passed
to the constructor.

Usage
-----
    # Run the built-in test on a local image:
    python aruco_to_6d.py

    # Or use as a library:
    from aruco_to_6d import ArucoTo6D
    detector = ArucoTo6D(
        camera_matrix=...,
        dist_coeffs=...,
        dict_name="DICT_6X6_250",
        marker_length_m=0.040,
    )
    results = detector.detect("path/to/image.jpg")
    for m in results:
        print(m["id"], m["pose6d"])

Dependencies (install in the same environment, e.g. the "aruco" conda env):
    pip install opencv-contrib-python==4.12.0.88 numpy

Coordinate / convention notes
-----------------------------
* Translation ``tvec`` is expressed in the SAME unit as ``marker_length_m``
  (default: meters).
* Rotation is returned BOTH as an OpenCV rotation vector (``rvec``) and as
  Euler angles in degrees (``euler_deg = [rx, ry, rz]``) using the intrinsic
  XYZ (roll-pitch-yaw) convention of ``R = Rx * Ry * Rz``.
* The object frame origin is the marker center, with the marker lying on the
  Z=0 plane (normal points toward +Z, out of the printed face).
* Results are returned sorted by ascending marker id.
"""

import os
import json
import datetime

import cv2
import numpy as np


# ======================================================================
# Default parameter values (used by the built-in test / demo).
# Edit these or pass your own values to the ArucoTo6D constructor.
# ======================================================================

# --- 0. Calibration JSON source of truth -----------------------------
# The calibration JSON is embedded here verbatim (the ``data`` per-image
# corner section is omitted because it is not needed for pose estimation).
# Every other global below is derived from this dict, so editing this literal
# is enough to re-point every consumer of the module globals.
CALIBRATION_JSON = {
    "intrinsics": {
        "focalLengthX": 305.8411078996921,
        "focalLengthY": 305.97810755627074,
        "principalPointX": 320.7003829948845,
        "principalPointY": 203.67339686911853,
    },
    "lensDistortions": {
        "radial": {
            "k1": -0.01751871431663325,
            "k2": -0.0007132983953582356,
            "k3": 0.01085962519418457,
        },
        "tangential": {
            "p1": -7.92705492499948e-05,
            "p2": -0.00021055836852920917,
        },
    },
    "calibrationSummary": {
        "rmsError": 0.148142149786095,
        "numImagesUsed": 35,
    },
}

# --- 1. Camera calibration parameters (intrinsics) --------------------
# Derived from CALIBRATION_JSON["intrinsics"].
_intr = CALIBRATION_JSON["intrinsics"]
_fx = float(_intr["focalLengthX"])
_fy = float(_intr["focalLengthY"])
_cx = float(_intr["principalPointX"])
_cy = float(_intr["principalPointY"])

# Camera matrix 3x3:
#   [[fx,  0, cx],
#    [ 0, fy, cy],
#    [ 0,  0,  1]]
DEFAULT_CAMERA_MATRIX = np.array([
    [_fx, 0.0, _cx],
    [0.0, _fy, _cy],
    [0.0, 0.0, 1.0]], dtype=np.float64)

# Distortion coefficients in order [k1, k2, p1, p2, k3], derived from
# CALIBRATION_JSON["lensDistortions"].
_radial = CALIBRATION_JSON["lensDistortions"]["radial"]
_tangential = CALIBRATION_JSON["lensDistortions"]["tangential"]
DEFAULT_DIST_COEFFS = np.array(
    [float(_radial["k1"]), float(_radial["k2"]),
     float(_tangential["p1"]), float(_tangential["p2"]),
     float(_radial["k3"])], dtype=np.float64)

# Calibration quality summary (derived from CALIBRATION_JSON).
DEFAULT_RMS_ERROR = float(CALIBRATION_JSON["calibrationSummary"]["rmsError"])
DEFAULT_NUM_IMAGES_USED = int(CALIBRATION_JSON["calibrationSummary"]["numImagesUsed"])

# --- 2. ArUco marker parameters --------------------------------------
# Predefined dictionary name, e.g. DICT_6X6_250, DICT_4X4_50, ...
DEFAULT_DICT_NAME = "DICT_6X6_250"
# Whether to enable the legacy dictionary layout.
DEFAULT_IS_LEGACY = False
# Physical side length of the black marker square (defines tvec unit).
DEFAULT_MARKER_LENGTH_M = 0.040     # meters (40 mm)
# Camera-to-marker axis drawing length, as a fraction of the marker length.
DEFAULT_AXIS_FRACTION = 0.5
DEFAULT_AXIS_THICKNESS = 3

# --- 3. Image / processing parameters --------------------------------
# Whether to draw detected markers and axes on the output image.
DEFAULT_DRAW_OVERLAY = True
# Built-in test image.
TEST_IMAGE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images", "g2-20261123", "130-left-g2.jpg",
)

# --- 4. ChArUco board calibration parameters -------------------------
# These describe the printed ChArUco board used for calibration. They mirror
# src/config/config.json. Lengths are in meters.
DEFAULT_BOARD_SQUARES_X = 5
DEFAULT_BOARD_SQUARES_Y = 7
DEFAULT_BOARD_SQUARE_LENGTH_M = 0.025
DEFAULT_BOARD_MARKER_LENGTH_M = 0.018

# Outlier-removal settings (mirror config.json "outlierRemoval").
DEFAULT_OUTLIER_PARAMS = {
    "reprojectionErrorThreshold": 1.0,
    "maxOutlierRemovalIterations": 3,
    "maxNormalizedReprojectionError": 2.0,
    "minImageInlierPct": 0.5,
    "minImageNum": 4,
}

# Built-in calibration test directory (g2 images).
TEST_CALIBRATION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images", "g2-20261123",
)


class ArucoTo6D:
    """Detect ArUco markers and estimate each marker's 6D pose.

    All parameters are optional. Any parameter left as ``None`` falls back to
    the corresponding module-level default (see the ``DEFAULT_*`` constants and
    ``DEFAULT_CAMERA_MATRIX`` / ``DEFAULT_DIST_COEFFS`` at the top of the file).

    Parameters
    ----------
    camera_matrix : ndarray, optional
        3x3 camera intrinsic matrix (default ``DEFAULT_CAMERA_MATRIX``).
    dist_coeffs : ndarray, optional
        Distortion coefficients [k1, k2, p1, p2, k3] (default
        ``DEFAULT_DIST_COEFFS``).
    dict_name : str, optional
        Predefined ArUco dictionary name (default ``DEFAULT_DICT_NAME``).
    marker_length_m : float, optional
        Marker side length in meters (default ``DEFAULT_MARKER_LENGTH_M``). The
        returned translation uses this unit.
    is_legacy : bool, optional
        Legacy dictionary layout flag (default ``DEFAULT_IS_LEGACY``).
    axis_fraction : float, optional
        Drawn axis length as a fraction of the marker length (default
        ``DEFAULT_AXIS_FRACTION``).
    axis_thickness : int, optional
        Drawn axis line thickness (default ``DEFAULT_AXIS_THICKNESS``).
    draw_overlay : bool, optional
        Whether to draw detected markers and axes (default
        ``DEFAULT_DRAW_OVERLAY``).
    output_image : str or None, optional
        Path to save the annotated image; ``None`` disables saving.
    output_json : str or None, optional
        Path to save the JSON result; ``None`` disables saving.
    """

    def __init__(self,
                 camera_matrix=None,
                 dist_coeffs=None,
                 dict_name=None,
                 marker_length_m=None,
                 is_legacy=None,
                 axis_fraction=None,
                 axis_thickness=None,
                 draw_overlay=None,
                 output_image=None,
                 output_json=None,
                 board_squares_x=None,
                 board_squares_y=None,
                 board_square_length_m=None,
                 board_marker_length_m=None,
                 outlier_params=None):
        if camera_matrix is None:
            camera_matrix = DEFAULT_CAMERA_MATRIX
        if dist_coeffs is None:
            dist_coeffs = DEFAULT_DIST_COEFFS
        if dict_name is None:
            dict_name = DEFAULT_DICT_NAME
        if marker_length_m is None:
            marker_length_m = DEFAULT_MARKER_LENGTH_M
        if is_legacy is None:
            is_legacy = DEFAULT_IS_LEGACY
        if axis_fraction is None:
            axis_fraction = DEFAULT_AXIS_FRACTION
        if axis_thickness is None:
            axis_thickness = DEFAULT_AXIS_THICKNESS
        if draw_overlay is None:
            draw_overlay = DEFAULT_DRAW_OVERLAY
        if board_squares_x is None:
            board_squares_x = DEFAULT_BOARD_SQUARES_X
        if board_squares_y is None:
            board_squares_y = DEFAULT_BOARD_SQUARES_Y
        if board_square_length_m is None:
            board_square_length_m = DEFAULT_BOARD_SQUARE_LENGTH_M
        if board_marker_length_m is None:
            board_marker_length_m = DEFAULT_BOARD_MARKER_LENGTH_M
        outlier_params = dict(DEFAULT_OUTLIER_PARAMS if outlier_params is None
                              else outlier_params)

        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        self.dict_name = dict_name
        self.marker_length_m = float(marker_length_m)
        self.is_legacy = bool(is_legacy)
        self.axis_fraction = float(axis_fraction)
        self.axis_thickness = int(axis_thickness)
        self.draw_overlay = bool(draw_overlay)
        self.output_image = output_image
        self.output_json = output_json

        # ChArUco board (calibration) parameters.
        self.board_squares_x = int(board_squares_x)
        self.board_squares_y = int(board_squares_y)
        self.board_square_length_m = float(board_square_length_m)
        self.board_marker_length_m = float(board_marker_length_m)
        self.outlier_params = outlier_params

        self._dictionary = self._build_dictionary(dict_name, is_legacy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self,
               image_path,
               marker_length_m=None,
               dict_name=None,
               is_legacy=None,
               camera_matrix=None,
               dist_coeffs=None,
               axis_fraction=None,
               axis_thickness=None,
               draw_overlay=None,
               return_image=False):
        """Detect ArUco markers and estimate each marker's 6D pose.

        All ArUco / camera parameters are optional per-call overrides. Any
        parameter left as ``None`` falls back to the value stored on the
        instance (which itself defaults to the module-level globals).

        Parameters
        ----------
        image_path : str
            Path to the input image.
        marker_length_m : float, optional
            Marker side length in meters. Overrides the instance value.
        dict_name : str, optional
            Predefined ArUco dictionary name. Overrides the instance value.
        is_legacy : bool, optional
            Legacy dictionary layout flag. Overrides the instance value.
        camera_matrix : ndarray, optional
            3x3 camera intrinsic matrix. Overrides the instance value.
        dist_coeffs : ndarray, optional
            Distortion coefficients. Overrides the instance value.
        axis_fraction : float, optional
            Drawn axis length fraction. Overrides the instance value.
        axis_thickness : int, optional
            Drawn axis line thickness. Overrides the instance value.
        draw_overlay : bool, optional
            Whether to draw detected markers and axes. Overrides the instance
            value.
        return_image : bool, optional
            If True, also return the annotated BGR image.

        Returns
        -------
        results : list[dict]  (sorted by ascending marker id)
            Each dict contains:
                id                    : int   marker id
                pose6d                : [x, y, z, rx, ry, rz]
                                        x,y,z = tvec (in marker_length unit);
                                        rx,ry,rz = Euler angles in degrees
                tvec                  : [x, y, z]  (marker_length unit)
                tvec_mm               : [x, y, z]  (millimeters)
                rvec                  : [rx, ry, rz]  OpenCV rotation vector
                euler_deg             : [rx, ry, rz]  (same as pose6d[3:])
                reprojection_error_px : float   mean per-corner reprojection error
        annotated_image : ndarray (only if return_image=True)
            BGR image with detected markers and pose axes drawn.
        """
        # --- Resolve effective parameters (per-call overrides) -------
        if marker_length_m is None:
            marker_length_m = self.marker_length_m
        if dict_name is None:
            dict_name = self.dict_name
        if is_legacy is None:
            is_legacy = self.is_legacy
        if camera_matrix is None:
            camera_matrix = self.camera_matrix
        if dist_coeffs is None:
            dist_coeffs = self.dist_coeffs
        if axis_fraction is None:
            axis_fraction = self.axis_fraction
        if axis_thickness is None:
            axis_thickness = self.axis_thickness
        if draw_overlay is None:
            draw_overlay = self.draw_overlay

        cam = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        marker_length_m = float(marker_length_m)
        axis_fraction = float(axis_fraction)
        axis_thickness = int(axis_thickness)

        # Rebuild the dictionary only when the requested one differs.
        if dict_name == self.dict_name and is_legacy == self.is_legacy:
            dictionary = self._dictionary
        else:
            dictionary = self._build_dictionary(dict_name, is_legacy)

        # --- Load image ----------------------------------------------
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- Detect markers ------------------------------------------
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

        annotated = image.copy()
        results = []
        if ids is not None and len(ids) > 0:
            obj_pts = self._marker_object_points(marker_length_m)
            axis_len = marker_length_m * axis_fraction

            for marker_corners, marker_id in zip(corners, ids.flatten()):
                image_pts = marker_corners.reshape(-1, 2).astype(np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj_pts, image_pts, cam, dist)
                if not ok:
                    continue

                # Reprojection error (mean per-corner distance in pixels).
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam, dist)
                reproj = float(np.mean(np.linalg.norm(
                    proj_pts.reshape(-1, 2) - image_pts, axis=1)))

                rvec_f = rvec.flatten()
                tvec_f = tvec.flatten()
                tvec_mm = tvec_f * 1000.0
                euler = self._rotation_matrix_to_euler_deg(rvec)

                results.append({
                    "id": int(marker_id),
                    "pose6d": [float(tvec_f[0]), float(tvec_f[1]), float(tvec_f[2]),
                               float(euler[0]), float(euler[1]), float(euler[2])],
                    "tvec": [float(x) for x in tvec_f],
                    "tvec_mm": [float(x) for x in tvec_mm],
                    "rvec": [float(x) for x in rvec_f],
                    "euler_deg": [float(x) for x in euler],
                    "reprojection_error_px": reproj,
                })

                if draw_overlay:
                    cv2.drawFrameAxes(annotated, cam, dist, rvec, tvec,
                                      axis_len, axis_thickness)

            if draw_overlay:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        # Sort by ascending marker id (stable order for downstream consumers).
        results.sort(key=lambda m: m["id"])

        if self.output_image:
            self._ensure_parent(self.output_image)
            cv2.imwrite(self.output_image, annotated)
            print(f"[aruco_to_6d] 标注结果图已保存到 {self.output_image}")

        if self.output_json:
            self._save_json(self.output_json, image_path, dict_name,
                            marker_length_m, results)

        if return_image:
            return results, annotated
        return results

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self,
                  image_dir,
                  output_json=None,
                  board_squares_x=None,
                  board_squares_y=None,
                  board_square_length_m=None,
                  board_marker_length_m=None,
                  dict_name=None,
                  is_legacy=None,
                  outlier_params=None):
        """Calibrate the camera from a directory of ChArUco board images.

        Mirrors the "Run Calibration" button of the GUI: it loads every image
        in ``image_dir``, detects ChArUco corners, runs
        ``cv2.aruco.calibrateCameraCharucoExtended`` with optional iterative
        outlier removal, writes a JSON result file and logs progress.

        Any parameter left as ``None`` falls back to the value stored on the
        instance (which itself defaults to the module-level globals).

        Parameters
        ----------
        image_dir : str
            Directory containing the calibration images.
        output_json : str, optional
            Path of the JSON result file. Defaults to ``image_dir/calibration.json``.
        board_squares_x, board_squares_y : int, optional
            Number of squares along the X / Y axis.
        board_square_length_m, board_marker_length_m : float, optional
            Chessboard square / marker side length in meters.
        dict_name : str, optional
            Predefined ArUco dictionary name.
        is_legacy : bool, optional
            Legacy dictionary layout flag.
        outlier_params : dict, optional
            Outlier-removal settings (see ``DEFAULT_OUTLIER_PARAMS``).

        Returns
        -------
        dict
            Calibration result: ``fx``, ``fy``, ``ppx``, ``ppy``, ``K1``,
            ``K2``, ``K3``, ``P1``, ``P2``, plus ``reprojectionError``,
            ``imageSize``, ``numImages`` and ``numImagesUsed``.
        """
        def log(msg):
            print(f"[aruco_to_6d] {msg}")

        # --- Resolve effective parameters ----------------------------
        if board_squares_x is None:
            board_squares_x = self.board_squares_x
        if board_squares_y is None:
            board_squares_y = self.board_squares_y
        if board_square_length_m is None:
            board_square_length_m = self.board_square_length_m
        if board_marker_length_m is None:
            board_marker_length_m = self.board_marker_length_m
        if dict_name is None:
            dict_name = self.dict_name
        if is_legacy is None:
            is_legacy = self.is_legacy
        outlier_params = dict(self.outlier_params if outlier_params is None
                              else outlier_params)

        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"Image directory not found: {image_dir}")
        if output_json is None:
            output_json = os.path.join(image_dir, "calibration.json")

        # --- Build board and dictionary -----------------------------
        dictionary = self._build_dictionary(dict_name, is_legacy)
        board = self._create_charuco_board(
            board_squares_x, board_squares_y, board_square_length_m,
            board_marker_length_m, dictionary)
        try:
            board.setLegacyPattern(bool(is_legacy))
        except AttributeError:
            pass

        image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        image_paths = sorted(
            os.path.join(image_dir, name) for name in os.listdir(image_dir)
            if name.lower().endswith(image_exts))
        if not image_paths:
            raise FileNotFoundError(f"目录中未找到标定图片: {image_dir}")

        log(f"开始标定，目录: '{image_dir}'（共 {len(image_paths)} 张图片）")
        log(f"标定板 {board_squares_x}x{board_squares_y} | 方格={board_square_length_m * 1000:.1f} mm"
            f" | 标记={board_marker_length_m * 1000:.1f} mm | 字典={dict_name}"
            f" | 旧版布局={is_legacy}")

        # --- Pass 1: load images and detect ChArUco corners ---------
        all_corners, all_ids, image_size = [], [], None
        for file_path in image_paths:
            name = os.path.basename(file_path)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                log(f"警告: 无法读取 {name}，已跳过。")
                continue
            if image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)

            try:
                corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
            except Exception as e:
                log(f"{name}: ArUco 标记检测出错: {e}")
                continue

            if ids is None or len(ids) == 0:
                log(f"{name}: 未检测到标记，已跳过。")
                continue

            try:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners, markerIds=ids,
                    image=img, board=board)
            except Exception as e:
                log(f"{name}: ChArUco 角点插值出错: {e}")
                continue

            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                log(f"{name}: 检测到 {len(charuco_ids)} 个 ChArUco 角点。")
            else:
                log(f"{name}: ChArUco 角点不足，已跳过。")

        if not all_corners:
            raise RuntimeError("所有图片中均未检测到有效的 ChArUco 角点。")
        if len(all_corners) < outlier_params["minImageNum"]:
            log(f"警告: 仅有 {len(all_corners)} 张有效图片，"
                f"minImageNum 为 {outlier_params['minImageNum']}。")

        # --- Pass 2: calibrate (+ iterative outlier removal) --------
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-9)
        used = len(all_corners)
        log(f"正在对 {used} 张有效图片运行 calibrateCameraCharucoExtended ...")
        (retval, cam_mat, dist_coeffs, _rvecs, _tvecs,
         _std_int, _std_ext, per_view) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners, charucoIds=all_ids, board=board,
            imageSize=image_size, cameraMatrix=None, distCoeffs=None,
            criteria=criteria)

        if retval > outlier_params["reprojectionErrorThreshold"]:
            log(f"重投影误差 ({retval:.4f}) 超过阈值 "
                f"({outlier_params['reprojectionErrorThreshold']:.4f})，"
                f"开始剔除离群图片 ...")
            keep = np.arange(used)
            for _ in range(outlier_params["maxOutlierRemovalIterations"]):
                per_view_flat = np.asarray(per_view).reshape(-1)
                median = np.median(per_view_flat)
                mad = np.median(np.abs(per_view_flat - median))
                if mad == 0:
                    break
                norm = np.abs(per_view_flat - median) / (1.428 * mad)
                keep = keep[np.nonzero(
                    norm < outlier_params["maxNormalizedReprojectionError"])[0]]

                if (keep.shape[0] == used
                        or keep.shape[0] < outlier_params["minImageNum"]
                        or keep.shape[0] <= outlier_params["minImageInlierPct"] * used):
                    break
                used = keep.shape[0]
                keep_corners = [all_corners[i] for i in keep]
                keep_ids = [all_ids[i] for i in keep]
                log(f"使用 {used} 张有效图片重新标定 ...")
                (retval, cam_mat, dist_coeffs, _rvecs, _tvecs,
                 _std_int, _std_ext, per_view) = cv2.aruco.calibrateCameraCharucoExtended(
                    charucoCorners=keep_corners, charucoIds=keep_ids, board=board,
                    imageSize=image_size, cameraMatrix=None, distCoeffs=None,
                    criteria=criteria)

        # --- Build and save the result ------------------------------
        dist_flat = np.asarray(dist_coeffs).reshape(-1)
        k = list(dist_flat) + [0.0] * 5  # pad to at least 5 coefficients
        result = {
            "fx": float(cam_mat[0, 0]),
            "fy": float(cam_mat[1, 1]),
            "ppx": float(cam_mat[0, 2]),
            "ppy": float(cam_mat[1, 2]),
            "K1": float(k[0]),
            "K2": float(k[1]),
            "K3": float(k[4]),
            "P1": float(k[2]),
            "P2": float(k[3]),
            "reprojectionError": float(retval),
            "imageSize": list(image_size),
            "numImages": len(image_paths),
            "numImagesUsed": int(used),
            "arucoDictionary": dict_name,
            "board": {
                "nSquaresX": board_squares_x,
                "nSquaresY": board_squares_y,
                "sqrLength": float(board_square_length_m),
                "markerLength": float(board_marker_length_m),
                "dictName": dict_name,
                "isLegacy": bool(is_legacy),
            },
            "cameraMatrix": np.asarray(cam_mat).tolist(),
            "distCoeffs": dist_flat.tolist(),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        self._ensure_parent(output_json)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=4)

        log("标定完成。")
        log(f"重投影误差: {retval:.4f}")
        log(f"相机内参矩阵:\n{cam_mat}")
        log(f"畸变系数:\n{dist_coeffs}")
        log(f"使用 {used}/{len(image_paths)} 张图片。")
        log(f"标定结果已保存到 JSON: {output_json}")

        # --- Quality evaluation ------------------------------------
        self.evaluate_calibration(result, per_view)

        return result

    def evaluate_calibration(self, result, per_view=None):
        """Evaluate a calibration result, log it and suggest improvements.

        Grades the RMS reprojection error, the inlier ratio, image coverage
        and (when available) per-view error spread, then emits concrete,
        actionable improvement suggestions for anything that looks weak.

        Parameters
        ----------
        result : dict
            A calibration result as returned by :meth:`calibrate`.
        per_view : array-like, optional
            Per-view reprojection errors (pixels). When provided, per-image
            consistency is also evaluated.

        Returns
        -------
        dict
            ``{"grade": str, "score": int, "rms": float, "issues": [...],
            "suggestions": [...]}``.
        """
        def log(msg):
            print(f"[aruco_to_6d] {msg}")

        rms = float(result.get("reprojectionError", float("inf")))
        num_used = int(result.get("numImagesUsed", 0))
        num_total = int(result.get("numImages", num_used))
        cam_mat = np.asarray(result.get("cameraMatrix"), dtype=np.float64)
        image_size = result.get("imageSize") or [0, 0]
        img_w, img_h = int(image_size[0]), int(image_size[1])

        issues = []
        suggestions = []

        # --- 1. RMS reprojection error ------------------------------
        if rms <= 0.3:
            rms_level = "优秀"
        elif rms <= 0.5:
            rms_level = "良好"
        elif rms <= 1.0:
            rms_level = "可接受"
        else:
            rms_level = "较差"

        log("=" * 60)
        log("标定质量评价")
        log("=" * 60)
        log(f"重投影误差 (RMS)        : {rms:.4f} px  -> {rms_level}")
        if rms > 1.0:
            issues.append(f"重投影误差过大 ({rms:.3f} px > 1.0 px)")
            suggestions.append(
                "重投影误差偏大：检查标定板是否平整、打印尺寸是否准确，"
                "避免运动模糊与过曝/欠曝，删除明显失焦或倾斜过大的图片后重新标定。")
        elif rms > 0.5:
            issues.append(f"重投影误差偏高 ({rms:.3f} px)")
            suggestions.append(
                "误差偏高：增加更多不同姿态（含倾斜、远近变化）的图片，"
                "确保标定板角点检测完整。")

        # --- 2. Inlier image ratio ----------------------------------
        inlier_ratio = (num_used / num_total) if num_total else 0.0
        log(f"有效图片数              : {num_used}/{num_total} "
            f"（{inlier_ratio * 100:.1f}%）")
        if num_used < 10:
            issues.append(f"有效图片数量偏少 ({num_used} 张)")
            suggestions.append(
                "有效图片数量偏少：建议至少 15~20 张覆盖全视场的图片，"
                "以提高内参稳定性。")
        if num_total and inlier_ratio < 0.6:
            issues.append(f"离群/无效图片比例较高 (仅 {inlier_ratio * 100:.1f}% 有效)")
            suggestions.append(
                "较多图片被剔除：检查这些图片中 ChArUco 角点是否被完整检测，"
                "避免标定板越界、模糊或反光。")

        # --- 3. Focal length sanity --------------------------------
        if cam_mat.size == 9 and img_w and img_h:
            fx, fy = cam_mat[0, 0], cam_mat[1, 1]
            ratio = fx / fy if fy else 0.0
            log(f"焦距                    : fx={fx:.2f}, fy={fy:.2f} "
                f"(fx/fy={ratio:.4f})")
            if abs(ratio - 1.0) > 0.05:
                issues.append(f"fx 与 fy 差异较大 (fx/fy={ratio:.4f})")
                suggestions.append(
                    "fx 与 fy 差异偏大：通常由图片覆盖区域不足或标定板姿态单一引起，"
                    "应让标定板出现在图像各个区域（尤其四角）。")

        # --- 4. Image coverage (principal point near center) --------
        if cam_mat.size == 9 and img_w and img_h:
            cx, cy = cam_mat[0, 2], cam_mat[1, 2]
            dx = abs(cx - img_w / 2.0) / (img_w / 2.0)
            dy = abs(cy - img_h / 2.0) / (img_h / 2.0)
            log(f"主点                    : ({cx:.2f}, {cy:.2f})，图像中心 "
                f"({img_w / 2.0:.1f}, {img_h / 2.0:.1f})")
            if dx > 0.15 or dy > 0.15:
                issues.append("主点偏离图像中心较多")
                suggestions.append(
                    "主点明显偏移：确保标定板覆盖图像四周而非集中在中心，"
                    "增加标定板位于画面边缘的图片。")

        # --- 5. Per-view error consistency --------------------------
        if per_view is not None:
            pv = np.asarray(per_view, dtype=np.float64).reshape(-1)
            pv = pv[np.isfinite(pv)]
            if pv.size:
                pv_median = float(np.median(pv))
                pv_max = float(np.max(pv))
                log(f"单张视角误差            : 中位数={pv_median:.3f} px，"
                    f"最大值={pv_max:.3f} px")
                if pv_max > max(3.0 * pv_median, 1.0):
                    issues.append(f"个别视角误差异常偏高 (max={pv_max:.3f} px)")
                    suggestions.append(
                        "存在个别误差异常大的图片：移除这些图片后重新标定，"
                        "它们很可能是失焦或角点检测错误的样本。")

        # --- Overall grade -----------------------------------------
        score = 100
        score -= min(60, int(rms * 40))
        if num_used < 10:
            score -= 20
        if num_total and inlier_ratio < 0.6:
            score -= 10
        score = max(0, min(100, score))
        if score >= 85:
            grade = "A (优秀)"
        elif score >= 70:
            grade = "B (良好)"
        elif score >= 50:
            grade = "C (可接受)"
        else:
            grade = "D (较差)"

        log(f"综合评级                : {grade}  (得分={score})")
        if issues:
            log("发现的问题:")
            for it in issues:
                log(f"  - {it}")
        if suggestions:
            log("改进建议:")
            for s in suggestions:
                log(f"  * {s}")
        if not issues:
            log("标定结果良好，无需明显改进。")
        log("=" * 60)

        return {
            "grade": grade,
            "score": score,
            "rms": rms,
            "issues": issues,
            "suggestions": suggestions,
        }

    @staticmethod
    def _create_charuco_board(squares_x, squares_y, square_length,
                              marker_length, aruco_dict):
        """Create a ChArUco board across OpenCV API variants."""
        if hasattr(cv2.aruco, "CharucoBoard_create"):
            try:
                return cv2.aruco.CharucoBoard_create(
                    squares_x, squares_y, square_length, marker_length, aruco_dict)
            except Exception:
                pass
        if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard, "create"):
            try:
                return cv2.aruco.CharucoBoard.create(
                    squares_x, squares_y, square_length, marker_length, aruco_dict)
            except Exception:
                pass
        try:
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_length, marker_length, aruco_dict)
        except Exception as e:
            raise AttributeError(
                f"Unable to create CharucoBoard. Error: {e}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_dictionary(dict_name, is_legacy):
        """Create a predefined ArUco dictionary object.

        ``Dictionary.setLegacyPattern`` is not exposed on every OpenCV build, so
        the legacy flag is applied defensively.
        """
        enum_val = getattr(cv2.aruco, dict_name, None)
        if enum_val is None:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(enum_val)
        try:
            dictionary.setLegacyPattern(bool(is_legacy))
        except AttributeError:
            # Older / alternate builds may place the flag on the detector
            # instead; detection still works with the default layout.
            pass
        return dictionary

    @staticmethod
    def _rotation_matrix_to_euler_deg(rvec):
        """Convert an OpenCV rotation vector to intrinsic XYZ Euler angles (deg).

        Uses R = Rx(rx) * Ry(ry) * Rz(rz):
            ry = atan2(-R[2,0], sqrt(R[0,0]^2 + R[1,0]^2))
            rx = atan2(R[2,1], R[2,2])
            rz = atan2(R[1,0], R[0,0])
        """
        rot, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
        if sy > 1e-6:
            rx = np.arctan2(rot[2, 1], rot[2, 2])
            ry = np.arctan2(-rot[2, 0], sy)
            rz = np.arctan2(rot[1, 0], rot[0, 0])
        else:  # gimbal lock
            rx = np.arctan2(-rot[1, 2], rot[1, 1])
            ry = np.arctan2(-rot[2, 0], sy)
            rz = 0.0
        return np.degrees([rx, ry, rz])

    @staticmethod
    def _marker_object_points(marker_length):
        """3D corner coordinates of a marker centered at origin on the Z=0 plane.

        Corner order matches OpenCV's detectMarkers order:
        top-left, top-right, bottom-right, bottom-left.
        """
        half = marker_length / 2.0
        return np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0]], dtype=np.float64)

    @staticmethod
    def _ensure_parent(path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _save_json(self, path, image_path, dict_name, marker_length_m, results):
        """Write recognition results to a JSON file (schema shared with the GUI)."""
        self._ensure_parent(path)
        payload = {
            "sourceImage": os.path.abspath(image_path),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "arucoDictionary": dict_name,
            "markerLength": float(marker_length_m),
            "markerLengthUnit": "m",
            "markerLengthMm": float(marker_length_m) * 1000.0,
            "intrinsicsSource": "explicit",
            "markers": results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"[aruco_to_6d] 检测结果已保存到 JSON: {path}")


# ======================================================================
# Built-in test / demo entry point
# ======================================================================
def _main():
    detector = ArucoTo6D()

    print(f"[aruco_to_6d] 测试图片 : {TEST_IMAGE}")
    print(f"[aruco_to_6d] 内参     : "
          f"(fx={detector.camera_matrix[0, 0]:.3f}, fy={detector.camera_matrix[1, 1]:.3f})")
    print(f"[aruco_to_6d] 字典     : {detector.dict_name} | 标记边长 = "
          f"{detector.marker_length_m * 1000:.1f} mm | 旧版布局 = {detector.is_legacy}")

    results = detector.detect(TEST_IMAGE)

    if not results:
        print("[aruco_to_6d] 未检测到 ArUco 标记。")
        return

    print(f"[aruco_to_6d] 检测到 {len(results)} 个标记:")
    header = (f"{'ID':>4} | {'x(mm)':>9} {'y(mm)':>9} {'z(mm)':>9} | "
              f"{'rx(度)':>9} {'ry(度)':>9} {'rz(度)':>9} | {'重投影(px)':>10}")
    print(header)
    print("-" * len(header))
    for m in results:
        x, y, z = m["tvec_mm"]
        rx, ry, rz = m["euler_deg"]
        print(f"{m['id']:>4} | {x:>9.1f} {y:>9.1f} {z:>9.1f} | "
              f"{rx:>9.2f} {ry:>9.2f} {rz:>9.2f} | "
              f"{m['reprojection_error_px']:>10.3f}")

    # --- Demo calibration from the built-in test directory ----------
    print()
    print(f"[aruco_to_6d] 标定演示目录 : {TEST_CALIBRATION_DIR}")
    out_json = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "calibration_demo.json")
    try:
        detector.calibrate(TEST_CALIBRATION_DIR, output_json=out_json)
    except Exception as e:
        print(f"[aruco_to_6d] 标定演示失败: {e}")


if __name__ == "__main__":
    _main()
