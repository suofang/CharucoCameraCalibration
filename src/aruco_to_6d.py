"""独立的 ArUco -> 6D 位姿（位置 + 姿态）识别工具。

本模块在图像中检测 ArUco 标记，并估计每个标记相对于相机的完整 6 自由度位姿
（平移 + 旋转）。它不依赖 PyQt 标定界面，既可被导入，也可直接运行。

识别逻辑封装在 :class:`ArucoTo6D` 类中。所有可调输入（相机内参、ArUco 参数、
图像选项）均通过构造函数传入。

使用方法
--------
    # 在本地图片上运行内置测试：
    python aruco_to_6d.py

    # 或作为库使用：
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

依赖（请在同一环境中安装，例如 "aruco" conda 环境）：
    # 同时兼容 opencv-contrib-python 4.12.0.88 与 5.0.0.93，任选其一：
    pip install opencv-contrib-python==4.12.0.88 numpy==2.2.6 --force-reinstall
    pip install opencv-contrib-python==5.0.0.93 numpy --force-reinstall

坐标系 / 约定说明
-----------------
* 平移 ``tvec`` 的单位与 ``marker_length_m`` 相同（默认：米）。
* 旋转同时以 OpenCV 旋转向量（``rvec``）和欧拉角（度）返回
  （``euler_deg = [rx, ry, rz]``），采用内旋 XYZ（横滚-俯仰-偏航）约定
  ``R = Rx * Ry * Rz``。
* 物体坐标系原点为标记中心，标记位于 Z=0 平面（法线指向 +Z，朝向打印面外侧）。
* 结果按标记 id 升序返回。
"""

import os
import json
import time
import datetime
import enum

import cv2
import numpy as np


class MarkerSortOrder(enum.Enum):
    """检测结果排序方式。"""

    ID_ASC = "ID_ASC"           # 按标记 id 升序（默认）
    ID_DESC = "ID_DESC"         # 按标记 id 降序
    LEFT_FIRST = "LEFT_FIRST"   # 从左到右（中心 x 升序）
    RIGHT_FIRST = "RIGHT_FIRST"  # 从右到左（中心 x 降序）
    TOP_FIRST = "TOP_FIRST"     # 从上到下（中心 y 升序）
    BOTTOM = "BOTTOM"           # 从下到上（中心 y 降序）


# ======================================================================
# 默认参数值（供内置测试 / 演示使用）。
# 可直接修改此处，或通过 ArucoTo6D 构造函数传入自定义值。
# ======================================================================

# --- 0. 标定 JSON 数据的唯一来源 -------------------------------------
# 直接复制本脚本 calibrate() 生成的标定 JSON 全文，替换下方字典内容即可。
# 下方所有全局变量（内参 / 畸变 / 字典 / 标记边长等）均从该字典派生，因此
# 只需替换此字面量即可统一改变所有使用模块全局变量的地方。
#
# 结构与界面 Save 按钮保存的 calibration.json 一致（省略 data 字段）：
# {
#     "intrinsics": {
#         "focalLengthX": fx, "focalLengthY": fy,
#         "principalPointX": cx, "principalPointY": cy
#     },
#     "lensDistortions": {
#         "radial": {"k1": k1, "k2": k2, "k3": k3},
#         "tangential": {"p1": p1, "p2": p2}
#     },
#     "calibrationSummary": {"rmsError": rms, "numImagesUsed": n},
#     "imageSize": [w, h],
#     "numImages": total,
#     "arucoDictionary": "DICT_6X6_250",
#     "board": {
#         "nSquaresX": 5, "nSquaresY": 7,
#         "sqrLength": 0.025, "markerLength": 0.018,
#         "dictName": "DICT_6X6_250", "isLegacy": false
#     },
#     "timestamp": "..."
# }
CALIBRATION_JSON = {
    "intrinsics": {
        "focalLengthX": 305.8411078996921,
        "focalLengthY": 305.97810755627074,
        "principalPointX": 320.7003829948845,
        "principalPointY": 203.67339686911853
    },
    "lensDistortions": {
        "radial": {
            "k1": -0.01751871431663325,
            "k2": -0.0007132983953582356,
            "k3": 0.01085962519418457
        },
        "tangential": {
            "p1": -7.92705492499948e-05,
            "p2": -0.00021055836852920917
        }
    },
    "calibrationSummary": {
        "rmsError": 0.148142149786095,
        "numImagesUsed": 35
    },
    "imageSize": [
        640,
        400
    ],
    "numImages": 42,
    "arucoDictionary": "DICT_6X6_250",
    "board": {
        "nSquaresX": 5,
        "nSquaresY": 7,
        "sqrLength": 0.025,
        "markerLength": 0.018,
        "dictName": "DICT_6X6_250",
        "isLegacy": False
    },
    "timestamp": "2026-11-24T15:52:52"
}

# --- 1. 相机标定参数（内参）------------------------------------------
# 从 CALIBRATION_JSON["intrinsics"] 派生 3x3 相机内参矩阵。
_intr = CALIBRATION_JSON.get("intrinsics", {})
DEFAULT_CAMERA_MATRIX = np.array([
    [float(_intr.get("focalLengthX", 1.0)), 0.0,
     float(_intr.get("principalPointX", 0.0))],
    [0.0, float(_intr.get("focalLengthY", 1.0)),
     float(_intr.get("principalPointY", 0.0))],
    [0.0, 0.0, 1.0]], dtype=np.float64)

# --- 1b. 畸变系数（从 CALIBRATION_JSON["lensDistortions"] 派生）------
# 组装为 OpenCV 约定顺序 [k1, k2, p1, p2, k3]。
_dist = CALIBRATION_JSON.get("lensDistortions", {})
_radial = _dist.get("radial", {})
_tangential = _dist.get("tangential", {})
DEFAULT_DIST_COEFFS = np.array(
    [float(_radial.get("k1", 0.0)), float(_radial.get("k2", 0.0)),
     float(_tangential.get("p1", 0.0)), float(_tangential.get("p2", 0.0)),
     float(_radial.get("k3", 0.0))], dtype=np.float64)

# 标定质量摘要（从 CALIBRATION_JSON["calibrationSummary"] 派生）。
_summary = CALIBRATION_JSON.get("calibrationSummary", {})
DEFAULT_RMS_ERROR = (float(_summary["rmsError"])
                     if _summary.get("rmsError") is not None
                     else float("nan"))
DEFAULT_NUM_IMAGES_USED = int(_summary.get("numImagesUsed") or 0)

# 标定时使用的图像尺寸（宽, 高）。内参、畸变系数专门针对该尺寸标定，
# 因此检测时必须用它来校验输入图片分辨率。若 JSON 缺失则为 None（跳过校验）。
_json_size = CALIBRATION_JSON.get("imageSize")
DEFAULT_IMAGE_SIZE = ((int(_json_size[0]), int(_json_size[1]))
                      if _json_size else None)

# --- 2. ArUco 标记参数（从 CALIBRATION_JSON 派生）--------------------
_board = CALIBRATION_JSON.get("board", {})
# 预定义字典名称，例如 DICT_6X6_250、DICT_4X4_50 ...
DEFAULT_DICT_NAME = (CALIBRATION_JSON.get("arucoDictionary")
                     or _board.get("dictName") or "DICT_6X6_250")
# 是否启用旧版（legacy）字典布局。
DEFAULT_IS_LEGACY = bool(_board.get("isLegacy", False))
# 被检测 ArUco 标记的实际边长（决定 tvec 的单位），单位米。
# 注意：这是「被检测的 ArUco 码」的尺寸，独立于标定棋盘中标记的尺寸
# （CALIBRATION_JSON["board"]["markerLength"]），两者不一定相同。
DEFAULT_DETECT_MARKER_LENGTH_M = 0.100  # 米（100 mm）
# 相机到标记的坐标轴绘制长度，以标记边长的比例表示。
DEFAULT_AXIS_FRACTION = 0.5
DEFAULT_AXIS_THICKNESS = 3

# --- 3. 图像 / 处理参数 ----------------------------------------------
# 是否在输出图像上绘制检测到的标记与坐标轴。
DEFAULT_DRAW_OVERLAY = True

# --- 4. ChArUco 标定板参数（从 CALIBRATION_JSON 派生）----------------
# 描述用于标定的已打印 ChArUco 标定板。长度单位为米。
DEFAULT_BOARD_SQUARES_X = int(_board.get("nSquaresX", 5))
DEFAULT_BOARD_SQUARES_Y = int(_board.get("nSquaresY", 7))
DEFAULT_BOARD_SQUARE_LENGTH_M = float(_board.get("sqrLength", 0.025))
DEFAULT_BOARD_MARKER_LENGTH_M = float(_board.get("markerLength", 0.018))

# 离群值剔除设置（对应 config.json 中的 "outlierRemoval"）。
DEFAULT_OUTLIER_PARAMS = {
    "reprojectionErrorThreshold": 1.0,
    "maxOutlierRemovalIterations": 3,
    "maxNormalizedReprojectionError": 2.0,
    "minImageInlierPct": 0.5,
    "minImageNum": 4,
}


class ArucoTo6D:
    """检测 ArUco 标记并估计每个标记的 6D 位姿。

    所有参数均可选。任一参数若为 ``None``，则回退到对应的模块级默认值
    （见文件顶部的 ``DEFAULT_*`` 常量以及 ``DEFAULT_CAMERA_MATRIX`` /
    ``DEFAULT_DIST_COEFFS``）。

    参数
    ----
    camera_matrix : ndarray, 可选
        3x3 相机内参矩阵（默认 ``DEFAULT_CAMERA_MATRIX``）。
    dist_coeffs : ndarray, 可选
        畸变系数 [k1, k2, p1, p2, k3]（默认 ``DEFAULT_DIST_COEFFS``）。
    dict_name : str, 可选
        预定义 ArUco 字典名称（默认 ``DEFAULT_DICT_NAME``）。
    marker_length_m : float, 可选
        标记边长，单位米（默认 ``DEFAULT_DETECT_MARKER_LENGTH_M``）。返回的平移
        即使用该单位。
    is_legacy : bool, 可选
        旧版（legacy）字典布局标志（默认 ``DEFAULT_IS_LEGACY``）。
    axis_fraction : float, 可选
        绘制的坐标轴长度，以标记边长的比例表示（默认
        ``DEFAULT_AXIS_FRACTION``）。
    axis_thickness : int, 可选
        绘制的坐标轴线宽（默认 ``DEFAULT_AXIS_THICKNESS``）。
    draw_overlay : bool, 可选
        是否绘制检测到的标记与坐标轴（默认 ``DEFAULT_DRAW_OVERLAY``）。
    output_image : str 或 None, 可选
        标注图像的保存路径；``None`` 表示不保存。
    output_json : str 或 None, 可选
        JSON 结果的保存路径；``None`` 表示不保存。
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
            marker_length_m = DEFAULT_DETECT_MARKER_LENGTH_M
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

        # ChArUco 标定板（标定用）参数。
        self.board_squares_x = int(board_squares_x)
        self.board_squares_y = int(board_squares_y)
        self.board_square_length_m = float(board_square_length_m)
        self.board_marker_length_m = float(board_marker_length_m)
        self.outlier_params = outlier_params

        # 预先构建 OpenCV 字典对象，避免每次检测重复构建。
        self._dictionary = self._build_dictionary(dict_name, is_legacy)

    # ------------------------------------------------------------------
    # 对外公开 API
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
               return_image=False,
               sort_by=MarkerSortOrder.ID_ASC):
        """检测 ArUco 标记并估计每个标记的 6D 位姿。

        所有 ArUco / 相机参数均可在单次调用时覆盖。任一参数若为 ``None``，则
        回退到实例上保存的值（实例值本身又默认来自模块级全局变量）。

        参数
        ----
        image_path : str
            输入图像路径。
        marker_length_m : float, 可选
            标记边长，单位米。覆盖实例值。
        dict_name : str, 可选
            预定义 ArUco 字典名称。覆盖实例值。
        is_legacy : bool, 可选
            旧版（legacy）字典布局标志。覆盖实例值。
        camera_matrix : ndarray, 可选
            3x3 相机内参矩阵。覆盖实例值。
        dist_coeffs : ndarray, 可选
            畸变系数。覆盖实例值。
        axis_fraction : float, 可选
            绘制的坐标轴长度比例。覆盖实例值。
        axis_thickness : int, 可选
            绘制的坐标轴线宽。覆盖实例值。
        draw_overlay : bool, 可选
            是否绘制检测到的标记与坐标轴。覆盖实例值。
        return_image : bool, 可选
            若为 True，则同时返回标注后的 BGR 图像。
        sort_by : MarkerSortOrder, 可选
            检测结果的排序方式（默认 ``MarkerSortOrder.ID_ASC``）：
                MarkerSortOrder.ID_ASC      : 按标记 id 升序（默认）
                MarkerSortOrder.ID_DESC     : 按标记 id 降序
                MarkerSortOrder.LEFT_FIRST  : 从左到右（按标记中心 x 升序）
                MarkerSortOrder.RIGHT_FIRST : 从右到左（按标记中心 x 降序）
                MarkerSortOrder.TOP_FIRST   : 从上到下（按标记中心 y 升序）
                MarkerSortOrder.BOTTOM      : 从下到上（按标记中心 y 降序）

        返回
        ----
        results : list[dict]（按 ``sort_by`` 指定的顺序排列）
            每个字典包含：
                id                    : int   标记 id
                pose6d                : [x, y, z, rx, ry, rz]
                                        x,y,z = tvec（单位同 marker_length）；
                                        rx,ry,rz = 欧拉角（度）
                tvec                  : [x, y, z]（单位同 marker_length）
                tvec_mm               : [x, y, z]（毫米）
                rvec                  : [rx, ry, rz]  OpenCV 旋转向量
                euler_deg             : [rx, ry, rz]（与 pose6d[3:] 相同）
                edge_length_px        : float   检测到的标记像素边长（四条边界均值）
                reprojection_error_px : float   每个角点的平均重投影误差
        annotated_image : ndarray（仅当 return_image=True 时返回）
            绘制了检测到的标记与位姿坐标轴的 BGR 图像。
        """
        # --- 解析本次调用实际生效的参数（单次覆盖）------------------
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

        # 仅当请求的字典与实例不同时才重建字典对象。
        if dict_name == self.dict_name and is_legacy == self.is_legacy:
            dictionary = self._dictionary
        else:
            dictionary = self._build_dictionary(dict_name, is_legacy)

        t_start = time.perf_counter()

        # --- 读取图像 ------------------------------------------------
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {image_path}")

        # --- 校验图像尺寸 --------------------------------------------
        # 内参 / 畸变系数是针对 CALIBRATION_JSON["imageSize"] 标定的，
        # 输入图片分辨率必须一致，否则位姿结果不可信。
        if DEFAULT_IMAGE_SIZE is not None:
            h, w = image.shape[:2]
            exp_w, exp_h = DEFAULT_IMAGE_SIZE
            if (w, h) != (exp_w, exp_h):
                raise ValueError(
                    f"图像尺寸不匹配: {os.path.basename(image_path)} 为 "
                    f"{w}x{h}，但内参针对 {exp_w}x{exp_h} "
                    f"（CALIBRATION_JSON['imageSize']）标定。请使用相同分辨率的图片。")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- 检测标记 ------------------------------------------------
        t_detect = time.perf_counter()
        aruco_detector = self._make_aruco_detector(dictionary)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        detect_ms = (time.perf_counter() - t_detect) * 1000.0

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

                # 重投影误差（各角点距离的平均值，单位像素）。
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam, dist)
                reproj = float(np.mean(np.linalg.norm(
                    proj_pts.reshape(-1, 2) - image_pts, axis=1)))

                rvec_f = rvec.flatten()
                tvec_f = tvec.flatten()
                tvec_mm = tvec_f * 1000.0
                euler = self._rotation_matrix_to_euler_deg(rvec)

                # 像素边长：检测到的四条边界长度平均值（与物理假设无关）。
                px_edges = [np.linalg.norm(image_pts[i] - image_pts[(i + 1) % 4])
                            for i in range(4)]
                edge_length_px = float(np.mean(px_edges))

                results.append({
                    "id": int(marker_id),
                    "pose6d": [float(tvec_f[0]), float(tvec_f[1]), float(tvec_f[2]),
                               float(euler[0]), float(euler[1]), float(euler[2])],
                    "tvec": [float(x) for x in tvec_f],
                    "tvec_mm": [float(x) for x in tvec_mm],
                    "rvec": [float(x) for x in rvec_f],
                    "euler_deg": [float(x) for x in euler],
                    "edge_length_px": edge_length_px,
                    "reprojection_error_px": reproj,
                    "center_px": [float(c) for c in image_pts.mean(axis=0)],
                })

                if draw_overlay:
                    cv2.drawFrameAxes(annotated, cam, dist, rvec, tvec,
                                      axis_len, axis_thickness)

            if draw_overlay:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        # 按 sort_by 指定的方式排序（保证下游使用者的顺序稳定）。
        results = self._sort_results(results, sort_by)

        if self.output_image:
            self._ensure_parent(self.output_image)
            cv2.imwrite(self.output_image, annotated)
            print(f"[aruco_to_6d] 标注结果图已保存到 {self.output_image}")

        if self.output_json:
            self._save_json(self.output_json, image_path, dict_name,
                            marker_length_m, results)

        total_ms = (time.perf_counter() - t_start) * 1000.0
        print(f"[aruco_to_6d] 检测耗时: 标记检测 {detect_ms:.2f} ms | "
              f"总计 {total_ms:.2f} ms（{len(results)} 个标记）")
        print(f"[aruco_to_6d] 检测到 {len(results)} 个标记：")
        if results:
            header = (f"{'ID':>4} | {'中心x(px)':>9} {'中心y(px)':>9} | "
                      f"{'x(mm)':>9} {'y(mm)':>9} {'z(mm)':>9} | "
                      f"{'rx(度)':>9} {'ry(度)':>9} {'rz(度)':>9} | "
                      f"{'像素边长(px)':>11} | {'重投影(px)':>10}")
            print(f"[aruco_to_6d] {header}")
            print(f"[aruco_to_6d] {'-' * len(header)}")
            for m in results:
                cx, cy = m["center_px"]
                x, y, z = m["tvec_mm"]
                rx, ry, rz = m["euler_deg"]
                print(f"[aruco_to_6d] {m['id']:>4} | {cx:>9.1f} {cy:>9.1f} | "
                      f"{x:>9.1f} {y:>9.1f} {z:>9.1f} | "
                      f"{rx:>9.2f} {ry:>9.2f} {rz:>9.2f} | "
                      f"{m['edge_length_px']:>11.2f} | "
                      f"{m['reprojection_error_px']:>10.3f}")

        if return_image:
            return results, annotated
        return results

    # ------------------------------------------------------------------
    # 标定
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
        """从包含 ChArUco 标定板图像的目录中标定相机。

        对应 GUI 的 "Run Calibration" 按钮：读取 ``image_dir`` 下的每张图片，
        用 ``cv2.aruco.CharucoDetector`` 检测 ChArUco 角点，再经
        ``board.matchImagePoints`` + ``cv2.calibrateCameraExtended`` 标定
        并可选进行迭代式离群剔除，最后写出 JSON 结果文件并打印进度日志。

        任一参数若为 ``None``，则回退到实例上保存的值（实例值本身又默认来自
        模块级全局变量）。

        参数
        ----
        image_dir : str
            包含标定图像的目录。
        output_json : str, 可选
            JSON 结果文件的路径。默认为
            ``<image_dir>/calibration-{YYYYMMDD_HHMMSS}.json``。
        board_squares_x, board_squares_y : int, 可选
            沿 X / Y 轴的方格数量。
        board_square_length_m, board_marker_length_m : float, 可选
            棋盘方格 / 标记的边长，单位米。
        dict_name : str, 可选
            预定义 ArUco 字典名称。
        is_legacy : bool, 可选
            旧版（legacy）字典布局标志。
        outlier_params : dict, 可选
            离群值剔除设置（见 ``DEFAULT_OUTLIER_PARAMS``）。

        返回
        ----
        dict
            标定结果：``fx``、``fy``、``ppx``、``ppy``、``K1``、``K2``、
            ``K3``、``P1``、``P2``，以及 ``reprojectionError``、``imageSize``、
            ``numImages``、``numImagesUsed``。
        """
        def log(msg):
            print(f"[aruco_to_6d] {msg}")

        t_start = time.perf_counter()

        # --- 解析实际生效的参数 --------------------------------------
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
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_json = os.path.join(image_dir, f"calibration-{stamp}.json")

        # --- 构建标定板与字典 ----------------------------------------
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

        # --- 第 1 遍：读取图像并检测 ChArUco 角点 --------------------
        t_corner = time.perf_counter()
        charuco_detector = self._make_charuco_detector(board)
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
                charuco_corners, charuco_ids, _mcorners, _mids = \
                    charuco_detector.detectBoard(img)
            except Exception as e:
                log(f"{name}: ChArUco 角点检测出错: {e}")
                continue

            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 3:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                log(f"{name}: 检测到 {len(charuco_ids)} 个 ChArUco 角点。")
            else:
                log(f"{name}: ChArUco 角点不足，已跳过。")

        corner_s = time.perf_counter() - t_corner
        log(f"角点检测耗时: {corner_s:.3f} s（{len(all_corners)}/{len(image_paths)} 张有效图片）")

        if not all_corners:
            raise RuntimeError("所有图片中均未检测到有效的 ChArUco 角点。")
        if len(all_corners) < outlier_params["minImageNum"]:
            log(f"警告: 仅有 {len(all_corners)} 张有效图片，"
                f"minImageNum 为 {outlier_params['minImageNum']}。")

        # --- 第 2 遍：标定（+ 迭代式离群剔除）------------------------
        # 迭代优化终止条件：达到最大迭代次数或精度满足容差。
        t_calib = time.perf_counter()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-9)
        n_calib_runs = 1
        used = len(all_corners)
        log(f"正在对 {used} 张有效图片运行 charuco calibrate ...")
        (retval, cam_mat, dist_coeffs, _rvecs, _tvecs,
         _std_int, _std_ext, per_view) = self._calibrate_from_corners(
            board, all_corners, all_ids, image_size, criteria)

        if retval > outlier_params["reprojectionErrorThreshold"]:
            log(f"重投影误差 ({retval:.4f}) 超过阈值 "
                f"({outlier_params['reprojectionErrorThreshold']:.4f})，"
                f"开始剔除离群图片 ...")
            keep = np.arange(used)
            for _ in range(outlier_params["maxOutlierRemovalIterations"]):
                # 使用 MAD（中位数绝对偏差）对单张图误差做归一化，较 RMS 更抗离群。
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
                 _std_int, _std_ext, per_view) = self._calibrate_from_corners(
                    board, keep_corners, keep_ids, image_size, criteria)
                n_calib_runs += 1

        calib_s = time.perf_counter() - t_calib
        log(f"标定计算耗时: {calib_s:.3f} s（共 {n_calib_runs} 次优化）")

        # --- 构建并保存结果 ------------------------------------------
        dist_flat = np.asarray(dist_coeffs).reshape(-1)
        k = list(dist_flat) + [0.0] * 5  # 补齐到至少 5 个系数
        # 结构与界面 Save 按钮保存的 calibration.json 完全对齐（data 字段除外）。
        result = {
            "intrinsics": {
                "focalLengthX": float(cam_mat[0, 0]),
                "focalLengthY": float(cam_mat[1, 1]),
                "principalPointX": float(cam_mat[0, 2]),
                "principalPointY": float(cam_mat[1, 2]),
            },
            "lensDistortions": {
                "radial": {
                    "k1": float(k[0]),
                    "k2": float(k[1]),
                    "k3": float(k[4]),
                },
                "tangential": {
                    "p1": float(k[2]),
                    "p2": float(k[3]),
                },
            },
            "calibrationSummary": {
                "rmsError": float(retval),
                "numImagesUsed": int(used),
            },
            "imageSize": list(image_size),
            "numImages": len(image_paths),
            "arucoDictionary": dict_name,
            "board": {
                "nSquaresX": board_squares_x,
                "nSquaresY": board_squares_y,
                "sqrLength": float(board_square_length_m),
                "markerLength": float(board_marker_length_m),
                "dictName": dict_name,
                "isLegacy": bool(is_legacy),
            },
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

        # --- 质量评价 ------------------------------------------------
        self.evaluate_calibration(result, per_view)

        total_s = time.perf_counter() - t_start
        log(f"标定总耗时: {total_s:.3f} s"
            f"（角点检测 {corner_s:.3f} s + 标定计算 {calib_s:.3f} s）")

        return result

    def evaluate_calibration(self, result, per_view=None):
        """评价标定结果，打印日志并给出改进建议。

        对 RMS 重投影误差、有效图片比例、图像覆盖度，以及（若有）单张视角
        误差离散程度进行分级，并针对表现较弱之处给出具体、可操作的改进建议。

        参数
        ----
        result : dict
            :meth:`calibrate` 返回的标定结果。
        per_view : array-like, 可选
            单张视角的重投影误差（像素）。提供后会额外评估逐图一致性。

        返回
        ----
        dict
            ``{"grade": str, "score": int, "rms": float, "issues": [...],
            "suggestions": [...]}``。
        """
        def log(msg):
            print(f"[aruco_to_6d] {msg}")

        summary = result.get("calibrationSummary") or {}
        intr = result.get("intrinsics") or {}
        rms = float(summary.get("rmsError", float("inf")))
        num_used = int(summary.get("numImagesUsed", 0))
        num_total = int(result.get("numImages", num_used))
        cam_mat = np.array([
            [float(intr.get("focalLengthX", 0.0)), 0.0,
             float(intr.get("principalPointX", 0.0))],
            [0.0, float(intr.get("focalLengthY", 0.0)),
             float(intr.get("principalPointY", 0.0))],
            [0.0, 0.0, 1.0]], dtype=np.float64)
        image_size = result.get("imageSize") or [0, 0]
        img_w, img_h = int(image_size[0]), int(image_size[1])

        issues = []
        suggestions = []

        # --- 1. RMS 重投影误差 --------------------------------------
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

        # --- 2. 有效（内点）图片比例 --------------------------------
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

        # --- 3. 焦距合理性 ------------------------------------------
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

        # --- 4. 图像覆盖度（主点应接近图像中心）---------------------
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

        # --- 5. 单张视角误差一致性 ----------------------------------
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

        # --- 综合评级 ----------------------------------------------
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
    def _make_aruco_detector(dictionary):
        """创建 ArUco 标记检测器。

        同时兼容 opencv-contrib-python 4.12.x 与 5.0.x：两者均提供面向对象
        的 ``cv2.aruco.ArucoDetector``（5.0 已移除模块级 ``detectMarkers``）。
        """
        return cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())

    @staticmethod
    def _make_charuco_detector(board):
        """创建 ChArUco 检测器。

        同时兼容 opencv-contrib-python 4.12.x 与 5.0.x：两者均提供
        ``cv2.aruco.CharucoDetector``（5.0 已移除模块级
        ``interpolateCornersCharuco``）。
        """
        return cv2.aruco.CharucoDetector(board)

    @staticmethod
    def _calibrate_from_corners(board, charuco_corners, charuco_ids,
                                image_size, criteria):
        """由 ChArUco 角点标定相机（兼容 4.12.x 与 5.0.x）。

        5.0 已移除 ``cv2.aruco.calibrateCameraCharucoExtended``，统一改用
        ``board.matchImagePoints`` + ``cv2.calibrateCameraExtended``。显式传入
        ``np.eye(3)`` / ``np.zeros(5)`` 作为初始猜测（与 ``None`` 等价，但跨版本
        更鲁棒；且每次新建，避免被上一次结果原地污染）。
        """
        obj_pts_all, img_pts_all = [], []
        for corners, ids in zip(charuco_corners, charuco_ids):
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            obj_pts_all.append(obj_pts)
            img_pts_all.append(img_pts)
        camera_matrix = np.eye(3, dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        return cv2.calibrateCameraExtended(
            obj_pts_all, img_pts_all, image_size, camera_matrix, dist_coeffs,
            criteria=criteria)

    @staticmethod
    def _create_charuco_board(squares_x, squares_y, square_length,
                              marker_length, aruco_dict):
        """创建 ChArUco 标定板。

        4.12.x 与 5.0.x 均支持 ``CharucoBoard((sx, sy), sq, mk, dict)``
        构造形式（``CharucoBoard_create`` / ``CharucoBoard.create`` 已废弃或
        不存在）。
        """
        try:
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_length, marker_length, aruco_dict)
        except Exception as e:
            raise AttributeError(
                f"Unable to create CharucoBoard. Error: {e}")

    # ------------------------------------------------------------------
    # 内部辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _build_dictionary(dict_name, is_legacy):
        """创建预定义 ArUco 字典对象。

        旧版（legacy）布局标志由 :class:`cv2.aruco.CharucoBoard` 的
        ``setLegacyPattern`` 控制（见 ``calibrate``），字典对象本身在
        4.12.x / 5.0.x 中均不提供该接口，故此处不再设置。``is_legacy``
        参数仅为保持签名兼容而保留。
        """
        enum_val = getattr(cv2.aruco, dict_name, None)
        if enum_val is None:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
        return cv2.aruco.getPredefinedDictionary(enum_val)

    @staticmethod
    def _rotation_matrix_to_euler_deg(rvec):
        """将 OpenCV 旋转向量转换为内旋 XYZ 欧拉角（度）。

        采用 R = Rx(rx) * Ry(ry) * Rz(rz)：
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
        else:  # 万向节死锁情况
            rx = np.arctan2(-rot[1, 2], rot[1, 1])
            ry = np.arctan2(-rot[2, 0], sy)
            rz = 0.0
        return np.degrees([rx, ry, rz])

    @staticmethod
    def _marker_object_points(marker_length):
        """标记位于 Z=0 平面、中心在原点时的 3D 角点坐标。

        角点顺序与 OpenCV detectMarkers 一致：
        左上、右上、右下、左下。
        """
        half = marker_length / 2.0
        return np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0]], dtype=np.float64)

    @staticmethod
    def _sort_results(results, sort_by):
        """按指定方式对检测结果排序并返回新列表。

        参数
        ----
        results : list[dict]
            :meth:`detect` 生成的结果列表（每项含 id / center_px）。
        sort_by : MarkerSortOrder
            排序方式（见 :class:`MarkerSortOrder`）。

        返回
        ----
        list[dict]
            排序后的新列表（不修改入参）。
        """
        if sort_by is None:
            sort_by = MarkerSortOrder.ID_ASC
        if not isinstance(sort_by, MarkerSortOrder):
            raise TypeError(
                f"sort_by 必须是 MarkerSortOrder 枚举，而非 {type(sort_by).__name__}")
        if sort_by is MarkerSortOrder.ID_ASC:
            return sorted(results, key=lambda m: m["id"])
        if sort_by is MarkerSortOrder.ID_DESC:
            return sorted(results, key=lambda m: m["id"], reverse=True)
        if sort_by is MarkerSortOrder.LEFT_FIRST:
            return sorted(results, key=lambda m: m["center_px"][0])
        if sort_by is MarkerSortOrder.RIGHT_FIRST:
            return sorted(results, key=lambda m: m["center_px"][0], reverse=True)
        if sort_by is MarkerSortOrder.TOP_FIRST:
            return sorted(results, key=lambda m: m["center_px"][1])
        if sort_by is MarkerSortOrder.BOTTOM:
            return sorted(results, key=lambda m: m["center_px"][1], reverse=True)
        raise ValueError(f"Unknown sort_by: {sort_by!r}")

    @staticmethod
    def _ensure_parent(path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _save_json(self, path, image_path, dict_name, marker_length_m, results):
        """将识别结果写入 JSON 文件（与 GUI 共享相同的数据结构）。"""
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
# 内置测试 / 演示入口
# ======================================================================

def test_calibrate():
    """在内置测试目录上运行标定流程，并打印质量评价。"""
    detector = ArucoTo6D()
    # 内置标定测试目录（g2 图片）。
    TEST_CALIBRATION_DIR = r"D:\dev\workspace\iss\aruco\CharucoCameraCalibration\images\g2-20261123"

    print()
    print(f"[aruco_to_6d] 标定测试目录 : {TEST_CALIBRATION_DIR}")
    try:
        detector.calibrate(TEST_CALIBRATION_DIR)
    except Exception as e:
        print(f"[aruco_to_6d] 标定测试失败: {e}")

def test_detect():
    """在内置测试图片上运行检测，并打印每个标记的 6D 位姿。"""
    detector = ArucoTo6D()
    # 内置测试图片。
    TEST_IMAGE = r"D:\dev\workspace\iss\aruco\CharucoCameraCalibration\images\g2-20261123\110-left-g2.jpg"

    
    print(f"[aruco_to_6d] 测试图片 : {TEST_IMAGE}")
    print(f"[aruco_to_6d] 内参     : "
            f"(fx={detector.camera_matrix[0, 0]:.3f}, fy={detector.camera_matrix[1, 1]:.3f})")
    print(f"[aruco_to_6d] 字典     : {detector.dict_name} | 标记边长 = "
            f"{detector.marker_length_m * 1000:.1f} mm | 旧版布局 = {detector.is_legacy}")

    results = detector.detect(TEST_IMAGE)

    if not results:
        print("[aruco_to_6d] 未检测到 ArUco 标记。")
        return



def _main():
    # 测试标定
    # test_calibrate()
    
    # 测试检测
    test_detect()


if __name__ == "__main__":
    _main()
