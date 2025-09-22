"""
相机参数加载模块
"""

import json
import os
import numpy as np
from typing import Tuple
from ament_index_python.packages import get_package_share_directory


def load_camera_params() -> Tuple[np.ndarray, np.ndarray]:
    """
    读取相机标定参数（内参和畸变），返回相机矩阵和畸变系数。
    """
    base_dir = get_package_share_directory('assets')
    calib_path = os.path.join(base_dir, 'calibration.json')
    with open(calib_path, 'r') as f:
        params = json.load(f)
    camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(params['dist_coeffs'], dtype=np.float32).reshape(-1, 1)
    return camera_matrix, dist_coeffs