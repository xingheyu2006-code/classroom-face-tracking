"""
卡尔曼滤波器模块
用于预测和更新目标的运动状态
"""
import numpy as np


class KalmanFilter:
    """
    标准卡尔曼滤波器实现
    不依赖外部库
    """
    
    def __init__(self, dim_x, dim_z):
        """
        初始化卡尔曼滤波器
        
        参数:
            dim_x: 状态维度
            dim_z: 观测维度
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # 状态向量
        self.x = np.zeros((dim_x, 1))
        
        # 状态转移矩阵
        self.F = np.eye(dim_x)
        
        # 观测矩阵
        self.H = np.zeros((dim_z, dim_x))
        
        # 过程噪声协方差
        self.Q = np.eye(dim_x)
        
        # 测量噪声协方差
        self.R = np.eye(dim_z)
        
        # 估计误差协方差
        self.P = np.eye(dim_x)
    
    def predict(self):
        """
        预测步骤
        """
        # 状态预测: x' = F * x
        self.x = np.dot(self.F, self.x)
        
        # 协方差预测: P' = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x
    
    def update(self, z):
        """
        更新步骤
        
        参数:
            z: 观测值
        """
        # 确保z是列向量
        z = np.atleast_2d(z).reshape(-1, 1)
        
        # 计算卡尔曼增益
        # K = P * H^T * (H * P * H^T + R)^-1
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 状态更新: x = x' + K * (z - H * x')
        y = z - np.dot(self.H, self.x)  # 残差
        self.x = self.x + np.dot(K, y)
        
        # 协方差更新: P = (I - K * H) * P
        I = np.eye(self.dim_x)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.x


class KalmanBoxTracker:
    """
    基于卡尔曼滤波的边界框跟踪器
    状态空间: [x, y, w, h, vx, vy, vw, vh]
    观测空间: [x, y, w, h]
    """
    count = 0
    
    def __init__(self, bbox):
        """
        初始化跟踪器
        bbox: [x1, y1, x2, y2] 或 [x, y, w, h]
        """
        # 定义卡尔曼滤波器 (8维状态, 4维观测)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 状态转移矩阵 (匀速模型)
        dt = 1.0  # 时间间隔
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # 测量噪声协方差
        self.kf.R = np.eye(4) * 10.0
        
        # 过程噪声协方差
        self.kf.Q = np.eye(8)
        self.kf.Q[4:, 4:] *= 0.01  # 速度噪声较小
        
        # 初始估计误差协方差
        self.kf.P = np.eye(8) * 10.0
        self.kf.P[4:, 4:] *= 1000.0  # 速度初始不确定性大
        
        # 将bbox转换为 [x, y, w, h] 格式并初始化状态
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        # 跟踪器状态
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # 人脸特征向量
        self.face_features = []
        
        # 空间位置历史
        self.spatial_history = []
        
        # 时间戳历史
        self.timestamp_history = []
    
    def _convert_bbox_to_z(self, bbox):
        """将 [x1, y1, x2, y2] 转换为 [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        return np.array([x, y, w, h]).reshape(-1, 1)
    
    def _convert_x_to_bbox(self, x, score=None):
        """将 [x, y, w, h] 转换为 [x1, y1, x2, y2]"""
        x, y, w, h = x[:4].flatten()
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        if score is None:
            return np.array([x1, y1, x2, y2])
        else:
            return np.array([x1, y1, x2, y2, score])
    
    def update(self, bbox, timestamp=None):
        """
        使用观测到的边界框更新状态
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # 卡尔曼滤波更新
        z = self._convert_bbox_to_z(bbox).flatten()
        self.kf.update(z)
        
        # 记录空间位置
        center_x, center_y = self.get_position()
        self.spatial_history.append({
            'x': center_x,
            'y': center_y,
            'bbox': bbox
        })
        
        # 限制历史长度
        if len(self.spatial_history) > 1000:
            self.spatial_history = self.spatial_history[-1000:]
        
        # 记录时间戳
        if timestamp is not None:
            self.timestamp_history.append(timestamp)
    
    def predict(self):
        """
        预测下一帧的位置
        """
        # 确保宽高为正
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] *= 0.0
        if (self.kf.x[3] + self.kf.x[7]) <= 0:
            self.kf.x[7] *= 0.0
        
        # 卡尔曼滤波预测
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        """获取当前状态 [x1, y1, x2, y2]"""
        return self._convert_x_to_bbox(self.kf.x)
    
    def get_position(self):
        """获取中心位置"""
        x, y, w, h = self.kf.x[:4].flatten()
        return x, y
    
    def get_velocity(self):
        """获取速度"""
        vx, vy, vw, vh = self.kf.x[4:].flatten()
        return vx, vy, vw, vh
    
    def add_face_feature(self, feature_vector):
        """添加人脸特征向量"""
        self.face_features.append(feature_vector)
        # 只保留最近100个特征向量
        if len(self.face_features) > 100:
            self.face_features = self.face_features[-100:]
    
    def get_average_face_feature(self):
        """获取平均人脸特征"""
        if len(self.face_features) == 0:
            return None
        return np.mean(self.face_features, axis=0)
    
    def get_spatial_vector(self):
        """
        获取空间向量
        包含: 位置、速度、活动范围、轨迹
        """
        if len(self.spatial_history) == 0:
            return None
        
        # 当前位置
        current_pos = self.spatial_history[-1]
        
        # 计算速度
        velocity = {'vx': 0, 'vy': 0}
        if len(self.spatial_history) >= 2:
            prev_pos = self.spatial_history[-2]
            dt = 1.0  # 假设1帧的时间间隔
            velocity['vx'] = (current_pos['x'] - prev_pos['x']) / dt
            velocity['vy'] = (current_pos['y'] - prev_pos['y']) / dt
        
        # 计算活动范围
        if len(self.spatial_history) >= 2:
            xs = [p['x'] for p in self.spatial_history]
            ys = [p['y'] for p in self.spatial_history]
            activity_range = {
                'x_range': max(xs) - min(xs),
                'y_range': max(ys) - min(ys),
                'area': (max(xs) - min(xs)) * (max(ys) - min(ys))
            }
        else:
            activity_range = {'x_range': 0, 'y_range': 0, 'area': 0}
        
        return {
            'position': current_pos,
            'velocity': velocity,
            'activity_range': activity_range,
            'trajectory': self.spatial_history[-50:]  # 最近50帧的轨迹
        }
    
    def get_temporal_vector(self):
        """
        获取时间向量
        包含: 出现时间、持续时间、出现次数
        """
        if len(self.timestamp_history) == 0:
            return None
        
        # 首次出现时间
        first_seen = self.timestamp_history[0]
        
        # 最后出现时间
        last_seen = self.timestamp_history[-1]
        
        # 总持续时间
        duration = last_seen - first_seen
        
        # 出现频率
        appearance_count = len(self.timestamp_history)
        
        # 计算时间间隔统计
        interval_stats = {}
        if len(self.timestamp_history) >= 2:
            intervals = []
            for i in range(1, len(self.timestamp_history)):
                interval = self.timestamp_history[i] - self.timestamp_history[i-1]
                intervals.append(interval)
            
            interval_stats = {
                'mean_interval': np.mean(intervals),
                'std_interval': np.std(intervals),
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals)
            }
        
        return {
            'first_seen': first_seen,
            'last_seen': last_seen,
            'duration': duration,
            'appearance_count': appearance_count,
            'interval_stats': interval_stats
        }


class KalmanFaceTracker(KalmanBoxTracker):
    """
    专门用于人脸跟踪的卡尔曼滤波器
    增加了人脸关键点跟踪
    """
    
    def __init__(self, bbox, face_landmarks=None):
        super().__init__(bbox)
        
        # 人脸关键点 (5个点: 左眼、右眼、鼻子、左嘴角、右嘴角)
        self.landmarks_history = []
        if face_landmarks is not None:
            self.landmarks_history.append(face_landmarks)
        
        # 人脸朝向估计
        self.head_pose_history = []
        
        # 注意力状态 (基于头部姿态)
        self.attention_history = []
    
    def update_landmarks(self, landmarks):
        """更新人脸关键点"""
        if landmarks is not None:
            self.landmarks_history.append(landmarks)
            if len(self.landmarks_history) > 100:
                self.landmarks_history = self.landmarks_history[-100:]
    
    def update_head_pose(self, pose):
        """更新头部姿态"""
        if pose is not None:
            self.head_pose_history.append(pose)
            if len(self.head_pose_history) > 100:
                self.head_pose_history = self.head_pose_history[-100:]
    
    def update_attention(self, attention_score):
        """更新注意力分数"""
        self.attention_history.append(attention_score)
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-100:]
