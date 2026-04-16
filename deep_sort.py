"""
DeepSORT 跟踪算法实现
包含：
1. 匈牙利算法（线性分配）
2. 级联匹配
3. IOU匹配
4. 特征距离计算
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque


def iou_batch(bb_test, bb_gt):
    """
    计算两组边界框之间的IOU矩阵
    bb_test: [N, 4] 检测框
    bb_gt: [M, 4] 跟踪框
    返回: [N, M] IOU矩阵
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    
    iou = wh / (area_gt + area_test - wh + 1e-6)
    
    return iou


def linear_assignment(cost_matrix):
    """
    匈牙利算法求解最优分配
    """
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def cosine_distance(a, b):
    """
    计算余弦距离
    a: [N, D] 特征向量
    b: [M, D] 特征向量
    返回: [N, M] 距离矩阵
    """
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    
    a_normalized = a / (a_norm + 1e-6)
    b_normalized = b / (b_norm + 1e-6)
    
    similarity = np.dot(a_normalized, b_normalized.T)
    distance = 1 - similarity
    
    return distance


class TrackState:
    """跟踪状态枚举"""
    Tentative = 1  # 暂定状态（新跟踪）
    Confirmed = 2  # 确认状态
    Deleted = 3    # 删除状态


class Track:
    """
    单目标跟踪器
    """
    def __init__(self, track_id, bbox, feature=None, max_age=30):
        self.track_id = track_id
        self.bbox = bbox
        self.features = deque(maxlen=100)
        if feature is not None:
            self.features.append(feature)
        
        self.state = TrackState.Tentative
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.max_age = max_age
        
        # 卡尔曼滤波器实例
        self.kf_tracker = None
        
    def update(self, bbox, feature=None):
        """更新跟踪状态"""
        self.bbox = bbox
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        
        if feature is not None:
            self.features.append(feature)
        
        # 如果命中次数足够，转为确认状态
        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed
    
    def predict(self):
        """预测下一帧位置"""
        self.age += 1
        self.time_since_update += 1
        if self.kf_tracker is not None:
            self.bbox = self.kf_tracker.predict()
    
    def mark_missed(self):
        """标记为丢失"""
        self.time_since_update += 1
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted
    
    def is_confirmed(self):
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        return self.state == TrackState.Deleted
    
    def is_tentative(self):
        return self.state == TrackState.Tentative
    
    def get_feature(self):
        """获取平均特征"""
        if len(self.features) == 0:
            return None
        return np.mean(self.features, axis=0)


class DeepSORT:
    """
    DeepSORT 多目标跟踪器
    """
    def __init__(self, 
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 max_cosine_distance=0.2,
                 nn_budget=100):
        """
        初始化DeepSORT
        
        参数:
            max_age: 跟踪器最大存活帧数
            min_hits: 确认跟踪所需的最小命中次数
            iou_threshold: IOU匹配阈值
            max_cosine_distance: 最大余弦距离
            nn_budget: 特征库预算
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections, features=None, frame_id=0):
        """
        更新跟踪器
        
        参数:
            detections: [N, 4] 检测框 [x1, y1, x2, y2]
            features: [N, D] 特征向量 (可选)
            frame_id: 当前帧ID
            
        返回:
            outputs: [M, 5] 跟踪结果 [x1, y1, x2, y2, track_id]
        """
        # 预测所有跟踪器的下一帧位置
        for track in self.tracks:
            track.predict()
        
        # 级联匹配 + IOU匹配
        matched, unmatched_dets, unmatched_tracks = \
            self._match(detections, features)
        
        # 更新匹配的跟踪器
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            bbox = detections[det_idx]
            feature = features[det_idx] if features is not None else None
            track.update(bbox, feature)
        
        # 处理未匹配的跟踪器
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 为未匹配的检测创建新跟踪器
        for det_idx in unmatched_dets:
            bbox = detections[det_idx]
            feature = features[det_idx] if features is not None else None
            self._initiate_track(bbox, feature)
        
        # 清理已删除的跟踪器
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # 输出结果
        outputs = []
        for track in self.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.bbox[:4]
                outputs.append(np.concatenate([bbox, [track.track_id]]))
        
        return np.array(outputs) if len(outputs) > 0 else np.empty((0, 5))
    
    def _match(self, detections, features):
        """
        级联匹配 + IOU匹配
        """
        # 分离确认状态和暂定状态的跟踪器
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        
        # 级联匹配 (仅对确认状态的跟踪器)
        matches_cascade, unmatched_tracks_cascade, unmatched_dets = \
            self._cascade_matching(detections, features, confirmed_tracks)
        
        # IOU匹配 (对未匹配的确认跟踪器和所有暂定跟踪器)
        iou_track_candidates = unconfirmed_tracks + \
            [k for k in unmatched_tracks_cascade if self.tracks[k].time_since_update == 1]
        unmatched_tracks_cascade = [
            k for k in unmatched_tracks_cascade if self.tracks[k].time_since_update != 1
        ]
        
        matches_iou, unmatched_tracks_iou, unmatched_dets = \
            self._iou_matching(detections, unmatched_dets, iou_track_candidates)
        
        # 合并匹配结果
        matches = matches_cascade + matches_iou
        unmatched_tracks = list(set(unmatched_tracks_cascade + unmatched_tracks_iou))
        
        return matches, unmatched_dets, unmatched_tracks
    
    def _cascade_matching(self, detections, features, track_indices):
        """
        级联匹配 - 基于外观特征
        """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
        
        # 构建成本矩阵 (余弦距离)
        cost_matrix = np.zeros((len(track_indices), len(detections)))
        
        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_feature = track.get_feature()
            
            if track_feature is not None and features is not None:
                distances = cosine_distance(track_feature.reshape(1, -1), features)
                cost_matrix[i, :] = distances[0]
            else:
                cost_matrix[i, :] = self.max_cosine_distance
        
        # 应用距离阈值
        cost_matrix[cost_matrix > self.max_cosine_distance] = self.max_cosine_distance + 1e-5
        
        # 匈牙利算法
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_dets = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.max_cosine_distance:
                matches.append((track_indices[row], col))
                unmatched_tracks.remove(track_indices[row])
                unmatched_dets.remove(col)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _iou_matching(self, detections, det_indices, track_indices):
        """
        IOU匹配 - 基于位置重叠
        """
        if len(track_indices) == 0 or len(det_indices) == 0:
            return [], track_indices, det_indices
        
        # 获取跟踪器和检测的边界框
        track_bboxes = np.array([self.tracks[i].bbox[:4] for i in track_indices])
        det_bboxes = detections[det_indices]
        
        # 计算IOU矩阵
        iou_matrix = iou_batch(det_bboxes, track_bboxes)
        
        # 转换为成本矩阵 (1 - IOU)
        cost_matrix = 1 - iou_matrix
        
        # 匈牙利算法
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_dets = list(det_indices)
        
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                matches.append((track_indices[col], det_indices[row]))
                unmatched_tracks.remove(track_indices[col])
                unmatched_dets.remove(det_indices[row])
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _initiate_track(self, bbox, feature=None):
        """初始化新跟踪器"""
        track = Track(self.next_id, bbox, feature, self.max_age)
        self.tracks.append(track)
        self.next_id += 1


class FaceDeepSORT(DeepSORT):
    """
    专门用于人脸跟踪的DeepSORT
    增加了人脸特征提取和档案管理
    """
    
    def __init__(self, 
                 max_age=30,
                 min_hits=3,
                 iou_threshold=0.3,
                 max_cosine_distance=0.2,
                 nn_budget=100,
                 face_feature_extractor=None):
        super().__init__(max_age, min_hits, iou_threshold, 
                        max_cosine_distance, nn_budget)
        
        self.face_feature_extractor = face_feature_extractor
        self.track_face_features = {}  # track_id -> face_features
        
    def update_with_faces(self, face_detections, frame, frame_id=0, timestamp=None):
        """
        使用人脸检测结果更新跟踪器
        
        参数:
            face_detections: [{'bbox': [x1,y1,x2,y2], 'landmarks': [...]}, ...]
            frame: 当前帧图像
            frame_id: 帧ID
            timestamp: 时间戳
        """
        if len(face_detections) == 0:
            # 没有检测到人脸，更新所有跟踪器
            return self.update(np.empty((0, 4)), None, frame_id)
        
        # 提取边界框
        bboxes = np.array([d['bbox'] for d in face_detections])
        
        # 提取人脸特征
        features = None
        if self.face_feature_extractor is not None:
            features = self._extract_face_features(frame, face_detections)
        
        # 更新跟踪器
        outputs = self.update(bboxes, features, frame_id)
        
        # 更新人脸特征库
        for output in outputs:
            x1, y1, x2, y2, track_id = output
            # 找到对应的检测
            for i, det in enumerate(face_detections):
                det_bbox = det['bbox']
                iou = self._compute_iou([x1, y1, x2, y2], det_bbox)
                if iou > 0.5:
                    if features is not None:
                        if track_id not in self.track_face_features:
                            self.track_face_features[track_id] = []
                        self.track_face_features[track_id].append(features[i])
                    break
        
        return outputs
    
    def _extract_face_features(self, frame, face_detections):
        """提取人脸特征向量"""
        if self.face_feature_extractor is None:
            return None
        
        features = []
        for det in face_detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                feature = self.face_feature_extractor.extract_feature(face_img)
                features.append(feature)
            else:
                features.append(np.zeros(128))  # 默认特征
        
        return np.array(features)
    
    def _compute_iou(self, bbox1, bbox2):
        """计算两个边界框的IOU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def get_track_face_feature(self, track_id):
        """获取指定跟踪器的人脸特征"""
        if track_id in self.track_face_features:
            features = self.track_face_features[track_id]
            if len(features) > 0:
                return np.mean(features, axis=0)
        return None
