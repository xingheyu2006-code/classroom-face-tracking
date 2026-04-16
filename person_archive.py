"""
个人档案管理模块
为每个人员创建独立的档案，记录时间向量和空间向量
"""
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class PersonArchive:
    """
    个人档案类
    存储单个人员的所有信息
    """
    
    def __init__(self, person_id, created_at=None):
        """
        初始化个人档案
        
        参数:
            person_id: 人员唯一ID
            created_at: 创建时间
        """
        self.person_id = person_id
        self.created_at = created_at if created_at else datetime.now().isoformat()
        
        # 基本信息
        self.info = {
            'person_id': person_id,
            'created_at': self.created_at,
            'last_updated': self.created_at,
            'total_appearances': 0,
            'total_duration': 0.0  # 秒
        }
        
        # 人脸特征向量历史
        self.face_features = []
        
        # 空间向量历史
        # 每个元素: {'timestamp': float, 'x': float, 'y': float, 'bbox': [...]}
        self.spatial_vectors = []
        
        # 时间向量历史
        # 每个元素: {'timestamp': float, 'frame_id': int}
        self.temporal_vectors = []
        
        # 轨迹历史 (最近N帧的位置)
        self.trajectory = []
        
        # 注意力历史 (如果有)
        self.attention_history = []
        
        # 人脸图像样本 (保存最近的几张)
        self.face_samples = []
        self.max_face_samples = 10
        
        # 元数据
        self.metadata = {}
    
    def update(self, timestamp, frame_id, bbox, face_feature=None, 
               face_image=None, attention_score=None):
        """
        更新档案
        
        参数:
            timestamp: 时间戳
            frame_id: 帧ID
            bbox: 边界框 [x1, y1, x2, y2]
            face_feature: 人脸特征向量
            face_image: 人脸图像 (可选)
            attention_score: 注意力分数 (可选)
        """
        # 更新基本信息
        self.info['last_updated'] = datetime.now().isoformat()
        self.info['total_appearances'] += 1
        
        # 计算中心位置
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # 添加空间向量
        spatial_vector = {
            'timestamp': timestamp,
            'frame_id': frame_id,
            'x': center_x,
            'y': center_y,
            'bbox': bbox,
            'width': x2 - x1,
            'height': y2 - y1
        }
        self.spatial_vectors.append(spatial_vector)
        
        # 添加时间向量
        temporal_vector = {
            'timestamp': timestamp,
            'frame_id': frame_id
        }
        self.temporal_vectors.append(temporal_vector)
        
        # 添加轨迹
        self.trajectory.append({
            'timestamp': timestamp,
            'x': center_x,
            'y': center_y
        })
        
        # 限制轨迹长度
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-1000:]
        
        # 添加人脸特征
        if face_feature is not None:
            self.face_features.append({
                'timestamp': timestamp,
                'frame_id': frame_id,
                'feature': face_feature
            })
            
            # 限制特征数量
            if len(self.face_features) > 1000:
                self.face_features = self.face_features[-1000:]
        
        # 添加人脸样本
        if face_image is not None:
            self._add_face_sample(face_image, timestamp)
        
        # 添加注意力分数
        if attention_score is not None:
            self.attention_history.append({
                'timestamp': timestamp,
                'frame_id': frame_id,
                'score': attention_score
            })
            
            if len(self.attention_history) > 1000:
                self.attention_history = self.attention_history[-1000:]
        
        # 更新总持续时间
        if len(self.temporal_vectors) >= 2:
            self.info['total_duration'] = \
                self.temporal_vectors[-1]['timestamp'] - self.temporal_vectors[0]['timestamp']
    
    def _add_face_sample(self, face_image, timestamp):
        """添加人脸样本"""
        if len(self.face_samples) < self.max_face_samples:
            self.face_samples.append({
                'timestamp': timestamp,
                'image': face_image
            })
    
    def get_spatial_vector(self):
        """
        获取当前空间向量
        
        返回:
            spatial_vector: 空间向量字典
        """
        if len(self.spatial_vectors) == 0:
            return None
        
        current = self.spatial_vectors[-1]
        
        # 计算速度
        velocity = {'vx': 0, 'vy': 0}
        if len(self.spatial_vectors) >= 2:
            prev = self.spatial_vectors[-2]
            dt = current['timestamp'] - prev['timestamp']
            if dt > 0:
                velocity['vx'] = (current['x'] - prev['x']) / dt
                velocity['vy'] = (current['y'] - prev['y']) / dt
        
        # 计算活动范围
        activity_range = self._compute_activity_range()
        
        # 获取轨迹
        trajectory = self.trajectory[-100:]  # 最近100个点
        
        return {
            'current_position': {
                'x': current['x'],
                'y': current['y'],
                'bbox': current['bbox']
            },
            'velocity': velocity,
            'activity_range': activity_range,
            'trajectory': trajectory
        }
    
    def get_temporal_vector(self):
        """
        获取当前时间向量
        
        返回:
            temporal_vector: 时间向量字典
        """
        if len(self.temporal_vectors) == 0:
            return None
        
        first_seen = self.temporal_vectors[0]['timestamp']
        last_seen = self.temporal_vectors[-1]['timestamp']
        duration = last_seen - first_seen
        
        # 计算出现频率
        appearance_count = len(self.temporal_vectors)
        
        # 计算时间间隔统计
        intervals = []
        for i in range(1, len(self.temporal_vectors)):
            interval = self.temporal_vectors[i]['timestamp'] - self.temporal_vectors[i-1]['timestamp']
            intervals.append(interval)
        
        interval_stats = {}
        if len(intervals) > 0:
            interval_stats = {
                'mean_interval': np.mean(intervals),
                'std_interval': np.std(intervals),
                'min_interval': np.min(intervals),
                'max_interval': np.max(intervals)
            }
        
        # 注意力统计
        attention_stats = {}
        if len(self.attention_history) > 0:
            scores = [a['score'] for a in self.attention_history]
            attention_stats = {
                'mean_attention': np.mean(scores),
                'std_attention': np.std(scores),
                'min_attention': np.min(scores),
                'max_attention': np.max(scores)
            }
        
        return {
            'first_seen': first_seen,
            'last_seen': last_seen,
            'duration': duration,
            'appearance_count': appearance_count,
            'interval_stats': interval_stats,
            'attention_stats': attention_stats
        }
    
    def get_face_feature_vector(self):
        """
        获取人脸特征向量
        
        返回:
            feature_vector: 平均特征向量
        """
        if len(self.face_features) == 0:
            return None
        
        features = [f['feature'] for f in self.face_features]
        avg_feature = np.mean(features, axis=0)
        
        # L2归一化
        avg_feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-6)
        
        return avg_feature
    
    def _compute_activity_range(self):
        """计算活动范围"""
        if len(self.spatial_vectors) < 2:
            return {'x_range': 0, 'y_range': 0, 'area': 0}
        
        xs = [s['x'] for s in self.spatial_vectors]
        ys = [s['y'] for s in self.spatial_vectors]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        area = x_range * y_range
        
        return {
            'x_range': x_range,
            'y_range': y_range,
            'area': area,
            'bounding_box': {
                'min_x': min(xs),
                'max_x': max(xs),
                'min_y': min(ys),
                'max_y': max(ys)
            }
        }
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'info': self.info,
            'spatial_vector': self.get_spatial_vector(),
            'temporal_vector': self.get_temporal_vector(),
            'face_feature': self.get_face_feature_vector().tolist() if self.get_face_feature_vector() is not None else None,
            'metadata': self.metadata
        }
    
    def save(self, filepath):
        """保存档案到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为pickle格式 (包含所有数据)
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self, f)
        
        # 保存为JSON格式 (仅包含可序列化数据)
        json_data = self.to_dict()
        with open(filepath.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(filepath):
        """从文件加载档案"""
        filepath = Path(filepath)
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            return pickle.load(f)


class ArchiveManager:
    """
    档案管理器
    管理所有人员的档案
    """
    
    def __init__(self, archive_dir='./archives'):
        """
        初始化档案管理器
        
        参数:
            archive_dir: 档案存储目录
        """
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 档案字典: person_id -> PersonArchive
        self.archives = {}
        
        # ID映射: track_id -> person_id
        self.track_to_person = {}
        
        # 下一个可用的人员ID
        self.next_person_id = 1
        
        # 统计信息
        self.stats = {
            'total_archives': 0,
            'active_archives': 0,
            'total_updates': 0
        }
    
    def get_or_create_archive(self, track_id, timestamp=None):
        """
        获取或创建档案
        
        参数:
            track_id: 跟踪ID
            timestamp: 时间戳
            
        返回:
            archive: PersonArchive实例
            is_new: 是否为新创建的档案
        """
        is_new = False
        
        if track_id in self.track_to_person:
            person_id = self.track_to_person[track_id]
            archive = self.archives[person_id]
        else:
            # 创建新档案
            person_id = self._generate_person_id()
            archive = PersonArchive(person_id, timestamp)
            self.archives[person_id] = archive
            self.track_to_person[track_id] = person_id
            is_new = True
            self.stats['total_archives'] += 1
            self.stats['active_archives'] += 1
        
        return archive, is_new
    
    def _generate_person_id(self):
        """生成新的人员ID"""
        person_id = f"person_{self.next_person_id:06d}"
        self.next_person_id += 1
        return person_id
    
    def update_archive(self, track_id, timestamp, frame_id, bbox, 
                       face_feature=None, face_image=None, attention_score=None):
        """
        更新档案
        
        参数:
            track_id: 跟踪ID
            timestamp: 时间戳
            frame_id: 帧ID
            bbox: 边界框
            face_feature: 人脸特征向量
            face_image: 人脸图像
            attention_score: 注意力分数
        """
        archive, is_new = self.get_or_create_archive(track_id, timestamp)
        
        archive.update(
            timestamp=timestamp,
            frame_id=frame_id,
            bbox=bbox,
            face_feature=face_feature,
            face_image=face_image,
            attention_score=attention_score
        )
        
        self.stats['total_updates'] += 1
        
        return archive, is_new
    
    def get_archive(self, person_id):
        """获取指定人员的档案"""
        return self.archives.get(person_id, None)
    
    def get_archive_by_track_id(self, track_id):
        """通过跟踪ID获取档案"""
        if track_id in self.track_to_person:
            person_id = self.track_to_person[track_id]
            return self.archives.get(person_id, None)
        return None
    
    def remove_archive(self, person_id):
        """删除档案"""
        if person_id in self.archives:
            del self.archives[person_id]
            
            # 更新track_to_person映射
            tracks_to_remove = [t for t, p in self.track_to_person.items() if p == person_id]
            for track in tracks_to_remove:
                del self.track_to_person[track]
            
            self.stats['active_archives'] -= 1
    
    def merge_archives(self, person_id1, person_id2):
        """
        合并两个档案
        当确定两个人员ID指向同一人时使用
        """
        if person_id1 not in self.archives or person_id2 not in self.archives:
            return False
        
        archive1 = self.archives[person_id1]
        archive2 = self.archives[person_id2]
        
        # 将archive2的数据合并到archive1
        archive1.spatial_vectors.extend(archive2.spatial_vectors)
        archive1.spatial_vectors.sort(key=lambda x: x['timestamp'])
        
        archive1.temporal_vectors.extend(archive2.temporal_vectors)
        archive1.temporal_vectors.sort(key=lambda x: x['timestamp'])
        
        archive1.face_features.extend(archive2.face_features)
        archive1.face_features.sort(key=lambda x: x['timestamp'])
        
        archive1.trajectory.extend(archive2.trajectory)
        archive1.trajectory.sort(key=lambda x: x['timestamp'])
        
        archive1.attention_history.extend(archive2.attention_history)
        archive1.attention_history.sort(key=lambda x: x['timestamp'])
        
        # 更新统计
        archive1.info['total_appearances'] += archive2.info['total_appearances']
        
        # 删除archive2
        self.remove_archive(person_id2)
        
        return True
    
    def save_all(self):
        """保存所有档案"""
        for person_id, archive in self.archives.items():
            filepath = self.archive_dir / person_id
            archive.save(filepath)
        
        # 保存统计信息
        stats_file = self.archive_dir / 'archive_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'track_to_person': self.track_to_person,
                'next_person_id': self.next_person_id
            }, f, indent=2)
        
        print(f"所有档案已保存到: {self.archive_dir}")
    
    def load_all(self):
        """加载所有档案"""
        # 加载统计信息
        stats_file = self.archive_dir / 'archive_stats.json'
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.stats = data['stats']
                self.track_to_person = data['track_to_person']
                self.next_person_id = data['next_person_id']
        
        # 加载所有档案文件
        for pkl_file in self.archive_dir.glob('person_*.pkl'):
            try:
                archive = PersonArchive.load(pkl_file)
                self.archives[archive.person_id] = archive
            except Exception as e:
                print(f"加载档案失败 {pkl_file}: {e}")
        
        print(f"已加载 {len(self.archives)} 个档案")
    
    def get_all_archives(self):
        """获取所有档案"""
        return self.archives
    
    def get_active_archives(self, last_seen_threshold=60):
        """
        获取活跃档案
        
        参数:
            last_seen_threshold: 最后出现时间的阈值（秒）
        """
        active = {}
        current_time = datetime.now().timestamp()
        
        for person_id, archive in self.archives.items():
            temporal = archive.get_temporal_vector()
            if temporal and current_time - temporal['last_seen'] < last_seen_threshold:
                active[person_id] = archive
        
        return active
    
    def get_statistics(self):
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算额外的统计
        total_duration = sum(
            a.get_temporal_vector()['duration'] 
            for a in self.archives.values() 
            if a.get_temporal_vector()
        )
        
        avg_duration = total_duration / len(self.archives) if len(self.archives) > 0 else 0
        
        stats['avg_duration'] = avg_duration
        stats['total_persons'] = len(self.archives)
        
        return stats
    
    def export_summary(self, filepath):
        """导出档案摘要"""
        summary = []
        
        for person_id, archive in self.archives.items():
            spatial = archive.get_spatial_vector()
            temporal = archive.get_temporal_vector()
            
            summary.append({
                'person_id': person_id,
                'created_at': archive.info['created_at'],
                'last_updated': archive.info['last_updated'],
                'total_appearances': archive.info['total_appearances'],
                'total_duration': archive.info['total_duration'],
                'current_position': spatial['current_position'] if spatial else None,
                'activity_range': spatial['activity_range'] if spatial else None,
                'first_seen': temporal['first_seen'] if temporal else None,
                'last_seen': temporal['last_seen'] if temporal else None
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"档案摘要已导出到: {filepath}")
