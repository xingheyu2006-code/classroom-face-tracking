"""
人脸特征提取模块
用于提取人脸的特征向量，用于身份识别和匹配
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class FaceFeatureExtractor:
    """
    人脸特征提取器基类
    """
    
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def extract(self, face_image):
        """
        提取人脸特征向量
        
        参数:
            face_image: 人脸图像 (numpy array)
            
        返回:
            feature: 特征向量 (numpy array)
        """
        raise NotImplementedError
    
    def preprocess(self, face_image, target_size=(112, 112)):
        """
        预处理人脸图像
        
        参数:
            face_image: 输入图像
            target_size: 目标大小
            
        返回:
            预处理后的图像
        """
        # 调整大小
        if face_image.shape[:2] != target_size:
            face_image = cv2.resize(face_image, target_size)
        
        # 归一化
        face_image = face_image.astype(np.float32) / 255.0
        
        # 减均值、除标准差 (ImageNet统计)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_image = (face_image - mean) / std
        
        return face_image


class SimpleCNNFeatureExtractor(FaceFeatureExtractor):
    """
    简单的CNN特征提取器
    不依赖外部库，使用PyTorch实现
    """
    
    def __init__(self, device=None, feature_dim=128):
        super().__init__(device)
        self.feature_dim = feature_dim
        
        # 构建简单的CNN
        self.model = self._build_model().to(self.device)
        self.model.eval()
    
    def _build_model(self):
        """构建CNN模型"""
        class SimpleCNN(nn.Module):
            def __init__(self, feature_dim=128):
                super().__init__()
                self.conv = nn.Sequential(
                    # 第一层
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    # 第二层
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    # 第三层
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    
                    # 第四层
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.fc = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, feature_dim)
                )
            
            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                # L2归一化
                x = nn.functional.normalize(x, p=2, dim=1)
                return x
        
        return SimpleCNN(self.feature_dim)
    
    def extract(self, face_image):
        """提取特征向量"""
        # 预处理
        face_tensor = self.preprocess(face_image, target_size=(112, 112))
        face_tensor = torch.from_numpy(face_tensor).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.float().to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(face_tensor)
        
        return features.cpu().numpy().flatten()


class ArcFaceFeatureExtractor(FaceFeatureExtractor):
    """
    基于ArcFace的人脸特征提取器
    尝试使用insightface或facenet-pytorch
    """
    
    def __init__(self, model_path=None, device=None, network='r50'):
        """
        初始化ArcFace特征提取器
        
        参数:
            model_path: 模型路径
            device: 计算设备
            network: 网络架构 ('r50', 'r100', 'mobileface')
        """
        super().__init__(device)
        self.network = network
        
        # 尝试加载模型
        self.model = self._load_model(model_path)
        
        # 如果外部模型加载失败，使用简单CNN
        if self.model is None:
            print("使用简单CNN作为特征提取器")
            self.model = SimpleCNNFeatureExtractor(device=device)
    
    def _load_model(self, model_path):
        """加载ArcFace模型"""
        # 尝试使用insightface
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            app = FaceAnalysis(name='buffalo_l', root='./models')
            app.prepare(ctx_id=0 if self.device.type == 'cuda' else -1, det_size=(640, 640))
            
            self.face_app = app
            print("ArcFace模型加载成功")
            return app
            
        except ImportError:
            print("insightface未安装")
        
        # 尝试使用facenet-pytorch
        try:
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print("使用InceptionResnetV1作为特征提取器")
            return model
        except ImportError:
            print("facenet-pytorch未安装")
        
        return None
    
    def extract(self, face_image):
        """提取特征向量"""
        if hasattr(self, 'face_app'):
            return self._extract_with_insightface(face_image)
        elif hasattr(self.model, 'extract'):
            return self.model.extract(face_image)
        else:
            # 使用facenet-pytorch
            return self._extract_with_facenet(face_image)
    
    def _extract_with_insightface(self, face_image):
        """使用insightface提取特征"""
        faces = self.face_app.get(face_image)
        
        if len(faces) > 0:
            return faces[0].embedding
        else:
            return np.zeros(512)
    
    def _extract_with_facenet(self, face_image):
        """使用facenet-pytorch提取特征"""
        # 预处理
        face_tensor = self.preprocess(face_image, target_size=(160, 160))
        face_tensor = torch.from_numpy(face_tensor).permute(2, 0, 1).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            embedding = self.model(face_tensor)
        
        return embedding.cpu().numpy().flatten()


class FaceFeatureManager:
    """
    人脸特征管理器
    管理所有人脸特征的存储和匹配
    """
    
    def __init__(self, feature_extractor=None, similarity_threshold=0.6):
        """
        初始化特征管理器
        
        参数:
            feature_extractor: 特征提取器实例
            similarity_threshold: 相似度阈值
        """
        self.similarity_threshold = similarity_threshold
        
        # 特征提取器
        if feature_extractor is None:
            # 使用默认的特征提取器 (简单CNN)
            self.feature_extractor = SimpleCNNFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # 特征库: person_id -> list of features
        self.feature_database = {}
        
        # 特征缓存
        self.feature_cache = {}
    
    def extract_feature(self, face_image):
        """提取单个人脸的特征"""
        return self.feature_extractor.extract(face_image)
    
    def add_feature(self, person_id, feature):
        """添加特征到数据库"""
        if person_id not in self.feature_database:
            self.feature_database[person_id] = []
        
        self.feature_database[person_id].append(feature)
        
        # 更新缓存
        self._update_cache(person_id)
    
    def _update_cache(self, person_id):
        """更新特征缓存"""
        if person_id in self.feature_database:
            features = self.feature_database[person_id]
            if len(features) > 0:
                # 计算平均特征
                avg_feature = np.mean(features, axis=0)
                # L2归一化
                avg_feature = avg_feature / (np.linalg.norm(avg_feature) + 1e-6)
                self.feature_cache[person_id] = avg_feature
    
    def match_face(self, feature, top_k=5):
        """
        匹配人脸特征
        
        参数:
            feature: 查询特征
            top_k: 返回前k个匹配结果
            
        返回:
            matches: [(person_id, similarity), ...]
        """
        if len(self.feature_cache) == 0:
            return []
        
        similarities = []
        
        for person_id, cached_feature in self.feature_cache.items():
            # 计算余弦相似度
            similarity = self._cosine_similarity(feature, cached_feature)
            similarities.append((person_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 过滤低相似度
        matches = [(pid, sim) for pid, sim in similarities if sim >= self.similarity_threshold]
        
        return matches[:top_k]
    
    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a_norm = a / (np.linalg.norm(a) + 1e-6)
        b_norm = b / (np.linalg.norm(b) + 1e-6)
        return np.dot(a_norm, b_norm)
    
    def get_person_features(self, person_id):
        """获取指定人的所有特征"""
        return self.feature_database.get(person_id, [])
    
    def get_person_average_feature(self, person_id):
        """获取指定人的平均特征"""
        return self.feature_cache.get(person_id, None)
    
    def remove_person(self, person_id):
        """删除指定人的特征"""
        if person_id in self.feature_database:
            del self.feature_database[person_id]
        if person_id in self.feature_cache:
            del self.feature_cache[person_id]
    
    def save_database(self, filepath):
        """保存特征数据库"""
        import pickle
        data = {
            'feature_database': self.feature_database,
            'feature_cache': self.feature_cache
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"特征数据库已保存到: {filepath}")
    
    def load_database(self, filepath):
        """加载特征数据库"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.feature_database = data['feature_database']
        self.feature_cache = data['feature_cache']
        print(f"特征数据库已加载: {filepath}")
    
    def get_all_person_ids(self):
        """获取所有人员ID"""
        return list(self.feature_database.keys())
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        total_persons = len(self.feature_database)
        total_features = sum(len(features) for features in self.feature_database.values())
        avg_features_per_person = total_features / total_persons if total_persons > 0 else 0
        
        return {
            'total_persons': total_persons,
            'total_features': total_features,
            'avg_features_per_person': avg_features_per_person
        }


class TemporalFeatureExtractor:
    """
    时间特征提取器
    提取人脸的时间维度特征
    """
    
    def __init__(self, window_size=30):
        """
        初始化
        
        参数:
            window_size: 时间窗口大小（帧数）
        """
        self.window_size = window_size
        
        # 存储每个人员的时间序列特征
        self.temporal_features = {}
    
    def add_frame_feature(self, person_id, feature, timestamp):
        """
        添加一帧的特征
        
        参数:
            person_id: 人员ID
            feature: 特征向量
            timestamp: 时间戳
        """
        if person_id not in self.temporal_features:
            self.temporal_features[person_id] = []
        
        self.temporal_features[person_id].append({
            'feature': feature,
            'timestamp': timestamp
        })
        
        # 保持窗口大小
        if len(self.temporal_features[person_id]) > self.window_size:
            self.temporal_features[person_id] = \
                self.temporal_features[person_id][-self.window_size:]
    
    def extract_temporal_vector(self, person_id):
        """
        提取时间向量
        
        返回:
            temporal_vector: 时间特征向量
        """
        if person_id not in self.temporal_features:
            return None
        
        features_list = self.temporal_features[person_id]
        
        if len(features_list) == 0:
            return None
        
        # 提取时间特征
        features = np.array([f['feature'] for f in features_list])
        timestamps = [f['timestamp'] for f in features_list]
        
        # 统计特征
        temporal_vector = {
            # 特征均值
            'mean_feature': np.mean(features, axis=0),
            # 特征方差
            'var_feature': np.var(features, axis=0),
            # 特征变化率
            'feature_change_rate': self._compute_change_rate(features),
            # 时间跨度
            'time_span': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            # 出现频率
            'appearance_count': len(features_list),
            # 时间序列稳定性
            'stability': self._compute_stability(features)
        }
        
        return temporal_vector
    
    def _compute_change_rate(self, features):
        """计算特征变化率"""
        if len(features) < 2:
            return 0.0
        
        changes = []
        for i in range(1, len(features)):
            change = np.linalg.norm(features[i] - features[i-1])
            changes.append(change)
        
        return np.mean(changes)
    
    def _compute_stability(self, features):
        """计算特征稳定性"""
        if len(features) < 2:
            return 1.0
        
        # 计算相邻特征之间的相似度
        similarities = []
        for i in range(1, len(features)):
            sim = self._cosine_similarity(features[i], features[i-1])
            similarities.append(sim)
        
        return np.mean(similarities)
    
    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a_norm = a / (np.linalg.norm(a) + 1e-6)
        b_norm = b / (np.linalg.norm(b) + 1e-6)
        return np.dot(a_norm, b_norm)
