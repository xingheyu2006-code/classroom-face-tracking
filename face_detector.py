"""
人脸检测模块
使用YOLOv8或OpenCV DNN进行人脸检测
"""
import cv2
import numpy as np
import torch
from pathlib import Path


class BaseFaceDetector:
    """
    人脸检测器基类
    """
    
    def __init__(self, conf_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, frame):
        """检测人脸，子类必须实现"""
        raise NotImplementedError
    
    def nms(self, boxes, scores):
        """非极大值抑制"""
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        return indices.flatten() if len(indices) > 0 else []


class OpenCVFaceDetector(BaseFaceDetector):
    """
    基于OpenCV DNN的人脸检测器
    不依赖外部库
    """
    
    def __init__(self, conf_threshold=0.5, iou_threshold=0.45):
        super().__init__(conf_threshold, iou_threshold)
        
        # 使用OpenCV的DNN人脸检测器
        # 模型文件路径
        self.prototxt_path = "models/deploy.prototxt"
        self.model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        # 尝试加载模型
        self.net = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 检查模型文件是否存在
            if Path(self.prototxt_path).exists() and Path(self.model_path).exists():
                self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
                print("OpenCV DNN人脸检测器加载成功")
            else:
                print("模型文件不存在，使用Haar级联分类器")
                self._load_haar_cascade()
        except Exception as e:
            print(f"DNN模型加载失败: {e}，使用Haar级联分类器")
            self._load_haar_cascade()
    
    def _load_haar_cascade(self):
        """加载Haar级联分类器作为备选"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        print("Haar级联分类器加载成功")
    
    def detect(self, frame):
        """
        检测图像中的人脸
        
        返回:
            detections: 检测框列表
        """
        if self.net is not None:
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_dnn(self, frame):
        """使用DNN检测"""
        h, w = frame.shape[:2]
        
        # 预处理
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        # 检测
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                results.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': 0
                })
        
        return results
    
    def _detect_haar(self, frame):
        """使用Haar级联检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Haar级联没有置信度，使用固定值
            results.append({
                'bbox': [float(x), float(y), float(x + w), float(y + h)],
                'confidence': 0.8,
                'class_id': 0
            })
        
        return results


class YOLOFaceDetector(BaseFaceDetector):
    """
    基于YOLOv8的人脸检测器
    """
    
    def __init__(self, model_path=None, conf_threshold=0.5, iou_threshold=0.45, 
                 device=None, img_size=640):
        super().__init__(conf_threshold, iou_threshold)
        
        self.img_size = img_size
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载YOLO模型
        self.model = None
        self._load_model(model_path)
    
    def _load_model(self, model_path):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO
            
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # 使用预训练的YOLOv8n
                print("加载YOLOv8n模型...")
                self.model = YOLO('yolov8n.pt')
            
            self.model.to(self.device)
            print("YOLO模型加载成功")
            
        except ImportError:
            print("ultralytics未安装，将使用OpenCV DNN检测器")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
    
    def detect(self, frame):
        """检测人脸"""
        if self.model is None:
            print("警告: YOLO模型未加载")
            return []
        
        # 使用YOLO进行检测
        results = self.model(frame, verbose=False, imgsz=self.img_size)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().item()
                class_id = int(box.cls[0].cpu().item())
                
                # 过滤低置信度检测
                if confidence < self.conf_threshold:
                    continue
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id
                }
                
                detections.append(detection)
        
        return detections


class ClassroomFaceDetector:
    """
    专门针对教室场景的人脸检测器
    自动选择可用的检测器
    """
    
    def __init__(self, 
                 model_path=None,
                 conf_threshold=0.3,
                 iou_threshold=0.45,
                 device=None,
                 img_size=640,
                 use_enhancement=True):
        """
        初始化教室人脸检测器
        """
        self.use_enhancement = use_enhancement
        
        # 首先尝试使用YOLO
        self.yolo_detector = None
        try:
            self.yolo_detector = YOLOFaceDetector(
                model_path, conf_threshold, iou_threshold, device, img_size
            )
        except Exception as e:
            print(f"YOLO检测器初始化失败: {e}")
        
        # 如果YOLO不可用，使用OpenCV
        if self.yolo_detector is None or self.yolo_detector.model is None:
            print("使用OpenCV DNN检测器")
            self.detector = OpenCVFaceDetector(conf_threshold, iou_threshold)
        else:
            self.detector = self.yolo_detector
        
        # 检测统计
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_detections_per_frame': 0
        }
    
    def enhance_image(self, frame):
        """
        图像增强，提高教室场景下的检测效果
        """
        if not self.use_enhancement:
            return frame
        
        # 转换为LAB颜色空间进行CLAHE增强
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect(self, frame, enhance=True):
        """
        检测教室中的人脸
        
        参数:
            frame: 输入图像
            enhance: 是否使用图像增强
            
        返回:
            detections: 检测结果
        """
        # 图像增强
        if enhance and self.use_enhancement:
            processed_frame = self.enhance_image(frame)
        else:
            processed_frame = frame
        
        # 检测人脸
        detections = self.detector.detect(processed_frame)
        
        # 更新统计
        self.detection_stats['total_frames'] += 1
        self.detection_stats['total_detections'] += len(detections)
        self.detection_stats['avg_detections_per_frame'] = \
            self.detection_stats['total_detections'] / self.detection_stats['total_frames']
        
        return detections
    
    def get_detection_stats(self):
        """获取检测统计信息"""
        return self.detection_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'avg_detections_per_frame': 0
        }
    
    def draw_detections(self, frame, detections, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制检测结果
        
        参数:
            frame: 输入图像
            detections: 检测结果列表
            color: 框的颜色 (B, G, R)
            thickness: 线宽
            
        返回:
            绘制后的图像
        """
        img = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            conf = det['confidence']
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制置信度
            label = f"Face: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(img, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
