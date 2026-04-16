"""
教室人脸追踪系统主程序
整合YOLO人脸检测、DeepSORT跟踪、卡尔曼滤波预测
以及个人档案管理
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict

# 导入自定义模块
from utils.face_detector import ClassroomFaceDetector
from utils.deep_sort import FaceDeepSORT
from utils.kalman_filter import KalmanFaceTracker
from utils.face_feature_extractor import FaceFeatureManager, TemporalFeatureExtractor
from utils.person_archive import ArchiveManager


class ClassroomFaceTracker:
    """
    教室人脸追踪系统
    
    主要功能：
    1. 人脸检测 (YOLO)
    2. 人脸跟踪 (DeepSORT + 卡尔曼滤波)
    3. 人脸特征提取
    4. 个人档案管理 (时间向量和空间向量)
    """
    
    def __init__(self, config=None):
        """
        初始化追踪系统
        
        参数:
            config: 配置字典
        """
        # 默认配置
        self.config = {
            # 检测配置
            'detector_model': None,
            'detector_conf_threshold': 0.3,
            'detector_iou_threshold': 0.45,
            'detector_img_size': 640,
            
            # 跟踪配置
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'max_cosine_distance': 0.2,
            
            # 特征提取配置
            'feature_similarity_threshold': 0.6,
            
            # 档案配置
            'archive_dir': './archives',
            
            # 输出配置
            'save_video': False,
            'output_video_path': './output.mp4',
            'show_display': True,
            
            # 设备配置
            'device': None  # None表示自动选择
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        print("=" * 60)
        print("教室人脸追踪系统初始化")
        print("=" * 60)
        
        # 初始化人脸检测器
        print("\n[1/5] 初始化人脸检测器...")
        self.face_detector = ClassroomFaceDetector(
            model_path=self.config['detector_model'],
            conf_threshold=self.config['detector_conf_threshold'],
            iou_threshold=self.config['detector_iou_threshold'],
            device=self.config['device'],
            img_size=self.config['detector_img_size'],
            use_enhancement=True
        )
        print("人脸检测器初始化完成")
        
        # 初始化特征管理器
        print("\n[2/5] 初始化特征管理器...")
        self.feature_manager = FaceFeatureManager(
            similarity_threshold=self.config['feature_similarity_threshold']
        )
        print("特征管理器初始化完成")
        
        # 初始化DeepSORT跟踪器
        print("\n[3/5] 初始化DeepSORT跟踪器...")
        self.tracker = FaceDeepSORT(
            max_age=self.config['max_age'],
            min_hits=self.config['min_hits'],
            iou_threshold=self.config['iou_threshold'],
            max_cosine_distance=self.config['max_cosine_distance'],
            face_feature_extractor=self.feature_manager
        )
        print("DeepSORT跟踪器初始化完成")
        
        # 初始化时间特征提取器
        print("\n[4/5] 初始化时间特征提取器...")
        self.temporal_extractor = TemporalFeatureExtractor(window_size=30)
        print("时间特征提取器初始化完成")
        
        # 初始化档案管理器
        print("\n[5/5] 初始化档案管理器...")
        self.archive_manager = ArchiveManager(
            archive_dir=self.config['archive_dir']
        )
        print("档案管理器初始化完成")
        
        # 统计信息
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
        # 跟踪结果
        self.tracking_results = []
        
        print("\n" + "=" * 60)
        print("系统初始化完成，准备开始追踪")
        print("=" * 60)
    
    def process_frame(self, frame, timestamp=None, frame_id=None):
        """
        处理单帧图像
        
        参数:
            frame: 输入图像 (BGR格式)
            timestamp: 时间戳 (秒)
            frame_id: 帧ID
            
        返回:
            processed_frame: 处理后的图像
            tracks: 跟踪结果
        """
        if timestamp is None:
            timestamp = time.time()
        if frame_id is None:
            frame_id = self.frame_count
        
        # 记录开始时间
        frame_start_time = time.time()
        
        # 步骤1: 人脸检测
        detections = self.face_detector.detect(frame, enhance=True)
        
        # 步骤2: 人脸跟踪
        tracks = self.tracker.update_with_faces(detections, frame, frame_id, timestamp)
        
        # 步骤3: 更新档案
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            bbox = [x1, y1, x2, y2]
            
            # 提取人脸特征
            face_feature = None
            face_image = None
            
            # 裁剪人脸区域
            h, w = frame.shape[:2]
            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
            x2_int, y2_int = min(w, int(x2)), min(h, int(y2))
            
            if x2_int > x1_int and y2_int > y1_int:
                face_image = frame[y1_int:y2_int, x1_int:x2_int]
                
                if face_image.size > 0:
                    try:
                        face_feature = self.feature_manager.extract_feature(face_image)
                    except Exception as e:
                        print(f"特征提取失败: {e}")
            
            # 更新档案
            archive, is_new = self.archive_manager.update_archive(
                track_id=track_id,
                timestamp=timestamp,
                frame_id=frame_id,
                bbox=bbox,
                face_feature=face_feature,
                face_image=face_image ,
                attention_score=None
            )
            
            # 更新时间特征
            if face_feature is not None:
                self.temporal_extractor.add_frame_feature(
                    archive.person_id, face_feature, timestamp
                )
        
        # 计算FPS
        frame_time = time.time() - frame_start_time
        self.fps = 1.0 / frame_time if frame_time > 0 else 0
        
        self.frame_count += 1
        
        # 绘制结果
        processed_frame = self._draw_results(frame, tracks, detections)
        
        return processed_frame, tracks
    
    def _draw_results(self, frame, tracks, detections):
        """
        绘制跟踪结果
        """
        img = frame.copy()
        h, w = img.shape[:2]
        
        # 绘制检测框 (半透明)
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # 绘制跟踪框和ID
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取对应的人员ID
            archive = self.archive_manager.get_archive_by_track_id(track_id)
            if archive:
                person_id = archive.person_id
                
                # 生成颜色 (基于person_id)
                color = self._get_color_from_id(person_id)
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 绘制ID标签
                label = f"ID: {person_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + 20
                
                cv2.rectangle(img, (x1, label_y - label_size[1] - 5),
                            (x1 + label_size[0], label_y + 5), color, -1)
                cv2.putText(img, label, (x1, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 绘制轨迹
                spatial = archive.get_spatial_vector()
                if spatial and len(spatial['trajectory']) > 1:
                    points = [(int(p['x']), int(p['y'])) for p in spatial['trajectory'][-30:]]
                    for i in range(1, len(points)):
                        cv2.line(img, points[i-1], points[i], color, 1)
        
        # 绘制统计信息
        stats_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {self.fps:.1f}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracks)}",
            f"Archives: {len(self.archive_manager.archives)}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return img
    
    def _get_color_from_id(self, person_id):
        """根据person_id生成颜色"""
        # 从person_id中提取数字
        try:
            num = int(person_id.split('_')[1])
        except:
            num = hash(person_id) % 1000
        
        # 生成HSV颜色
        hue = (num * 137) % 180  # 使用黄金角度
        saturation = 200 + (num % 55)
        value = 200 + (num % 55)
        
        # 转换为BGR
        hsv = np.uint8([[[hue, saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        
        return tuple(int(v) for v in bgr)
    
    def process_video(self, video_path, output_path=None, show_display=True):
        """
        处理视频文件
        
        参数:
            video_path: 输入视频路径
            output_path: 输出视频路径 (可选)
            show_display: 是否显示实时画面
        """
        print(f"\n开始处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height} @ {fps:.2f}fps, 总帧数: {total_frames}")
        
        # 初始化视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出视频将保存到: {output_path}")
        
        self.start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 计算时间戳
                timestamp = self.frame_count / fps if fps > 0 else time.time()
                
                # 处理帧
                processed_frame, tracks = self.process_frame(frame, timestamp, self.frame_count)
                
                # 写入输出视频
                if writer:
                    writer.write(processed_frame)
                
                # 显示画面
                if show_display:
                    cv2.imshow('Classroom Face Tracking', processed_frame)
                    
                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("用户中断")
                        break
                
                # 打印进度
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"进度: {progress:.1f}% ({self.frame_count}/{total_frames}), "
                          f"FPS: {self.fps:.1f}, 追踪人数: {len(tracks)}")
        
        finally:
            # 释放资源
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # 保存档案
            print("\n保存档案...")
            self.archive_manager.save_all()
            
            # 打印统计
            elapsed = time.time() - self.start_time
            print(f"\n处理完成!")
            print(f"总帧数: {self.frame_count}")
            print(f"总时间: {elapsed:.1f}秒")
            print(f"平均FPS: {self.frame_count / elapsed:.1f}")
            print(f"档案数量: {len(self.archive_manager.archives)}")
    
    def process_image(self, image_path, output_path=None, show_display=True):
        """
        处理单张图像
        
        参数:
            image_path: 输入图像路径
            output_path: 输出图像路径 (可选)
            show_display: 是否显示结果
        """
        print(f"\n处理图像: {image_path}")
        
        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 处理图像
        timestamp = time.time()
        processed_frame, tracks = self.process_frame(frame, timestamp, 0)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"结果已保存到: {output_path}")
        
        # 显示结果
        if show_display:
            cv2.imshow('Classroom Face Tracking', processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # 保存档案
        self.archive_manager.save_all()
        
        print(f"检测到 {len(tracks)} 个人脸")
        
        return processed_frame, tracks
    
    def get_tracking_summary(self):
        """获取跟踪摘要"""
        summary = {
            'total_frames': self.frame_count,
            'total_archives': len(self.archive_manager.archives),
            'archive_stats': self.archive_manager.get_statistics(),
            'detection_stats': self.face_detector.get_detection_stats()
        }
        return summary
    
    def export_archives(self, output_dir):
        """导出所有档案"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 导出JSON摘要
        summary_file = output_path / 'archive_summary.json'
        self.archive_manager.export_summary(summary_file)
        
        # 保存所有档案
        self.archive_manager.save_all()
        
        print(f"档案已导出到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='教室人脸追踪系统')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入视频或图像路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出路径')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时画面')
    parser.add_argument('--archive-dir', type=str, default='./archives',
                       help='档案存储目录')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='检测置信度阈值')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'detector_conf_threshold': args.conf_threshold,
        'archive_dir': args.archive_dir,
        'device': args.device,
        'show_display': not args.no_display
    }
    
    # 创建追踪器
    tracker = ClassroomFaceTracker(config)
    
    # 判断输入类型
    input_path = Path(args.input)
    
    if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        # 视频文件
        tracker.process_video(
            str(input_path),
            output_path=args.output,
            show_display=not args.no_display
        )
    else:
        # 图像文件
        tracker.process_image(
            str(input_path),
            output_path=args.output,
            show_display=not args.no_display
        )
    
    # 打印摘要
    summary = tracker.get_tracking_summary()
    print("\n" + "=" * 60)
    print("跟踪摘要")
    print("=" * 60)
    print(f"总帧数: {summary['total_frames']}")
    print(f"档案数量: {summary['total_archives']}")
    print(f"平均检测数: {summary['detection_stats']['avg_detections_per_frame']:.2f}")


if __name__ == '__main__':
    main()
