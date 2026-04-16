"""
可视化工具模块
用于可视化跟踪结果和档案数据
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class TrackingVisualizer:
    """
    跟踪结果可视化器
    """
    
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        
    def draw_tracking_frame(self, frame, tracks, archives, show_trajectory=True, 
                           show_info=True, show_face=False):
        """
        绘制跟踪帧
        
        参数:
            frame: 输入图像
            tracks: 跟踪结果 [[x1, y1, x2, y2, track_id], ...]
            archives: 档案字典 {track_id: PersonArchive}
            show_trajectory: 是否显示轨迹
            show_info: 是否显示信息
            show_face: 是否显示人脸缩略图
            
        返回:
            绘制后的图像
        """
        img = frame.copy()
        h, w = img.shape[:2]
        
        # 为每个人分配颜色
        colors = {}
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取档案
            if track_id not in archives:
                continue
            
            archive = archives[track_id]
            person_id = archive.person_id
            
            # 生成颜色
            if person_id not in colors:
                colors[person_id] = self._generate_color(person_id)
            color = colors[person_id]
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制ID和信息
            if show_info:
                # 获取时间向量
                temporal = archive.get_temporal_vector()
                duration = temporal['duration'] if temporal else 0
                
                # 构建标签
                label = f"{person_id}"
                if duration > 0:
                    label += f" ({duration:.1f}s)"
                
                # 绘制标签背景
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = y1 - 5 if y1 - 5 > label_size[1] else y1 + label_size[1] + 5
                
                cv2.rectangle(img, (x1, label_y - label_size[1] - 5),
                            (x1 + label_size[0], label_y + 5), color, -1)
                cv2.putText(img, label, (x1, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 绘制轨迹
            if show_trajectory:
                spatial = archive.get_spatial_vector()
                if spatial and len(spatial['trajectory']) > 1:
                    trajectory = spatial['trajectory']
                    points = [(int(p['x']), int(p['y'])) for p in trajectory[-50:]]
                    
                    # 绘制轨迹线
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        line_color = tuple(int(c * alpha + 255 * (1 - alpha)) for c in color)
                        cv2.line(img, points[i-1], points[i], line_color, 1)
        
        return img
    
    def draw_heatmap(self, archives, frame_shape, output_path=None):
        """
        绘制活动热力图
        
        参数:
            archives: 档案字典
            frame_shape: 帧形状 (h, w)
            output_path: 输出路径
            
        返回:
            热力图图像
        """
        h, w = frame_shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # 累加所有位置
        for archive in archives.values():
            spatial = archive.get_spatial_vector()
            if spatial:
                for point in spatial['trajectory']:
                    x, y = int(point['x']), int(point['y'])
                    if 0 <= x < w and 0 <= y < h:
                        # 使用高斯核
                        self._add_gaussian(heatmap, x, y, sigma=20)
        
        # 归一化
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        if output_path:
            cv2.imwrite(output_path, heatmap_color)
        
        return heatmap_color
    
    def _add_gaussian(self, heatmap, x, y, sigma=10):
        """在热力图上添加高斯分布"""
        h, w = heatmap.shape
        
        # 生成高斯核
        size = int(3 * sigma)
        x_coords = np.arange(max(0, x - size), min(w, x + size + 1))
        y_coords = np.arange(max(0, y - size), min(h, y + size + 1))
        
        for yi in y_coords:
            for xi in x_coords:
                dist_sq = (xi - x) ** 2 + (yi - y) ** 2
                value = np.exp(-dist_sq / (2 * sigma ** 2)) * 255
                heatmap[yi, xi] += value
    
    def _generate_color(self, person_id):
        """为人员生成颜色"""
        try:
            num = int(person_id.split('_')[1])
        except:
            num = hash(person_id) % 1000
        
        hue = (num * 137) % 180
        saturation = 200 + (num % 55)
        value = 200 + (num % 55)
        
        hsv = np.uint8([[[hue, saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        
        return tuple(int(v) for v in bgr)


class ArchiveAnalyzer:
    """
    档案分析器
    分析档案数据并生成可视化报告
    """
    
    def __init__(self, archive_manager):
        self.archive_manager = archive_manager
    
    def analyze_attendance(self, time_range=None):
        """
        分析出勤情况
        
        参数:
            time_range: 时间范围 (start, end)
            
        返回:
            出勤统计
        """
        attendance = {}
        
        for person_id, archive in self.archive_manager.archives.items():
            temporal = archive.get_temporal_vector()
            if temporal is None:
                continue
            
            # 检查时间范围
            if time_range:
                start, end = time_range
                if temporal['last_seen'] < start or temporal['first_seen'] > end:
                    continue
            
            attendance[person_id] = {
                'first_seen': temporal['first_seen'],
                'last_seen': temporal['last_seen'],
                'duration': temporal['duration'],
                'appearances': temporal['appearance_count']
            }
        
        return attendance
    
    def analyze_attention(self):
        """
        分析注意力情况
        
        返回:
            注意力统计
        """
        attention_stats = {}
        
        for person_id, archive in self.archive_manager.archives.items():
            temporal = archive.get_temporal_vector()
            if temporal and 'attention_stats' in temporal:
                stats = temporal['attention_stats']
                if stats:
                    attention_stats[person_id] = stats
        
        return attention_stats
    
    def analyze_movement(self):
        """
        分析移动情况
        
        返回:
            移动统计
        """
        movement_stats = {}
        
        for person_id, archive in self.archive_manager.archives.items():
            spatial = archive.get_spatial_vector()
            if spatial:
                movement_stats[person_id] = {
                    'activity_range': spatial['activity_range'],
                    'velocity': spatial['velocity'],
                    'trajectory_length': len(spatial['trajectory'])
                }
        
        return movement_stats
    
    def generate_report(self, output_path):
        """
        生成分析报告
        
        参数:
            output_path: 输出路径
        """
        import json
        
        report = {
            'summary': {
                'total_persons': len(self.archive_manager.archives),
                'stats': self.archive_manager.get_statistics()
            },
            'attendance': self.analyze_attendance(),
            'attention': self.analyze_attention(),
            'movement': self.analyze_movement()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"分析报告已生成: {output_path}")
        return report
    
    def plot_timeline(self, output_path=None):
        """
        绘制时间线图表
        
        参数:
            output_path: 输出路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 出勤时间分布
        ax1 = axes[0, 0]
        durations = []
        for archive in self.archive_manager.archives.values():
            temporal = archive.get_temporal_vector()
            if temporal:
                durations.append(temporal['duration'])
        
        ax1.hist(durations, bins=20, edgecolor='black')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Number of Persons')
        ax1.set_title('Attendance Duration Distribution')
        
        # 2. 出现次数分布
        ax2 = axes[0, 1]
        appearances = []
        for archive in self.archive_manager.archives.values():
            temporal = archive.get_temporal_vector()
            if temporal:
                appearances.append(temporal['appearance_count'])
        
        ax2.hist(appearances, bins=20, edgecolor='black')
        ax2.set_xlabel('Appearance Count')
        ax2.set_ylabel('Number of Persons')
        ax2.set_title('Appearance Count Distribution')
        
        # 3. 活动范围分布
        ax3 = axes[1, 0]
        activity_areas = []
        for archive in self.archive_manager.archives.values():
            spatial = archive.get_spatial_vector()
            if spatial:
                activity_areas.append(spatial['activity_range']['area'])
        
        ax3.hist(activity_areas, bins=20, edgecolor='black')
        ax3.set_xlabel('Activity Area (pixels^2)')
        ax3.set_ylabel('Number of Persons')
        ax3.set_title('Activity Range Distribution')
        
        # 4. 人员活动热力图
        ax4 = axes[1, 1]
        # 这里可以绘制位置分布
        ax4.set_title('Position Distribution (placeholder)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"时间线图表已保存: {output_path}")
        
        return fig
    
    def plot_trajectories(self, frame_shape, output_path=None):
        """
        绘制所有人员的轨迹
        
        参数:
            frame_shape: 帧形状
            output_path: 输出路径
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        h, w = frame_shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Y轴反转
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Person Trajectories')
        
        # 为每个人绘制轨迹
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.archive_manager.archives)))
        
        for idx, (person_id, archive) in enumerate(self.archive_manager.archives.items()):
            spatial = archive.get_spatial_vector()
            if spatial and len(spatial['trajectory']) > 1:
                trajectory = spatial['trajectory']
                xs = [p['x'] for p in trajectory]
                ys = [p['y'] for p in trajectory]
                
                ax.plot(xs, ys, color=colors[idx % len(colors)], 
                       alpha=0.6, linewidth=1, label=person_id)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已保存: {output_path}")
        
        return fig


def create_tracking_video(tracker, video_path, output_path, 
                          show_trajectory=True, show_info=True):
    """
    创建带跟踪结果的视频
    
    参数:
        tracker: ClassroomFaceTracker实例
        video_path: 输入视频路径
        output_path: 输出视频路径
        show_trajectory: 是否显示轨迹
        show_info: 是否显示信息
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    visualizer = TrackingVisualizer(width, height)
    frame_count = 0
    
    print("正在生成跟踪视频...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / fps
        processed_frame, tracks = tracker.process_frame(frame, timestamp, frame_count)
        
        # 获取档案
        archives = {}
        for track in tracks:
            track_id = int(track[4])
            archive = tracker.archive_manager.get_archive_by_track_id(track_id)
            if archive:
                archives[track_id] = archive
        
        # 绘制结果
        result_frame = visualizer.draw_tracking_frame(
            frame, tracks, archives, 
            show_trajectory=show_trajectory, 
            show_info=show_info
        )
        
        writer.write(result_frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")
    
    cap.release()
    writer.release()
    
    print(f"跟踪视频已保存: {output_path}")
