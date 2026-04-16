"""
教室人脸追踪系统 - 模拟演示
使用模拟数据展示系统功能
"""
import cv2
import numpy as np
import time
from pathlib import Path

# 导入系统模块
from utils.kalman_filter import KalmanFaceTracker
from utils.deep_sort import DeepSORT
from utils.person_archive import PersonArchive, ArchiveManager
from utils.face_feature_extractor import SimpleCNNFeatureExtractor, FaceFeatureManager
from utils.visualizer import TrackingVisualizer, ArchiveAnalyzer


def simulate_classroom_tracking(num_frames=100, num_persons=10):
    """
    模拟教室追踪场景
    
    参数:
        num_frames: 模拟帧数
        num_persons: 模拟人数
    """
    print("=" * 70)
    print("教室人脸追踪系统 - 模拟演示")
    print("=" * 70)
    
    # 创建输出目录
    output_path = Path('./mock_output')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化组件
    print("\n[1/4] 初始化组件...")
    
    # 创建DeepSORT跟踪器
    tracker = DeepSORT(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    )
    
    # 创建特征管理器
    feature_manager = FaceFeatureManager()
    
    # 创建档案管理器
    archive_manager = ArchiveManager(archive_dir=str(output_path / 'archives'))
    
    # 创建可视化器
    visualizer = TrackingVisualizer(1920, 1080)
    
    print("  ✓ 组件初始化完成")
    
    # 模拟人员初始位置
    print(f"\n[2/4] 模拟 {num_persons} 名学生在教室中的活动...")
    
    np.random.seed(42)
    
    # 初始化人员位置 (教室座位布局)
    persons = {}
    for i in range(num_persons):
        # 教室座位布局 (5排 x 2列)
        row = i // 2
        col = i % 2
        
        base_x = 400 + col * 600 + np.random.randint(-50, 50)
        base_y = 200 + row * 150 + np.random.randint(-30, 30)
        
        persons[i] = {
            'base_x': base_x,
            'base_y': base_y,
            'x': base_x,
            'y': base_y,
            'vx': 0,
            'vy': 0,
            'track_id': None
        }
    
    # 模拟多帧追踪
    print(f"\n[3/4] 运行模拟 ({num_frames} 帧)...")
    
    frame_history = []
    
    for frame_id in range(num_frames):
        # 创建空白帧
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 240
        
        # 绘制教室背景
        # 绘制课桌
        for i in range(num_persons):
            row = i // 2
            col = i % 2
            desk_x = 350 + col * 600
            desk_y = 150 + row * 150
            cv2.rectangle(frame, (desk_x, desk_y), (desk_x + 200, desk_y + 100), (180, 180, 180), -1)
            cv2.rectangle(frame, (desk_x, desk_y), (desk_x + 200, desk_y + 100), (100, 100, 100), 2)
        
        # 模拟人员移动
        detections = []
        for person_id, person in persons.items():
            # 添加随机移动 (模拟学生的小动作)
            person['vx'] += np.random.randn() * 0.5
            person['vy'] += np.random.randn() * 0.5
            
            # 阻尼
            person['vx'] *= 0.9
            person['vy'] *= 0.9
            
            # 限制在座位附近
            dx = person['base_x'] - person['x']
            dy = person['base_y'] - person['y']
            person['vx'] += dx * 0.01
            person['vy'] += dy * 0.01
            
            # 更新位置
            person['x'] += person['vx']
            person['y'] += person['vy']
            
            # 生成检测框
            x = person['x']
            y = person['y']
            w = 80 + np.random.randint(-10, 10)
            h = 100 + np.random.randint(-10, 10)
            
            bbox = [x - w/2, y - h/2, x + w/2, y + h/2]
            detections.append(bbox)
            
            # 绘制人员
            color = visualizer._generate_color(f"person_{person_id:06d}")
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        
        detections = np.array(detections)
        
        # 更新跟踪器
        timestamp = frame_id * 0.033  # 假设30fps
        tracks = tracker.update(detections, frame_id=frame_id)
        
        # 更新档案
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            bbox = [x1, y1, x2, y2]
            
            # 生成模拟人脸特征
            face_feature = np.random.randn(128)
            face_feature = face_feature / np.linalg.norm(face_feature)
            
            # 更新档案
            archive, is_new = archive_manager.update_archive(
                track_id=track_id,
                timestamp=timestamp,
                frame_id=frame_id,
                bbox=bbox,
                face_feature=face_feature
            )
        
        # 绘制跟踪结果
        archives = {}
        for track in tracks:
            track_id = int(track[4])
            archive = archive_manager.get_archive_by_track_id(track_id)
            if archive:
                archives[track_id] = archive
        
        result_frame = visualizer.draw_tracking_frame(
            frame, tracks, archives,
            show_trajectory=True, show_info=True
        )
        
        # 添加统计信息
        stats_text = [
            f"Frame: {frame_id + 1}/{num_frames}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracks)}",
            f"Archives: {len(archive_manager.archives)}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(result_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 25
        
        frame_history.append(result_frame)
        
        # 打印进度
        if (frame_id + 1) % 20 == 0:
            print(f"  进度: {frame_id + 1}/{num_frames} 帧, "
                  f"追踪人数: {len(tracks)}")
    
    print("  ✓ 模拟完成")
    
    # 保存结果
    print(f"\n[4/4] 保存结果...")
    
    # 保存最后一帧
    last_frame_path = output_path / 'last_frame.jpg'
    cv2.imwrite(str(last_frame_path), frame_history[-1])
    print(f"  ✓ 最后一帧已保存: {last_frame_path}")
    
    # 生成热力图
    heatmap = visualizer.draw_heatmap(
        archive_manager.archives,
        (1080, 1920),
        str(output_path / 'activity_heatmap.jpg')
    )
    print(f"  ✓ 热力图已保存")
    
    # 保存档案
    archive_manager.save_all()
    print(f"  ✓ 档案已保存")
    
    # 生成分析报告
    analyzer = ArchiveAnalyzer(archive_manager)
    analyzer.generate_report(str(output_path / 'analysis_report.json'))
    print(f"  ✓ 分析报告已生成")
    
    # 生成可视化图表
    analyzer.plot_timeline(str(output_path / 'timeline_analysis.png'))
    print(f"  ✓ 时间线分析图已生成")
    
    analyzer.plot_trajectories((1080, 1920), str(output_path / 'trajectories.png'))
    print(f"  ✓ 轨迹图已生成")
    
    # 打印结果摘要
    print("\n" + "=" * 70)
    print("模拟结果摘要")
    print("=" * 70)
    print(f"总帧数: {num_frames}")
    print(f"模拟人数: {num_persons}")
    print(f"创建档案数: {len(archive_manager.archives)}")
    print(f"输出目录: {output_path}")
    
    # 打印档案详情
    print("\n" + "=" * 70)
    print("档案详情")
    print("=" * 70)
    
    for person_id, archive in archive_manager.archives.items():
        print(f"\n{person_id}:")
        
        spatial = archive.get_spatial_vector()
        if spatial:
            print(f"  空间向量:")
            print(f"    - 当前位置: ({spatial['current_position']['x']:.1f}, "
                  f"{spatial['current_position']['y']:.1f})")
            print(f"    - 速度: ({spatial['velocity']['vx']:.2f}, "
                  f"{spatial['velocity']['vy']:.2f}) 像素/帧")
            print(f"    - 活动范围: X={spatial['activity_range']['x_range']:.1f}, "
                  f"Y={spatial['activity_range']['y_range']:.1f}")
            print(f"    - 轨迹点数: {len(spatial['trajectory'])}")
        
        temporal = archive.get_temporal_vector()
        if temporal:
            print(f"  时间向量:")
            print(f"    - 持续时间: {temporal['duration']:.2f} 秒")
            print(f"    - 出现次数: {temporal['appearance_count']}")
            if temporal['interval_stats']:
                print(f"    - 平均间隔: {temporal['interval_stats']['mean_interval']:.3f} 秒")
    
    return archive_manager


def demonstrate_kalman_filter():
    """
    演示卡尔曼滤波器的预测能力
    """
    print("\n" + "=" * 70)
    print("卡尔曼滤波器演示")
    print("=" * 70)
    
    # 创建卡尔曼滤波器
    initial_bbox = [100, 100, 200, 200]
    kf = KalmanFaceTracker(initial_bbox)
    
    print(f"\n初始边界框: {initial_bbox}")
    print(f"初始位置: {kf.get_position()}")
    
    # 模拟目标运动
    print("\n模拟目标匀速运动:")
    
    true_positions = []
    predicted_positions = []
    estimated_positions = []
    
    for i in range(20):
        # 真实位置 (匀速运动)
        true_x = 150 + i * 5
        true_y = 150 + i * 3
        true_positions.append((true_x, true_y))
        
        # 预测
        predicted = kf.predict()
        predicted_pos = kf.get_position()
        predicted_positions.append(predicted_pos)
        
        # 模拟观测 (添加噪声)
        noise_x = np.random.randn() * 3
        noise_y = np.random.randn() * 3
        observed_bbox = [true_x - 50 + noise_x, true_y - 50 + noise_y,
                        true_x + 50 + noise_x, true_y + 50 + noise_y]
        
        # 更新
        kf.update(observed_bbox, timestamp=i * 0.033)
        estimated_pos = kf.get_position()
        estimated_positions.append(estimated_pos)
        
        if i % 5 == 0:
            print(f"  帧 {i}:")
            print(f"    真实位置: ({true_x:.1f}, {true_y:.1f})")
            print(f"    预测位置: ({predicted_pos[0]:.1f}, {predicted_pos[1]:.1f})")
            print(f"    估计位置: ({estimated_pos[0]:.1f}, {estimated_pos[1]:.1f})")
    
    # 计算误差
    pred_errors = [np.sqrt((t[0]-p[0])**2 + (t[1]-p[1])**2) 
                   for t, p in zip(true_positions, predicted_positions)]
    est_errors = [np.sqrt((t[0]-e[0])**2 + (t[1]-e[1])**2) 
                  for t, e in zip(true_positions, estimated_positions)]
    
    print(f"\n预测平均误差: {np.mean(pred_errors):.2f} 像素")
    print(f"估计平均误差: {np.mean(est_errors):.2f} 像素")
    
    # 获取空间和时间向量
    spatial = kf.get_spatial_vector()
    temporal = kf.get_temporal_vector()
    
    print(f"\n空间向量:")
    print(f"  活动范围: X={spatial['activity_range']['x_range']:.1f}, "
          f"Y={spatial['activity_range']['y_range']:.1f}")
    
    print(f"\n时间向量:")
    print(f"  出现次数: {temporal['appearance_count']}")
    print(f"  持续时间: {temporal['duration']:.3f} 秒")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='教室人脸追踪系统 - 模拟演示')
    parser.add_argument('--frames', '-f', type=int, default=100,
                       help='模拟帧数')
    parser.add_argument('--persons', '-p', type=int, default=10,
                       help='模拟人数')
    parser.add_argument('--kalman-only', action='store_true',
                       help='仅演示卡尔曼滤波器')
    
    args = parser.parse_args()
    
    if args.kalman_only:
        demonstrate_kalman_filter()
    else:
        # 运行完整模拟
        simulate_classroom_tracking(args.frames, args.persons)
        
        # 演示卡尔曼滤波器
        demonstrate_kalman_filter()
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
