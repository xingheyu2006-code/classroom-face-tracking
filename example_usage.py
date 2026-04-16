"""
教室人脸追踪系统 - 使用示例
展示如何处理教室监控图像
"""
import cv2
import numpy as np
import time
from pathlib import Path

# 导入系统模块
from classroom_tracker import ClassroomFaceTracker
from utils.visualizer import TrackingVisualizer, ArchiveAnalyzer


def process_classroom_image(image_path, output_dir='./example_output'):
    """
    处理教室监控图像
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
    """
    print("=" * 70)
    print("教室人脸追踪系统 - 图像处理示例")
    print("=" * 70)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 配置系统
    config = {
        'detector_conf_threshold': 0.25,  # 教室场景使用较低的阈值
        'archive_dir': str(output_path / 'archives'),
        'show_display': False
    }
    
    # 创建追踪器
    print("\n[1/5] 初始化追踪系统...")
    tracker = ClassroomFaceTracker(config)
    
    # 读取图像
    print(f"\n[2/5] 读取图像: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    print(f"  图像尺寸: {frame.shape[1]}x{frame.shape[0]}")
    
    # 处理图像
    print("\n[3/5] 处理图像...")
    timestamp = time.time()
    processed_frame, tracks = tracker.process_frame(frame, timestamp, 0)
    
    print(f"  检测到 {len(tracks)} 个人脸")
    
    # 获取档案信息
    print("\n[4/5] 获取档案信息...")
    archives = {}
    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = track
        archive = tracker.archive_manager.get_archive_by_track_id(track_id)
        
        if archive:
            archives[track_id] = archive
            
            print(f"\n  人员 {i+1}:")
            print(f"    ID: {archive.person_id}")
            print(f"    边界框: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
            
            # 空间向量
            spatial = archive.get_spatial_vector()
            if spatial:
                print(f"    中心位置: ({spatial['current_position']['x']:.1f}, "
                      f"{spatial['current_position']['y']:.1f})")
                print(f"    活动范围: X={spatial['activity_range']['x_range']:.1f}, "
                      f"Y={spatial['activity_range']['y_range']:.1f}")
            
            # 时间向量
            temporal = archive.get_temporal_vector()
            if temporal:
                print(f"    出现次数: {temporal['appearance_count']}")
    
    # 可视化结果
    print("\n[5/5] 生成可视化结果...")
    
    # 创建可视化器
    visualizer = TrackingVisualizer(frame.shape[1], frame.shape[0])
    
    # 绘制跟踪结果
    result_frame = visualizer.draw_tracking_frame(
        processed_frame, tracks, archives,
        show_trajectory=True, show_info=True
    )
    
    # 保存结果
    result_path = output_path / 'tracking_result.jpg'
    cv2.imwrite(str(result_path), result_frame)
    print(f"  ✓ 跟踪结果已保存: {result_path}")
    
    # 生成热力图
    heatmap = visualizer.draw_heatmap(
        archives,
        frame.shape,
        str(output_path / 'activity_heatmap.jpg')
    )
    print(f"  ✓ 热力图已保存")
    
    # 保存档案
    print("\n保存档案...")
    tracker.archive_manager.save_all()
    
    # 生成分析报告
    print("生成分析报告...")
    analyzer = ArchiveAnalyzer(tracker.archive_manager)
    analyzer.generate_report(str(output_path / 'analysis_report.json'))
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"检测到人脸数: {len(tracks)}")
    print(f"创建档案数: {len(tracker.archive_manager.archives)}")
    print(f"输出目录: {output_path}")
    
    return tracker


def print_spatial_vectors(tracker):
    """
    打印所有人员的空间向量
    """
    print("\n" + "=" * 70)
    print("空间向量详情")
    print("=" * 70)
    
    for person_id, archive in tracker.archive_manager.archives.items():
        spatial = archive.get_spatial_vector()
        
        if spatial:
            print(f"\n{person_id}:")
            print(f"  当前位置:")
            print(f"    - X: {spatial['current_position']['x']:.2f}")
            print(f"    - Y: {spatial['current_position']['y']:.2f}")
            print(f"    - 边界框: {spatial['current_position']['bbox']}")
            
            print(f"  速度:")
            print(f"    - VX: {spatial['velocity']['vx']:.2f} 像素/帧")
            print(f"    - VY: {spatial['velocity']['vy']:.2f} 像素/帧")
            
            print(f"  活动范围:")
            print(f"    - X范围: {spatial['activity_range']['x_range']:.2f} 像素")
            print(f"    - Y范围: {spatial['activity_range']['y_range']:.2f} 像素")
            print(f"    - 面积: {spatial['activity_range']['area']:.2f} 像素²")
            
            print(f"  轨迹点数: {len(spatial['trajectory'])}")


def print_temporal_vectors(tracker):
    """
    打印所有人员的时间向量
    """
    print("\n" + "=" * 70)
    print("时间向量详情")
    print("=" * 70)
    
    for person_id, archive in tracker.archive_manager.archives.items():
        temporal = archive.get_temporal_vector()
        
        if temporal:
            print(f"\n{person_id}:")
            print(f"  首次出现: {temporal['first_seen']:.3f}")
            print(f"  最后出现: {temporal['last_seen']:.3f}")
            print(f"  持续时间: {temporal['duration']:.3f} 秒")
            print(f"  出现次数: {temporal['appearance_count']}")
            
            if temporal['interval_stats']:
                print(f"  时间间隔统计:")
                print(f"    - 平均间隔: {temporal['interval_stats']['mean_interval']:.3f} 秒")
                print(f"    - 标准差: {temporal['interval_stats']['std_interval']:.3f} 秒")
                print(f"    - 最小间隔: {temporal['interval_stats']['min_interval']:.3f} 秒")
                print(f"    - 最大间隔: {temporal['interval_stats']['max_interval']:.3f} 秒")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='教室人脸追踪系统 - 使用示例')
    parser.add_argument('--input', '-i', type=str, 
                       default='/mnt/okcomputer/upload/image.png',
                       help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, 
                       default='./example_output',
                       help='输出目录')
    parser.add_argument('--show-vectors', action='store_true',
                       help='显示向量的详细信息')
    
    args = parser.parse_args()
    
    # 处理图像
    tracker = process_classroom_image(args.input, args.output)
    
    if tracker and args.show_vectors:
        # 打印空间向量
        print_spatial_vectors(tracker)
        
        # 打印时间向量
        print_temporal_vectors(tracker)
    
    print("\n" + "=" * 70)
    print("示例运行完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
