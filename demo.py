"""
教室人脸追踪系统演示脚本
展示如何使用系统处理视频和图像
"""
import sys
import cv2
import time
from pathlib import Path

# 导入主追踪器
from classroom_tracker import ClassroomFaceTracker
from utils.visualizer import TrackingVisualizer, ArchiveAnalyzer, create_tracking_video


def demo_image(image_path, output_path=None):
    """
    演示：处理单张图像
    
    参数:
        image_path: 输入图像路径
        output_path: 输出图像路径 (可选)
    """
    print("=" * 60)
    print("教室人脸追踪系统 - 图像演示")
    print("=" * 60)
    
    # 创建追踪器
    config = {
        'detector_conf_threshold': 0.3,
        'archive_dir': './demo_archives',
        'show_display': True
    }
    
    tracker = ClassroomFaceTracker(config)
    
    # 处理图像
    processed_frame, tracks = tracker.process_image(
        image_path,
        output_path=output_path,
        show_display=True
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("处理结果")
    print("=" * 60)
    print(f"检测到 {len(tracks)} 个人脸")
    
    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = track
        archive = tracker.archive_manager.get_archive_by_track_id(track_id)
        if archive:
            print(f"\n人员 {i+1}:")
            print(f"  ID: {archive.person_id}")
            print(f"  位置: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
            
            spatial = archive.get_spatial_vector()
            if spatial:
                print(f"  中心位置: ({spatial['current_position']['x']:.1f}, "
                      f"{spatial['current_position']['y']:.1f})")
            
            temporal = archive.get_temporal_vector()
            if temporal:
                print(f"  出现次数: {temporal['appearance_count']}")
    
    # 生成可视化
    print("\n生成可视化...")
    visualizer = TrackingVisualizer()
    
    # 获取档案
    archives = {}
    for track in tracks:
        track_id = int(track[4])
        archive = tracker.archive_manager.get_archive_by_track_id(track_id)
        if archive:
            archives[track_id] = archive
    
    # 绘制结果
    result_img = visualizer.draw_tracking_frame(
        processed_frame, tracks, archives,
        show_trajectory=True, show_info=True
    )
    
    # 保存结果
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"结果已保存: {output_path}")
    
    # 导出档案
    tracker.export_archives('./demo_output')
    
    # 生成分析报告
    analyzer = ArchiveAnalyzer(tracker.archive_manager)
    analyzer.generate_report('./demo_output/analysis_report.json')
    
    print("\n演示完成!")
    return tracker


def demo_video(video_path, output_path=None, max_frames=None):
    """
    演示：处理视频
    
    参数:
        video_path: 输入视频路径
        output_path: 输出视频路径 (可选)
        max_frames: 最大处理帧数 (用于快速演示)
    """
    print("=" * 60)
    print("教室人脸追踪系统 - 视频演示")
    print("=" * 60)
    
    # 创建追踪器
    config = {
        'detector_conf_threshold': 0.3,
        'archive_dir': './demo_archives',
        'show_display': True
    }
    
    tracker = ClassroomFaceTracker(config)
    
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
    
    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f}fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames / fps:.1f}秒")
    
    # 初始化视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\n输出视频: {output_path}")
    
    # 初始化可视化器
    visualizer = TrackingVisualizer(width, height)
    
    frame_count = 0
    start_time = time.time()
    
    print("\n开始处理...")
    print("按 'q' 退出\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 限制帧数 (用于演示)
            if max_frames and frame_count >= max_frames:
                print(f"\n已达到最大帧数限制 ({max_frames})")
                break
            
            # 计算时间戳
            timestamp = frame_count / fps
            
            # 处理帧
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
                processed_frame, tracks, archives,
                show_trajectory=True, show_info=True
            )
            
            # 写入输出视频
            if writer:
                writer.write(result_frame)
            
            # 显示结果
            cv2.imshow('Classroom Face Tracking', result_frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断")
                break
            
            frame_count += 1
            
            # 打印进度
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"进度: {progress:.1f}% | 帧: {frame_count}/{total_frames} | "
                      f"FPS: {current_fps:.1f} | 追踪人数: {len(tracks)}")
    
    finally:
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # 保存档案
        print("\n保存档案...")
        tracker.archive_manager.save_all()
        
        # 生成热力图
        print("生成热力图...")
        heatmap = visualizer.draw_heatmap(
            tracker.archive_manager.archives,
            (height, width),
            './demo_output/activity_heatmap.jpg'
        )
        
        # 生成分析报告
        print("生成分析报告...")
        analyzer = ArchiveAnalyzer(tracker.archive_manager)
        analyzer.generate_report('./demo_output/analysis_report.json')
        
        # 生成图表
        print("生成可视化图表...")
        analyzer.plot_timeline('./demo_output/timeline_analysis.png')
        analyzer.plot_trajectories((height, width), './demo_output/trajectories.png')
        
        # 打印统计
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"总帧数: {frame_count}")
        print(f"总时间: {elapsed:.1f}秒")
        print(f"平均FPS: {frame_count / elapsed:.1f}")
        print(f"档案数量: {len(tracker.archive_manager.archives)}")
        print(f"输出目录: ./demo_output/")
    
    return tracker


def demo_camera(camera_id=0, output_path=None):
    """
    演示：实时摄像头处理
    
    参数:
        camera_id: 摄像头ID
        output_path: 输出视频路径 (可选)
    """
    print("=" * 60)
    print("教室人脸追踪系统 - 摄像头演示")
    print("=" * 60)
    
    # 创建追踪器
    config = {
        'detector_conf_threshold': 0.3,
        'archive_dir': './demo_archives',
        'show_display': True
    }
    
    tracker = ClassroomFaceTracker(config)
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"无法打开摄像头: {camera_id}")
        return
    
    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0
    
    print(f"\n摄像头信息:")
    print(f"  分辨率: {width}x{height}")
    
    # 初始化视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化可视化器
    visualizer = TrackingVisualizer(width, height)
    
    frame_count = 0
    start_time = time.time()
    
    print("\n开始处理...")
    print("按 'q' 退出\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败")
                break
            
            # 计算时间戳
            timestamp = time.time() - start_time
            
            # 处理帧
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
                processed_frame, tracks, archives,
                show_trajectory=True, show_info=True
            )
            
            # 写入输出视频
            if writer:
                writer.write(result_frame)
            
            # 显示结果
            cv2.imshow('Classroom Face Tracking', result_frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户退出")
                break
            
            frame_count += 1
            
            # 打印FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"FPS: {current_fps:.1f} | 追踪人数: {len(tracks)}")
    
    finally:
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # 保存档案
        print("\n保存档案...")
        tracker.archive_manager.save_all()
        
        # 打印统计
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("演示完成!")
        print("=" * 60)
        print(f"总帧数: {frame_count}")
        print(f"总时间: {elapsed:.1f}秒")
        print(f"平均FPS: {frame_count / elapsed:.1f}")
        print(f"档案数量: {len(tracker.archive_manager.archives)}")
    
    return tracker


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='教室人脸追踪系统演示')
    parser.add_argument('--mode', '-m', type=str, required=True,
                       choices=['image', 'video', 'camera'],
                       help='运行模式: image/video/camera')
    parser.add_argument('--input', '-i', type=str,
                       help='输入路径 (图像或视频)')
    parser.add_argument('--output', '-o', type=str,
                       help='输出路径')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='摄像头ID (仅camera模式)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='最大处理帧数 (仅video模式)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path('./demo_output').mkdir(exist_ok=True)
    Path('./demo_archives').mkdir(exist_ok=True)
    
    if args.mode == 'image':
        if not args.input:
            print("错误: image模式需要指定--input参数")
            return
        demo_image(args.input, args.output)
    
    elif args.mode == 'video':
        if not args.input:
            print("错误: video模式需要指定--input参数")
            return
        demo_video(args.input, args.output, args.max_frames)
    
    elif args.mode == 'camera':
        demo_camera(args.camera_id, args.output)


if __name__ == '__main__':
    main()
