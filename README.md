# 教室人脸追踪系统

基于YOLO人脸检测、DeepSORT跟踪、卡尔曼滤波预测的智能教室监控系统。系统能够为教室中的每个人员创建独立档案，记录其时间向量和空间向量。

## 功能特点

- **人脸检测**: 使用YOLOv8进行高效准确的人脸检测
- **多目标跟踪**: 基于DeepSORT算法实现稳定的人脸跟踪
- **卡尔曼滤波**: 预测目标运动轨迹，处理遮挡和短暂消失
- **人脸特征提取**: 提取人脸特征向量用于身份识别
- **个人档案管理**: 为每个人员创建独立档案，记录完整的时间-空间信息
- **可视化分析**: 生成热力图、轨迹图、时间线分析等

## 系统架构

```
教室人脸追踪系统
├── 人脸检测模块 (YOLO)
│   └── 教室场景优化
│   └── 图像增强
│
├── 目标跟踪模块 (DeepSORT)
│   ├── 级联匹配 (外观特征)
│   ├── IOU匹配 (位置重叠)
│   └── 卡尔曼滤波预测
│
├── 特征提取模块
│   ├── 人脸特征向量提取
│   ├── 时间特征提取
│   └── 特征匹配与管理
│
└── 档案管理模块
    ├── 个人档案创建
    ├── 空间向量记录
    ├── 时间向量记录
    └── 数据持久化
```

## 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 下载YOLO模型 (可选)
# 系统会自动下载默认模型
```

## 快速开始

### 1. 处理单张图像

```bash
python demo.py --mode image --input /path/to/image.jpg --output result.jpg
```

### 2. 处理视频文件

```bash
python demo.py --mode video --input /path/to/video.mp4 --output output.mp4
```

### 3. 实时摄像头

```bash
python demo.py --mode camera --camera-id 0
```

### 4. 使用主程序

```bash
# 处理视频
python classroom_tracker.py --input video.mp4 --output output.mp4

# 处理图像
python classroom_tracker.py --input image.jpg --output result.jpg

# 不显示实时画面 (后台运行)
python classroom_tracker.py --input video.mp4 --no-display
```

## 核心模块说明

### 1. 人脸检测器 (face_detector.py)

```python
from utils.face_detector import ClassroomFaceDetector

# 创建检测器
detector = ClassroomFaceDetector(
    model_path=None,  # 使用默认模型
    conf_threshold=0.3,
    device='cuda'
)

# 检测人脸
detections = detector.detect(frame)
```

### 2. DeepSORT跟踪器 (deep_sort.py)

```python
from utils.deep_sort import FaceDeepSORT

# 创建跟踪器
tracker = FaceDeepSORT(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

# 更新跟踪
tracks = tracker.update_with_faces(detections, frame, frame_id)
```

### 3. 卡尔曼滤波器 (kalman_filter.py)

```python
from utils.kalman_filter import KalmanFaceTracker

# 创建跟踪器
kf_tracker = KalmanFaceTracker(bbox)

# 预测
predicted_bbox = kf_tracker.predict()

# 更新
kf_tracker.update(bbox, timestamp)

# 获取空间向量
spatial_vector = kf_tracker.get_spatial_vector()

# 获取时间向量
temporal_vector = kf_tracker.get_temporal_vector()
```

### 4. 档案管理器 (person_archive.py)

```python
from utils.person_archive import ArchiveManager

# 创建档案管理器
archive_manager = ArchiveManager(archive_dir='./archives')

# 更新档案
archive, is_new = archive_manager.update_archive(
    track_id=track_id,
    timestamp=timestamp,
    frame_id=frame_id,
    bbox=bbox,
    face_feature=face_feature
)

# 获取档案
archive = archive_manager.get_archive_by_track_id(track_id)

# 获取空间向量
spatial_vector = archive.get_spatial_vector()

# 获取时间向量
temporal_vector = archive.get_temporal_vector()

# 保存所有档案
archive_manager.save_all()
```

## 数据结构

### 空间向量 (Spatial Vector)

```python
{
    'current_position': {
        'x': float,        # 中心X坐标
        'y': float,        # 中心Y坐标
        'bbox': [x1, y1, x2, y2]  # 边界框
    },
    'velocity': {
        'vx': float,       # X方向速度
        'vy': float        # Y方向速度
    },
    'activity_range': {
        'x_range': float,  # X方向活动范围
        'y_range': float,  # Y方向活动范围
        'area': float      # 活动面积
    },
    'trajectory': [      # 轨迹历史
        {'timestamp': float, 'x': float, 'y': float},
        ...
    ]
}
```

### 时间向量 (Temporal Vector)

```python
{
    'first_seen': float,        # 首次出现时间戳
    'last_seen': float,         # 最后出现时间戳
    'duration': float,          # 总持续时间(秒)
    'appearance_count': int,    # 出现次数
    'interval_stats': {         # 时间间隔统计
        'mean_interval': float,
        'std_interval': float,
        'min_interval': float,
        'max_interval': float
    },
    'attention_stats': {        # 注意力统计(可选)
        'mean_attention': float,
        'std_attention': float,
        'min_attention': float,
        'max_attention': float
    }
}
```

## 配置说明

编辑 `config.yaml` 文件来自定义系统配置:

```yaml
# 检测配置
detector:
  conf_threshold: 0.3      # 检测置信度阈值
  iou_threshold: 0.45      # NMS IOU阈值
  img_size: 640           # 输入图像大小

# 跟踪配置
tracker:
  max_age: 30             # 跟踪器最大存活帧数
  min_hits: 3             # 确认跟踪所需最小命中次数
  iou_threshold: 0.3      # IOU匹配阈值

# 档案配置
archive:
  archive_dir: "./archives"  # 档案存储目录
  max_face_samples: 10       # 最大人脸样本数
```

## 输出文件

系统运行后会生成以下文件:

```
archives/
├── person_000001.pkl       # 个人档案 (二进制)
├── person_000001.json      # 个人档案 (JSON)
├── person_000002.pkl
├── person_000002.json
├── ...
└── archive_stats.json      # 档案统计信息

demo_output/
├── analysis_report.json    # 分析报告
├── timeline_analysis.png   # 时间线分析图
├── trajectories.png        # 轨迹图
└── activity_heatmap.jpg    # 活动热力图
```

## 可视化工具

```python
from utils.visualizer import TrackingVisualizer, ArchiveAnalyzer

# 创建可视化器
visualizer = TrackingVisualizer()

# 绘制跟踪帧
result_frame = visualizer.draw_tracking_frame(
    frame, tracks, archives, 
    show_trajectory=True, 
    show_info=True
)

# 生成热力图
heatmap = visualizer.draw_heatmap(
    archives, 
    frame_shape, 
    'heatmap.jpg'
)

# 创建分析器
analyzer = ArchiveAnalyzer(archive_manager)

# 生成分析报告
analyzer.generate_report('report.json')

# 绘制时间线
analyzer.plot_timeline('timeline.png')

# 绘制轨迹
analyzer.plot_trajectories(frame_shape, 'trajectories.png')
```

## 算法原理

### 1. 人脸检测 (YOLO)

- 使用YOLOv8进行端到端的人脸检测
- 针对教室场景进行优化，降低置信度阈值以检测更多人脸
- 使用CLAHE图像增强提高检测效果

### 2. DeepSORT跟踪

DeepSORT算法包含两个阶段:

**阶段1: 级联匹配**
- 使用外观特征进行匹配
- 计算余弦距离
- 优先匹配最近出现过的目标

**阶段2: IOU匹配**
- 对未匹配的目标使用IOU匹配
- 基于位置重叠进行关联

### 3. 卡尔曼滤波

状态空间: `[x, y, w, h, vx, vy, vw, vh]`

- `x, y`: 边界框中心坐标
- `w, h`: 边界框宽高
- `vx, vy, vw, vh`: 对应的速度

预测步骤:
```
x' = F * x
P' = F * P * F^T + Q
```

更新步骤:
```
K = P' * H^T * (H * P' * H^T + R)^-1
x = x' + K * (z - H * x')
P = (I - K * H) * P'
```

### 4. 人脸特征提取

- 使用ArcFace或FaceNet提取128/512维特征向量
- L2归一化处理
- 使用余弦相似度进行匹配

## 性能优化

1. **GPU加速**: 自动检测并使用CUDA
2. **批量处理**: 支持批量图像检测
3. **特征缓存**: 缓存人脸特征避免重复计算
4. **轨迹压缩**: 限制轨迹历史长度

## 注意事项

1. **隐私保护**: 本系统仅用于教学研究目的，使用时需遵守相关法律法规
2. **光线条件**: 教室光线变化可能影响检测效果
3. **遮挡处理**: 严重遮挡可能导致跟踪丢失
4. **计算资源**: 建议使用GPU以获得更好的实时性能

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。
