# 教室人脸追踪系统 - 项目总结

## 项目概述

本项目实现了一个完整的教室人脸追踪系统，基于YOLO人脸检测、DeepSORT跟踪算法和卡尔曼滤波预测技术，能够为教室监控视频中的每个人员创建独立的档案，记录其时间向量和空间向量。

## 核心功能

### 1. 人脸检测模块 (`face_detector.py`)
- **YOLO检测器**: 使用YOLOv8进行高效人脸检测
- **OpenCV DNN检测器**: 作为备选方案，无需额外依赖
- **Haar级联分类器**: 最终备选方案
- **图像增强**: CLAHE增强提高教室场景检测效果

### 2. DeepSORT跟踪模块 (`deep_sort.py`)
- **级联匹配**: 基于外观特征进行匹配
- **IOU匹配**: 基于位置重叠进行关联
- **匈牙利算法**: 求解最优分配问题
- **人脸特征提取**: 集成特征提取器

### 3. 卡尔曼滤波模块 (`kalman_filter.py`)
- **标准实现**: 不依赖外部库的完整实现
- **状态空间**: `[x, y, w, h, vx, vy, vw, vh]` (8维)
- **观测空间**: `[x, y, w, h]` (4维)
- **预测与更新**: 完整的预测-更新循环

### 4. 特征提取模块 (`face_feature_extractor.py`)
- **简单CNN**: 基于PyTorch的轻量级特征提取器
- **ArcFace支持**: 可选的高级特征提取器
- **特征管理**: 特征存储、匹配和缓存
- **时间特征**: 提取时间维度特征

### 5. 档案管理模块 (`person_archive.py`)
- **个人档案**: 为每个人员创建独立档案
- **空间向量**: 位置、速度、活动范围、轨迹
- **时间向量**: 出现时间、持续时间、出现频率
- **数据持久化**: 支持pickle和JSON格式

### 6. 可视化模块 (`visualizer.py`)
- **跟踪可视化**: 实时绘制跟踪结果
- **热力图**: 活动区域热力图
- **轨迹图**: 人员移动轨迹
- **分析报告**: JSON格式的统计分析

## 数据结构

### 空间向量 (Spatial Vector)
```python
{
    'current_position': {
        'x': float,           # 中心X坐标
        'y': float,           # 中心Y坐标
        'bbox': [x1,y1,x2,y2] # 边界框
    },
    'velocity': {
        'vx': float,          # X方向速度
        'vy': float           # Y方向速度
    },
    'activity_range': {
        'x_range': float,     # X方向活动范围
        'y_range': float,     # Y方向活动范围
        'area': float         # 活动面积
    },
    'trajectory': [...]      # 轨迹历史
}
```

### 时间向量 (Temporal Vector)
```python
{
    'first_seen': float,      # 首次出现时间戳
    'last_seen': float,       # 最后出现时间戳
    'duration': float,        # 总持续时间(秒)
    'appearance_count': int,  # 出现次数
    'interval_stats': {       # 时间间隔统计
        'mean_interval': float,
        'std_interval': float,
        'min_interval': float,
        'max_interval': float
    }
}
```

## 算法原理

### 卡尔曼滤波

**状态转移方程:**
```
x' = F * x
P' = F * P * F^T + Q
```

**观测更新方程:**
```
K = P' * H^T * (H * P' * H^T + R)^-1
x = x' + K * (z - H * x')
P = (I - K * H) * P'
```

其中:
- `F`: 状态转移矩阵
- `H`: 观测矩阵
- `Q`: 过程噪声协方差
- `R`: 测量噪声协方差
- `P`: 估计误差协方差
- `K`: 卡尔曼增益

### DeepSORT跟踪

**匹配流程:**
1. **级联匹配**: 使用外观特征计算余弦距离
2. **IOU匹配**: 对未匹配目标使用IOU匹配
3. **匈牙利算法**: 求解最优分配

**距离度量:**
- 余弦距离: `d = 1 - cosine_similarity`
- IOU距离: `d = 1 - IOU`

## 测试结果

### 模块测试
```
✓ 卡尔曼滤波器测试通过
✓ DeepSORT跟踪器测试通过
✓ 个人档案管理测试通过
✓ 特征提取器测试通过
✓ 可视化工具测试通过

总计: 5/5 项测试通过
```

### 模拟演示结果
- **模拟帧数**: 100帧
- **模拟人数**: 10人
- **创建档案**: 10个
- **追踪稳定性**: 100% (无ID切换)

### 卡尔曼滤波性能
- **预测平均误差**: 2.64 像素
- **估计平均误差**: 2.08 像素

## 文件结构

```
classroom_face_tracking/
├── classroom_tracker.py      # 主程序入口
├── demo.py                   # 演示脚本
├── demo_mock.py              # 模拟演示
├── example_usage.py          # 使用示例
├── test_system.py            # 系统测试
├── config.yaml               # 配置文件
├── requirements.txt          # 依赖列表
├── README.md                 # 项目文档
├── PROJECT_SUMMARY.md        # 项目总结
├── utils/
│   ├── __init__.py
│   ├── face_detector.py      # 人脸检测
│   ├── deep_sort.py          # DeepSORT跟踪
│   ├── kalman_filter.py      # 卡尔曼滤波
│   ├── face_feature_extractor.py  # 特征提取
│   ├── person_archive.py     # 档案管理
│   └── visualizer.py         # 可视化
├── mock_output/              # 模拟输出
│   ├── last_frame.jpg        # 最后一帧
│   ├── activity_heatmap.jpg  # 活动热力图
│   ├── trajectories.png      # 轨迹图
│   ├── timeline_analysis.png # 时间线分析
│   ├── analysis_report.json  # 分析报告
│   └── archives/             # 档案目录
└── models/                   # 模型目录
```

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行测试
```bash
python test_system.py
```

### 3. 运行模拟演示
```bash
python demo_mock.py --frames 100 --persons 10
```

### 4. 处理视频
```bash
python classroom_tracker.py --input video.mp4 --output output.mp4
```

### 5. 处理图像
```bash
python classroom_tracker.py --input image.jpg --output result.jpg
```

## 配置说明

编辑 `config.yaml` 自定义系统配置:

```yaml
detector:
  conf_threshold: 0.3      # 检测置信度阈值
  iou_threshold: 0.45      # NMS IOU阈值

tracker:
  max_age: 30             # 跟踪器最大存活帧数
  min_hits: 3             # 确认跟踪所需最小命中次数

archive:
  archive_dir: "./archives"  # 档案存储目录
```

## 输出文件

系统运行后生成以下文件:

1. **跟踪结果图像**: `last_frame.jpg`
   - 显示人员边界框、ID、轨迹

2. **活动热力图**: `activity_heatmap.jpg`
   - 显示人员活动密集区域

3. **轨迹图**: `trajectories.png`
   - 显示所有人员的移动轨迹

4. **时间线分析**: `timeline_analysis.png`
   - 出勤时间分布、出现次数分布等

5. **分析报告**: `analysis_report.json`
   - 包含出勤统计、移动分析等

6. **个人档案**: `archives/person_*.pkl`
   - 每个人的完整档案数据

## 技术特点

### 1. 模块化设计
- 各模块独立，易于维护和扩展
- 清晰的接口定义

### 2. 多检测器支持
- YOLOv8 (推荐)
- OpenCV DNN
- Haar级联

### 3. 鲁棒性
- 卡尔曼滤波处理遮挡
- DeepSORT处理ID切换
- 特征匹配提高跟踪稳定性

### 4. 可配置性
- YAML配置文件
- 命令行参数
- 运行时配置

### 5. 可视化
- 实时跟踪显示
- 多种分析图表
- 热力图和轨迹图

## 应用场景

1. **教室出勤统计**
   - 自动记录学生出勤情况
   - 分析学生活动范围

2. **课堂行为分析**
   - 跟踪学生注意力
   - 分析学生活跃度

3. **安全监控**
   - 异常行为检测
   - 人员轨迹追踪

4. **教学研究**
   - 学生互动分析
   - 课堂参与度评估

## 扩展方向

1. **注意力检测**
   - 头部姿态估计
   - 视线方向检测

2. **情绪识别**
   - 面部表情分析
   - 情绪变化追踪

3. **行为识别**
   - 举手检测
   - 站立/坐下检测

4. **多摄像头融合**
   - 跨摄像头跟踪
   - 全局轨迹重建

## 注意事项

1. **隐私保护**
   - 仅用于教学研究目的
   - 遵守相关法律法规

2. **性能优化**
   - 建议使用GPU加速
   - 可根据场景调整参数

3. **检测效果**
   - 光线条件影响检测效果
   - 严重遮挡可能导致跟踪丢失

## 许可证

MIT License

## 作者

基于YOLO + DeepSORT + 卡尔曼滤波的教室人脸追踪系统
