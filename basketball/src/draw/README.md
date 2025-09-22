# 篮球轨迹可视化节点

此节点订阅 `ball_trajectory` 话题，将篮球的三维轨迹投影到原始视频上并保存为新视频。

## 绘制模式配置

节点支持多种绘制模式的配置，可以通过ROS2参数进行设置：

### 参数列表

- `trail_length` (int, 默认: 170)
  - 轨迹长度：只显示最近的n个轨迹点
  - 范围：1-∞

- `enable_color_fade` (bool, 默认: true)
  - 是否启用颜色渐变：近处的点颜色较深，远处的点颜色较浅
  - 效果：轨迹从深到浅渐变

- `enable_size_variation` (bool, 默认: true)
  - 是否启用大小变化：近处的点较大，远处的点较小
  - 效果：模拟透视效果

- `base_radius` (int, 默认: 30)
  - 基准半径：用于计算点的大小
  - 仅在启用大小变化时有效

- `max_trail_length` (int, 默认: 300)
  - 最大轨迹长度限制：防止轨迹过长影响性能

- `enable_interpolation_color` (bool, 默认: false)
  - 是否启用插帧颜色区分：同一帧内的插帧点使用不同颜色
  - 效果：原始检测点用正常颜色，插帧点用反色

### 使用方法

#### 1. 启动时设置参数

```bash
ros2 run draw draw --ros-args -p trail_length:=100 -p enable_color_fade:=false
```

#### 2. 动态修改参数

```bash
# 查看当前参数
ros2 param list /trajectory_visualizer

# 修改参数
ros2 param set /trajectory_visualizer trail_length 50
ros2 param set /trajectory_visualizer enable_color_fade false
```

### 示例配置

#### 默认配置（推荐）

```bash
ros2 run draw draw
```

- 轨迹长度：170点
- 颜色渐变：开启
- 大小变化：开启

#### 简化模式（性能优化）

```bash
ros2 run draw draw --ros-args -p trail_length:=50 -p enable_color_fade:=false -p enable_size_variation:=false
```

- 轨迹长度：50点
- 颜色渐变：关闭
- 大小变化：关闭

#### 插帧颜色区分模式

```bash
ros2 run draw draw --ros-args -p enable_interpolation_color:=true -p trail_length:=100
```

- 插帧颜色区分：开启
- 轨迹长度：100点
- 原始检测点：正常颜色
- 插帧点：反色

#### 详细轨迹模式

```bash
ros2 run draw draw --ros-args -p trail_length:=300 -p max_trail_length:=500
```

- 轨迹长度：300点
- 最大长度：500点

#### 开启插帧颜色区分

```bash
ros2 run draw draw --ros-args -p enable_interpolation_color:=true
```
