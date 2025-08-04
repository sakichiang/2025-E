# 2025-E
2025电赛e题视觉部分开源

一个基于OpenCV的实时目标检测和跟踪系统，专门用于检测A4纸等矩形目标，并通过串口将坐标信息发送给下位机。

## 功能特点

- **实时目标检测**：使用OpenCV检测摄像头中的矩形目标（如A4纸）
- **自适应曝光控制**：根据环境光线自动调节摄像头曝光，提高检测精度
- **串口通信**：支持多线程串口通信，可靠传输坐标数据
- **相机标定**：支持相机标定功能，消除镜头畸变
- **激光点模拟**：可以模拟激光点位置用于瞄准系统

## 项目结构

```
├── main.py           # 主程序入口
├── imgprocess.py     # 图像处理模块
├── uart.py           # 串口通信模块
├── calibration.py    # 相机标定工具
├── uarttest.py       # 串口测试程序
└── README.md         # 项目说明
```

## 依赖库

```bash
pip install opencv-python numpy pyserial
```

## 快速开始

### 1. 基本使用

```bash
python main.py
```

程序会自动检测可用的摄像头和串口，开始实时目标跟踪。

### 2. 串口测试

如果需要测试串口通信是否正常：

```bash
python uarttest.py
```

这会发送LED控制指令来验证串口连接。

### 3. 相机标定（可选）

为了获得更好的检测精度，建议先进行相机标定：

```bash
python calibration.py
```

需要准备一个12x9的棋盘格标定板，按照程序提示拍摄25张标定图片。

## 配置说明

### 图像处理参数（imgprocess.py）

```python
# 摄像头设置
FRAME_WIDTH = 640        # 图像宽度
FRAME_HEIGHT = 480       # 图像高度

# 检测参数
MIN_CONTOUR_BRIGHTNESS = 120    # 目标最小亮度
MIN_CONTOUR_AREA = 1000        # 轮廓最小面积
A4_ASPECT_RATIO = 1.414        # A4纸长宽比

# 激光点偏移
LASER_OFFSET_X_PIXELS = -20    # 水平偏移
LASER_OFFSET_Y_PIXELS = -15    # 垂直偏移
```

### 串口配置（main.py）

```python
SERIAL_PORT = '/dev/ttyS1'     # Linux: /dev/ttyS1, Windows: COM1
BAUD_RATE = 115200             # 波特率
COORDINATE_ADDRESS = 0x10      # 坐标数据地址
```

## 通信协议

串口使用自定义帧格式：

```
[0xA5][地址][数据长度][数据...][校验和][0x5A]
```

坐标数据格式（小端序）：
- 目标X坐标（2字节）
- 目标Y坐标（2字节）  
- 激光X坐标（2字节）
- 激光Y坐标（2字节）

## 常见问题

### 1. 找不到摄像头
- 检查摄像头是否正确连接
- 尝试修改 `CAMERA_INDEX` 参数
- 确认摄像头驱动已安装

### 2. 串口连接失败
- 检查串口设备路径是否正确
- 确认波特率设置匹配
- 检查串口是否被其他程序占用

### 3. 检测效果不好
- 确保目标是白色且与背景对比明显
- 调整 `MIN_CONTOUR_BRIGHTNESS` 参数
- 考虑进行相机标定
- 检查光线条件

### 4. struct.error错误
这通常是坐标值超出范围导致的，程序已经添加了范围限制，确保坐标在0-65535之间。

## 开发说明

### 添加新的检测算法

在 `imgprocess.py` 中的 `_process_frame` 方法里添加你的检测逻辑。

### 自定义串口协议

继承 `Uart` 类并重写 `_handle_message` 方法来实现自定义的消息处理。


## 贡献

欢迎提交Issue和Pull Request！



## 更新日志

- v1.0.0: 初始版本，支持基本的目标检测和串口通信
