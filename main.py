# main.py
# 主程序入口 - 集成图像处理和串口通信的实时目标跟踪系统
# 这个程序能够通过摄像头实时检测A4纸等矩形目标，并将坐标通过串口发送给下位机

import cv2
import time
import struct
import serial.tools.list_ports

# 导入我们自己写的模块
from imgprocess import FrameDetector, resize_for_display
from uart import Uart, MyUartHandler

# ==================== 基础配置 ====================
# 是否显示图形界面，调试时建议开启，实际运行时可以关闭提高性能
SHOW_GUI = False
# =====================================================

def find_available_port():
    """
    扫描系统中所有可用的串口
    返回第一个找到的串口，如果没找到就返回None
    """
    ports = serial.tools.list_ports.comports()
    print("正在扫描可用串口...")

    if not ports:
        print("  没找到任何串口，请检查设备连接")
        return None

    for i, port in enumerate(ports):
        print(f"  找到串口 [{i+1}]: {port.device} - {port.description}")

    return ports[0].device if ports else None

def main():
    """
    主程序循环
    负责：
    1. 初始化串口通信
    2. 启动摄像头和目标检测
    3. 实时发送坐标数据给下位机
    4. 显示帧率和检测结果
    """
    # 串口配置 - 根据你的硬件修改这些参数
    SERIAL_PORT = '/dev/ttyS1'  # Linux下的串口，Windows改成COM1这样的
    BAUD_RATE = 115200          # 波特率要和下位机保持一致
    COORDINATE_ADDRESS = 0x10   # 坐标数据的地址标识

    comm = None
    try:
        # 先尝试自动找串口
        auto_port = find_available_port()
        if auto_port:
            SERIAL_PORT = auto_port
            print(f"自动选择串口: {SERIAL_PORT}")

        print(f"正在连接串口 {SERIAL_PORT}，波特率 {BAUD_RATE}...")
        comm = MyUartHandler(SERIAL_PORT, BAUD_RATE, byte_delay=0.001)
        comm.start()
        print("串口连接成功!")

    except Exception as e:
        print(f"串口连接失败: {e}")
        print("程序将继续运行，但不会发送数据")
        comm = None

    # 初始化图像检测器
    print("\n正在启动摄像头...")
    try:
        detector = FrameDetector()
    except IOError as e:
        print(f"摄像头启动失败: {e}")
        return

    print("系统启动完成! 按 'q' 键退出")

    # 用来计算帧率的变量
    prev_time = 0
    current_time = 0

    try:
        while True:
            # 开始计时
            current_time = time.time()

            # 获取当前帧的检测结果
            target_center, laser_point, result_frame = detector.get_current_data()

            if result_frame is None:
                print("摄像头读取失败，退出程序")
                break

            # 计算当前帧率
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            fps_text = f"FPS: {int(fps)}"

            # 如果检测到目标并且串口连接正常，就发送坐标数据
            if comm and target_center and laser_point:
                try:
                    # 获取坐标并转换为整数
                    tx, ty = int(target_center[0]), int(target_center[1])
                    lx, ly = int(laser_point[0]), int(laser_point[1])

                    # 限制坐标范围在0-65535之间（16位无符号整数的范围）
                    # 这样做是为了防止struct.pack报错
                    tx = max(0, min(65535, tx))
                    ty = max(0, min(65535, ty))
                    lx = max(0, min(65535, lx))
                    ly = max(0, min(65535, ly))

                    # 打包成4个16位无符号整数发送
                    coord_data = struct.pack('<HHHH', tx, ty, lx, ly)
                    comm.send_message(COORDINATE_ADDRESS, coord_data)

                except struct.error as e:
                    print(f"\n坐标数据打包失败: {e}")
                    print(f"问题坐标: 目标({tx}, {ty}), 激光({lx}, {ly})")
                except Exception as e:
                    print(f"\n串口发送失败: {e}")

            # 在控制台显示当前状态
            if target_center:
                status = f"目标: {target_center}, 激光: {laser_point}, {fps_text}"
                print(f"\r{status}  ", end="")
            else:
                print(f"\r正在搜索目标... {fps_text}                    ", end="")

            # 在图像上显示帧率信息
            cv2.putText(result_frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 如果开启了GUI就显示图像
            if SHOW_GUI:
                display_frame = resize_for_display(result_frame, max_width=1280, max_height=720)
                cv2.imshow("Control View - FPS Test", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭程序...")

    finally:
        # 清理资源
        print("\n正在释放资源...")
        if comm:
            comm.stop()
        detector.release()
        cv2.destroyAllWindows()
        print("程序已安全退出")

def main1():
    """
    串口通信测试函数
    用来测试串口是否能正常发送数据，以及是否存在延时问题
    """
    port = '/dev/ttyS1'
    baud_rate = 115200
    com = None

    try:
        com = MyUartHandler(port, baud_rate, byte_delay=0.001)
        com.start()
        print(f"串口测试开始: {port}, {baud_rate}")

        # 发送5组测试数据
        for i in range(5):
            # 模拟发送坐标数据：目标(300+i, 300) 激光(200+i, 250)
            test_data = struct.pack('<HHHH', 300+i, 300, 200+i, 250)
            print(f"发送第{i+1}组测试数据: {test_data.hex()}")
            com.send_message(0x10, test_data, use_delay=True)
            time.sleep(0.5)

    except serial.SerialException as e:
        print(f"串口错误: {e}")
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"测试过程出错: {e}")
    finally:
        if com:
            try:
                com.stop()
                print("串口连接已关闭")
            except Exception as e:
                print(f"关闭串口时出错: {e}")

if __name__ == '__main__':
    main()