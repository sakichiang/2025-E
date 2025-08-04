#!/usr/bin/env python3
"""
简单的串口测试脚本
用于验证串口通信和多线程发送功能
"""

import time
import serial.tools.list_ports
from uart import MyUartHandler

def test_serial_communication():
    """测试串口通信功能"""
    print("=== 串口通信测试 ===")

    # 列出可用串口
    ports = serial.tools.list_ports.comports()
    print("可用串口:")
    if not ports:
        print("  未找到任何串口!")
        return False

    for i, port in enumerate(ports):
        print(f"  [{i+1}] {port.device}: {port.description}")

    # 选择串口
    if len(ports) == 0:
        print("没有可用的串口，请检查硬件连接")
        return False

    # 使用第一个可用串口进行测试
    test_port = ports[0].device
    print(f"\n使用串口: {test_port}")

    try:
        # 创建串口对象，设置2ms字节延时
        comm = MyUartHandler(test_port, 115200, byte_delay=0.002)
        comm.start()
        print("串口通信已启动")

        # 测试1: 发送简单数据
        print("\n--- 测试1: 发送简单数据 ---")
        for i in range(5):
            data = bytes([i])
            print(f"发送数据 {i}: {data.hex()}")
            comm.send_message(0x01, data, use_delay=True)
            time.sleep(0.2)

        # 测试2: 发送复杂数据包
        print("\n--- 测试2: 发送复杂数据包 ---")
        test_data = b"Hello"
        print(f"发送字符串: {test_data}")
        comm.send_message(0x02, test_data, use_delay=True)
        time.sleep(0.5)

        # 测试3: 不使用延时发送
        print("\n--- 测试3: 无延时发送 ---")
        fast_data = bytes([0xFF, 0xAA, 0x55])
        print(f"快速发送: {fast_data.hex()}")
        comm.send_message(0x03, fast_data, use_delay=False)
        time.sleep(0.5)

        # 检查发送队列状态
        queue_size = comm.get_send_queue_size()
        print(f"\n当前发送队列大小: {queue_size}")

        # 等待发送完成
        print("等待发送完成...")
        time.sleep(2)

        print("\n测试完成，正在关闭串口...")
        comm.stop()
        print("串口已关闭")

        return True

    except Exception as e:
        print(f"串口测试失败: {e}")
        return False

def test_byte_delay_settings():
    """测试字节延时设置功能"""
    print("\n=== 字节延时设置测试 ===")

    try:
        # 使用虚拟串口进行测试（如果没有真实串口）
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("没有可用串口，跳过延时测试")
            return

        test_port = ports[0].device
        comm = MyUartHandler(test_port, 115200, byte_delay=0.001)
        comm.start()

        # 测试不同的延时设置
        delays = [0.001, 0.005, 0.01]  # 1ms, 5ms, 10ms

        for delay in delays:
            comm.set_byte_delay(delay)
            test_data = bytes([0x12, 0x34, 0x56])
            print(f"使用 {delay*1000:.1f}ms 延时发送: {test_data.hex()}")
            comm.send_message(0x10, test_data, use_delay=True)
            time.sleep(0.5)

        comm.stop()
        print("延时测试完成")

    except Exception as e:
        print(f"延时测试失败: {e}")

if __name__ == "__main__":
    print("开始串口功能测试...")

    # 基本通信测试
    success = test_serial_communication()

    if success:
        # 延时功能测试
        test_byte_delay_settings()
        print("\n所有测试完成!")
    else:
        print("\n基本测试失败，请检查串口连接")

    print("\n按任意键退出...")
    input()
