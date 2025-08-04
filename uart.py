# uart.py
# 串口通信模块 - 支持多线程的可靠数据传输
# 这个模块实现了一个自定义的串口协议，能够：
# 1. 发送和接收带校验和的数据帧
# 2. 支持异步发送（队列机制）
# 3. 可以设置字节间延时来适应不同的下位机
# 4. 自动处理数据帧的封装和解析

import serial
import threading
import time
import struct
import collections
import queue

# 协议定义
FRAME_START = 0xA5  # 帧头标识
FRAME_END = 0x5A    # 帧尾标识
MIN_FRAME_LENGTH = 5  # 最小帧长度（帧头+地址+长度+校验+帧尾）

class FrameError(Exception):
    """数据帧格式错误"""
    pass

class ChecksumError(Exception):
    """校验和错误，通常表示数据传输过程中出现了问题"""
    def __init__(self, message, frame=None):
        super().__init__(message)
        self.frame = frame

class Uart:
    """
    多线程串口通信类

    协议格式: [0xA5][地址][数据长度][数据...][校验和][0x5A]

    特点：
    - 支持可变长度数据（最多255字节）
    - 自动计算和验证校验和
    - 多线程收发，不会阻塞主程序
    - 支持字节间延时，适应慢速下位机
    """

    def __init__(self, port, baudrate=9600, timeout=1, byte_delay=0.001):
        """
        初始化串口

        参数:
        - port: 串口名称，如 'COM3' 或 '/dev/ttyUSB0'
        - baudrate: 波特率，常用 9600, 115200
        - timeout: 读取超时时间
        - byte_delay: 发送时每个字节间的延时（秒）
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
        except serial.SerialException as e:
            print(f"打开串口 {port} 失败: {e}")
            raise

        self.read_buffer = bytearray()  # 接收缓冲区
        self.is_running = False
        self.read_thread = None
        self.send_thread = None
        self.lock = threading.Lock()

        # 发送队列和设置
        self.send_queue = queue.Queue()
        self.byte_delay = byte_delay

    def start(self):
        """启动收发线程"""
        if not self.is_running:
            self.is_running = True

            # 启动接收线程
            self.read_thread = threading.Thread(target=self._read_serial)
            self.read_thread.daemon = True
            self.read_thread.start()

            # 启动发送线程
            self.send_thread = threading.Thread(target=self._send_worker)
            self.send_thread.daemon = True
            self.send_thread.start()

            print("串口收发线程已启动")

    def stop(self):
        """停止通信并关闭串口"""
        self.is_running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join()
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join()
        if self.ser.is_open:
            self.ser.close()
            print("串口已关闭")

    def _calculate_checksum(self, address, data_len, data_bytes):
        """
        计算校验和
        把地址、数据长度、所有数据字节加起来，取低8位
        """
        checksum = address + data_len
        for byte in data_bytes:
            checksum += byte
        return checksum & 0xFF

    def send_message(self, address, data_payload, use_delay=True):
        """
        异步发送消息（推荐使用）

        参数:
        - address: 功能地址 (0-255)
        - data_payload: 要发送的数据 (bytes类型)
        - use_delay: 是否使用字节间延时
        """
        if not isinstance(data_payload, bytes):
            raise TypeError("数据必须是bytes类型")
        if not (0 <= address <= 255):
            raise ValueError("地址必须在0-255之间")

        data_len = len(data_payload)
        if data_len > 255:
            raise ValueError("数据长度不能超过255字节")

        # 计算校验和
        checksum = self._calculate_checksum(address, data_len, data_payload)

        # 组装完整的数据帧
        header = bytearray([FRAME_START, address, data_len])
        trailer = bytearray([checksum, FRAME_END])
        frame = header + data_payload + trailer

        # 加入发送队列
        self.send_queue.put((frame, use_delay))

    def send_message_sync(self, address, data_payload, use_delay=True):
        """
        同步发送消息（会阻塞）
        适合需要立即发送的场合
        """
        if not isinstance(data_payload, bytes):
            raise TypeError("数据必须是bytes类型")
        if not (0 <= address <= 255):
            raise ValueError("地址必须在0-255之间")

        data_len = len(data_payload)
        if data_len > 255:
            raise ValueError("数据长度不能超过255字节")

        checksum = self._calculate_checksum(address, data_len, data_payload)

        header = bytearray([FRAME_START, address, data_len])
        trailer = bytearray([checksum, FRAME_END])
        frame = header + data_payload + trailer

        # 直接发送
        self._send_frame_with_delay(frame, use_delay)

    def _send_worker(self):
        """发送线程的工作函数，从队列中取数据帧发送"""
        while self.is_running:
            try:
                # 从队列中取一个待发送的帧，超时1秒
                frame, use_delay = self.send_queue.get(timeout=1)
                self._send_frame_with_delay(frame, use_delay)
                self.send_queue.task_done()
            except queue.Empty:
                continue  # 队列空了就继续等
            except Exception as e:
                print(f"发送线程出错: {e}")

    def _send_frame_with_delay(self, frame, use_delay=True):
        """
        实际发送数据帧的函数

        参数:
        - frame: 要发送的完整数据帧
        - use_delay: 是否在每个字节间加延时
        """
        with self.lock:
            if self.ser.is_open:
                if use_delay and self.byte_delay > 0:
                    # 逐字节发送，每个字节间有延时
                    # 这对一些慢速的单片机很有帮助
                    for byte in frame:
                        self.ser.write(bytes([byte]))
                        time.sleep(self.byte_delay)
                else:
                    # 一次性发送整个帧
                    self.ser.write(frame)

    def set_byte_delay(self, delay_seconds):
        """
        设置字节间延时
        参数: delay_seconds - 延时时间（秒），例如0.001表示1毫秒
        """
        self.byte_delay = delay_seconds
        print(f"字节间延时设置为 {delay_seconds*1000:.1f}毫秒")

    def get_send_queue_size(self):
        """获取当前发送队列中待发送的消息数量"""
        return self.send_queue.qsize()

    def clear_send_queue(self):
        """清空发送队列"""
        while not self.send_queue.empty():
            try:
                self.send_queue.get_nowait()
                self.send_queue.task_done()
            except queue.Empty:
                break
        print("发送队列已清空")

    def _read_serial(self):
        """
        接收线程的工作函数
        持续从串口读取数据并解析成完整的数据帧
        """
        while self.is_running:
            try:
                # 从串口读取数据
                if self.ser.in_waiting > 0:
                    data = self.ser.read(self.ser.in_waiting)
                    self.read_buffer.extend(data)

                # 解析缓冲区中的数据
                self._parse_buffer()

            except serial.SerialException as e:
                self._handle_error(e)
                self.is_running = False
                break

            # 稍微休眠一下，避免CPU占用过高
            time.sleep(0.01)

    def _parse_buffer(self):
        """
        从接收缓冲区中解析出完整的数据帧
        这个函数处理粘包、半包等问题
        """
        while len(self.read_buffer) >= MIN_FRAME_LENGTH:
            # 寻找帧头
            start_index = self.read_buffer.find(FRAME_START)
            if start_index == -1:
                # 没找到帧头，清空缓冲区
                self.read_buffer.clear()
                return

            # 删除帧头前面的无用数据
            if start_index > 0:
                del self.read_buffer[:start_index]

            # 检查是否有足够的数据来读取帧头信息
            if len(self.read_buffer) < 3:
                return

            # 读取地址和数据长度
            address = self.read_buffer[1]
            data_len = self.read_buffer[2]
            full_frame_len = data_len + 5  # 数据长度 + 帧头帧尾等固定部分

            # 检查是否收到了完整的帧
            if len(self.read_buffer) < full_frame_len:
                return

            # 提取完整的帧
            frame = self.read_buffer[:full_frame_len]

            # 检查帧尾
            if frame[-1] != FRAME_END:
                # 帧尾不对，删除当前帧头，继续寻找
                del self.read_buffer[0]
                continue

            # 提取数据部分和校验和
            data_bytes = frame[3:-2]
            received_checksum = frame[-2]

            # 验证数据长度
            if len(data_bytes) != data_len:
                del self.read_buffer[0]
                continue

            # 验证校验和
            calculated_checksum = self._calculate_checksum(address, data_len, data_bytes)

            if received_checksum == calculated_checksum:
                # 校验通过，组装消息并处理
                message = {'address': address, 'data': data_bytes}
                try:
                    self._handle_message(message)
                except Exception as e:
                    print(f"消息处理出错: {e}")
            else:
                # 校验失败
                error = ChecksumError(f"校验和错误! 期望 {calculated_checksum}, 收到 {received_checksum}",
                                    frame=bytes(frame))
                self._handle_error(error)

            # 删除已处理的帧
            del self.read_buffer[:full_frame_len]

    def _handle_message(self, message):
        """
        默认的消息处理函数
        子类应该重写这个函数来实现具体的业务逻辑
        """
        address = message['address']
        data = message['data']
        print(f"[收到消息] 地址=0x{address:02x}, 数据={data.hex()}")

    def _handle_error(self, error):
        """
        默认的错误处理函数
        子类可以重写这个函数来自定义错误处理
        """
        if isinstance(error, ChecksumError):
            print(f"[校验错误] 数据帧: {error.frame.hex()}")
        else:
            print(f"[通信错误] {error}")

class MyUartHandler(Uart):
    """
    继承自Uart的示例类
    演示如何实现自定义的消息处理逻辑
    """
    def __init__(self, port, baudrate=9600, timeout=1, byte_delay=0.001):
        super().__init__(port, baudrate, timeout, byte_delay)
        # 用队列存储接收到的数据
        self.data_queue = collections.deque()

    def _handle_message(self, message):
        """重写消息处理，实现自定义的业务逻辑"""
        address = message['address']
        data = message['data']
        print(f"[自定义处理] 地址=0x{address:02x}, 数据={data.hex()}")

        # 专门处理地址为0x03的消息
        if address == 0x03:
            if len(data) == 2:
                try:
                    # 解析为16位无符号整数（小端序）
                    value = struct.unpack('<H', data)[0]
                    print(f"    -> 地址0x03数据解析: {value}")
                    # 存储到队列中
                    self.data_queue.append(value)
                    print(f"    -> 数据已存储，队列内容: {list(self.data_queue)}")
                except struct.error:
                    print("    -> 数据解析失败")
            else:
                print(f"    -> 地址0x03期望2字节数据，实际收到{len(data)}字节")
            return

        # 其他地址的消息，尝试解析为浮点数
        if len(data) == 4:
            try:
                float_val = struct.unpack('<f', data)[0]
                print(f"    -> 解析为浮点数: {float_val:.4f}")
            except struct.error:
                print("    -> 无法解析为浮点数")

    def _handle_error(self, error):
        """重写错误处理"""
        if isinstance(error, ChecksumError):
            print(f"[错误处理] 校验失败! 数据帧: {error.frame.hex()}")
        else:
            print(f"[错误处理] 通信异常: {error}")

# 如果直接运行这个文件，就执行测试代码
if __name__ == '__main__':
    try:
        # 请根据你的实际情况修改串口名称
        comm = MyUartHandler('/dev/ttyS3', baudrate=115200, byte_delay=0.001)
        comm.start()

        # 发送测试数据
        print("发送测试数据...")

        # 发送一个整数到地址0x03
        my_int = 270
        int_bytes = struct.pack('<H', my_int)
        print(f"发送整数 {my_int}，字节: {int_bytes.hex()}")
        comm.send_message(0x03, int_bytes)
        time.sleep(1)

        # 发送一个浮点数到地址0x10
        my_float = 3.14159
        float_bytes = struct.pack('<f', my_float)
        print(f"发送浮点数 {my_float}，字节: {float_bytes.hex()}")
        comm.send_message(0x10, float_bytes)
        time.sleep(1)

        print("测试完成，停止通信")
        comm.stop()

        # 显示最终的队列内容
        print(f"最终队列内容: {list(comm.data_queue)}")

    except (serial.SerialException, FileNotFoundError) as e:
        print(f"测试失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
