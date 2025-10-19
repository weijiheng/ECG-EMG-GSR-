# -*- coding: utf-8 -*-

import sys
import numpy as np
import threading
import pyqtgraph as pg
import serial
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QColor, QTextCharFormat, QFont
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QLineEdit, \
    QPushButton, QTextBrowser, QApplication, QGroupBox,QCheckBox
import datetime
import os  # 导入os模块用于文件路径操作
import socket
import time


import socket
import time
from PyQt5.QtCore import QThread, pyqtSignal


class MarkerServer(QThread):
    # 信号：发送marker数字和elapsed时间
    marker_received = pyqtSignal(int, float)
    start_signal = pyqtSignal()  # 开始采集信号
    end_signal = pyqtSignal()    # 结束采集信号
    save_signal = pyqtSignal()   # 保存数据信号
    restart_signal = pyqtSignal()  # 重新开始信号
    start_Resting_state_signal = pyqtSignal()  # 开始静息态信号
    save_Resting_state_signal = pyqtSignal()   # 保存静息态信号


    def __init__(self, host='127.0.0.1', port=9001):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = True  # 控制服务端是否持续运行

    def run(self):
        # 创建服务端 socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 允许端口重用
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)  # 只允许一个排队连接
            print(f"[Server] Listening on {self.host}:{self.port}...")
        except Exception as e:
            print(f"[Server] Failed to bind: {e}")
            return

        while self.is_running:
            print("[Server] Waiting for client connection...")
            try:
                client_sock, client_addr = self.server_socket.accept()
                print(f"[Server] Client connected from {client_addr}")

                # 处理该客户端连接（直到断开）
                self.handle_client(client_sock)

                print(f"[Server] Client {client_addr} disconnected.")
            except Exception as e:
                if self.is_running:
                    print(f"[Server] Accept error: {e}")

        self.server_socket.close()
        print("[Server] Server stopped.")

    def handle_client(self, sock):
        """处理单个客户端连接的完整生命周期"""
        buffer = b""
        start_time = None  # 每个客户端独立计时
        Formal_exp = False  # 是否正式实验
        video_count = 0
        phase = 1  # 当前实验阶段

        try:
            while True:
                data = sock.recv(1024)
                if not data:
                    break  # 客户端断开
                buffer += data

                # 逐字节处理
                while len(buffer) >= 1:
                    marker_byte = buffer[:1]
                    buffer = buffer[1:]
                    # 安全获取单字节整数值
                    try:
                        marker_code = marker_byte[0]
                    except Exception as e:
                        print(f"[Server] 解析marker失败: {marker_byte} 错误: {e}")
                        continue
                    # 调试输出原始字节
                    print(f"[Server] DEBUG 原始字节: {marker_byte.hex()} -> {marker_code}", flush=True)

                    if marker_code == 4:
                        # 开始 marker
                        start_time = time.perf_counter()
                        self.start_signal.emit()
                        Formal_exp = True
                        print(f"[Server] 接收到marker：{marker_code}", flush=True)
                        self.marker_received.emit(marker_code, 0.0)

                    elif marker_code == 7 and Formal_exp:
                        # 结束 marker
                        if start_time is None:
                            print("[Server] ⚠️ 未收到 start，无法计算时间")
                        else:
                            self.end_signal.emit()
                            print(f"[Server] 接收到marker：{marker_code}")
                            print("[Server] 实验结束，客户端将断开。")
                            break  # 结束当前客户端会话

                    elif marker_code == 5 and Formal_exp:
                        # 重新开始 marker
                        if video_count == 0 and phase != 1:
                            start_time = time.perf_counter()
                            self.marker_received.emit(marker_code, 0.0)
                            self.restart_signal.emit()
                        else:
                            if start_time is None:
                                print("[Server] ⚠️ 未收到 start，无法计算时间")
                            else:
                                elapsed = time.perf_counter() - start_time
                                print(f"[Server] 接收到marker：{marker_code}")
                                self.marker_received.emit(marker_code, elapsed)
                        video_count += 1

                    elif marker_code == 6 and Formal_exp:
                        # 中途保存 marker
                        if start_time is None:
                            print("[Server] ⚠️ 未收到 start，无法计算时间")
                        else:
                            elapsed = time.perf_counter() - start_time
                            print(f"[Server] 接收到marker：{marker_code}")
                            self.marker_received.emit(marker_code, elapsed)
                            if video_count == 4:
                                self.save_signal.emit()
                                video_count = 0
                                phase += 1

                    elif (8 <= marker_code <= 25) and Formal_exp:
                        # 低频 marker
                        if start_time is None:
                            print("[Server] ⚠️ 未收到 start，无法计算时间")
                        else:
                            elapsed = time.perf_counter() - start_time
                            print(f"[Server] 接收到marker：{marker_code}")
                            self.marker_received.emit(marker_code, elapsed)

                    elif marker_code == 26:
                        # 静息态开始 marker
                        start_time = time.perf_counter()
                        self.start_Resting_state_signal.emit()
                        print("[Server] 静息态实验开始", flush=True)
                        print(f"[Server] 接收到marker：{marker_code}", flush=True)
                        self.marker_received.emit(marker_code, 0.0)

                    elif marker_code == 33:
                        # 静息态结束 marker
                        if start_time is None:
                            print("[Server] ⚠️ 未收到 start，无法计算时间")
                        else:
                            elapsed = time.perf_counter() - start_time
                            print(f"[Server] 接收到marker：{marker_code}", flush=True)
                            self.marker_received.emit(marker_code, elapsed)
                            # 触发保存（在主线程中会先停止采集并flush）
                            self.save_Resting_state_signal.emit()
                            print("[Server] 静息态实验结束，客户端将断开。", flush=True)
                            break
                    elif  27 <= marker_code <= 32:
                        # 高频 marker
                        if start_time is None:
                            print("[Server] ⚠️ 未收到 start，无法计算时间")
                        else:
                            elapsed = time.perf_counter() - start_time
                            print(f"[Server] 接收到marker：{marker_code}")
                            self.marker_received.emit(marker_code, elapsed)

                    else:
                        # 未知 marker
                        print(f"[Server] 未知的marker代码: {marker_code}")

        except Exception as e:
            print(f"[Server] Client handling error: {e}")
        finally:
            sock.close()

    def stop(self):
        """安全停止服务端"""
        self.is_running = False
        # 可选：创建一个临时连接来中断 accept()
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_sock.connect((self.host, self.port))
            temp_sock.close()
        except:
            pass

# 多线程通信
# 多线程通信
class SerialThread(QThread):
    """串口 数据接收线程"""
    data_received = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, baudrate=500000, buffer_threshold=140):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.is_recording = False
        self.buffer = bytearray()
        self.buffer_threshold = buffer_threshold
        self.packet_size = 14
        self.max_buffer_size = 4096
        
        # 数据积累器
        self.accumulate_threshold = 10
        self.accumulated_ecg_1 = []
        self.accumulated_ecg_2 = []
        self.accumulated_emg_1 = []
        self.accumulated_emg_2 = []
        self.accumulated_gsr_1 = []
        self.accumulated_gsr_2 = []
        
        # 异常包计数器和最后一个正常包的备份
        self.abnormal_packet_count = 0
        self.last_valid_data = None  # 保存最后一组正常数据

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            self.ser.set_buffer_size(rx_size=65536, tx_size=4096)
            self.running = True
            self.error_occurred.emit(f"串口 {self.port} 已连接，开始预采集")
            
            while self.running:
                if self.ser.in_waiting:
                    # 读取所有可用数据
                    data = self.ser.read(self.ser.in_waiting)
                    if data:
                        self.buffer.extend(data)
                        
                        if len(self.buffer) >= self.buffer_threshold:
                            self.process_buffer_data()

                else:
                    # 即使没有新数据，也处理现有缓冲区
                    if len(self.buffer) >= self.packet_size:
                        self.process_buffer_data()
                    self.msleep(1)
                    
        except Exception as e:
            self.error_occurred.emit(f"串口错误: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def process_buffer_data(self):
        """优化的缓冲区数据处理，增强异常包处理"""
        buffer_len = len(self.buffer)
        packet_size = self.packet_size
        
        processed_bytes = 0
        i = 0
        
        # 尽可能处理所有完整的数据包
        while i <= buffer_len - packet_size:
            if self.buffer[i] == 0xAA:
                # 检查是否有完整数据包
                if i + packet_size <= buffer_len and self.buffer[i + packet_size - 1] == 0x55:
                    # 找到完整的正常数据包
                    data = self.buffer[i:i + packet_size]
                    
                    try:
                        # 解析数据并直接添加到积累器
                        gsr1 = (data[2] << 8) | data[1]
                        emg1 = (data[4] << 8) | data[3]
                        ecg1 = (data[6] << 8) | data[5]
                        gsr2 = (data[8] << 8) | data[7]
                        emg2 = (data[10] << 8) | data[9]
                        ecg2 = (data[12] << 8) | data[11]
                        
                        # 保存最后一组正常数据用于异常恢复
                        self.last_valid_data = (gsr1, emg1, ecg1, gsr2, emg2, ecg2)
                        
                        # 添加到积累器
                        self.accumulated_ecg_1.append(ecg1)
                        self.accumulated_ecg_2.append(ecg2)
                        self.accumulated_emg_1.append(emg1)
                        self.accumulated_emg_2.append(emg2)
                        self.accumulated_gsr_1.append(gsr1)
                        self.accumulated_gsr_2.append(gsr2)
                        
                        # 检查是否达到积累阈值
                        if len(self.accumulated_ecg_1) >= self.accumulate_threshold:
                            self.send_accumulated_data()
                            self.reset_accumulator()
                        
                        i += packet_size
                        processed_bytes = i
                        
                    except Exception as e:
                        self.error_occurred.emit(f"数据解析错误: {e}")
                        i += 1
                else:
                    # 包头正确但包尾错误的异常包处理
                    self.handle_abnormal_packet_with_correct_header(i, buffer_len)
                    i += 1
            else:
                # 包头错误，计为异常包
                self.abnormal_packet_count += 1
                i += 1
        
        # 优化的缓冲区清理逻辑
        self.cleanup_buffer(processed_bytes, buffer_len)

    def handle_abnormal_packet_with_correct_header(self, header_pos, buffer_len):
        """处理包头正确但包尾异常的数据包"""
        remaining_bytes = buffer_len - header_pos
        
        if remaining_bytes < self.packet_size:
            # 情况1: 剩余字节不足一个完整包，可能是不完整的包
            # 这些数据将在cleanup_buffer中被保留到下次处理
            pass
        else:
            # 情况2: 剩余字节足够一个包但包尾不正确
            # 使用上一个正常数据包的数据进行恢复
            if self.last_valid_data:
                gsr1, emg1, ecg1, gsr2, emg2, ecg2 = self.last_valid_data
                
                # 添加到积累器
                self.accumulated_ecg_1.append(ecg1)
                self.accumulated_ecg_2.append(ecg2)
                self.accumulated_emg_1.append(emg1)
                self.accumulated_emg_2.append(emg2)
                self.accumulated_gsr_1.append(gsr1)
                self.accumulated_gsr_2.append(gsr2)
                
                self.error_occurred.emit(f"检测到异常包，使用上一个正常包数据进行恢复")
            
        self.abnormal_packet_count += 1

    def cleanup_buffer(self, processed_bytes, buffer_len):
        """优化的缓冲区清理逻辑"""
        if processed_bytes > 0:
            # 清理已处理的数据
            self.buffer = self.buffer[processed_bytes:]
        else:
            # 没有处理任何完整包的情况
            # 检查是否有不完整的包头需要保留
            incomplete_packet_start = self.find_incomplete_packet_start()
            
            if incomplete_packet_start is not None:
                # 保留不完整的包数据到缓冲区开头
                self.buffer = self.buffer[incomplete_packet_start:]
            else:
                # 没有找到有效的包头，清理部分无效数据，但保留一些数据以防遗漏
                if len(self.buffer) > self.packet_size * 2:
                    # 如果缓冲区太大，保留最后packet_size长度的数据
                    self.buffer = self.buffer[-self.packet_size:]
        
        # 防止缓冲区过大
        if len(self.buffer) > self.max_buffer_size:
            # 只保留最后一部分数据
            self.buffer = self.buffer[-self.max_buffer_size//2:]
            self.error_occurred.emit("缓冲区过大，已清理部分数据")

    def find_incomplete_packet_start(self):
        """查找可能的不完整包开始位置"""
        buffer_len = len(self.buffer)
        
        # 从缓冲区末尾向前查找最后一个可能的包头
        for i in range(buffer_len - 1, -1, -1):
            if self.buffer[i] == 0xAA:
                remaining_bytes = buffer_len - i
                if remaining_bytes < self.packet_size:
                    # 找到不完整的包头
                    return i
                elif remaining_bytes >= self.packet_size:
                    # 检查这个位置是否有完整的包
                    if self.buffer[i + self.packet_size - 1] != 0x55:
                        # 包头正确但包尾错误，继续查找
                        continue
                    else:
                        # 找到完整包，不需要保留
                        return None
        
        return None

    def send_accumulated_data(self):
        """发送积累的数据到主线程"""
        if self.accumulated_ecg_1:
            self.data_received.emit(
                np.array(self.accumulated_ecg_1, dtype=np.uint16),
                np.array(self.accumulated_ecg_2, dtype=np.uint16),
                np.array(self.accumulated_emg_1, dtype=np.uint16),
                np.array(self.accumulated_emg_2, dtype=np.uint16),
                np.array(self.accumulated_gsr_1, dtype=np.uint16),
                np.array(self.accumulated_gsr_2, dtype=np.uint16)
            )

        self.reset_accumulator()
            
    def reset_accumulator(self):
        """重置数据积累器"""
        self.accumulated_ecg_1.clear()
        self.accumulated_ecg_2.clear()
        self.accumulated_emg_1.clear()
        self.accumulated_emg_2.clear()
        self.accumulated_gsr_1.clear()
        self.accumulated_gsr_2.clear()

    def stop(self):
        self.running = False
        self.is_recording = False
        
        # 发送剩余积累的数据
        if self.accumulated_ecg_1:
            self.send_accumulated_data()
            self.reset_accumulator()
            
        if self.ser and self.ser.is_open:
            self.ser.close()

        # 报告异常包统计
        self.error_occurred.emit(f"数据采集结束，本次共检测到 {self.abnormal_packet_count} 个异常数据包")

    def send_data(self, data):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data)
            except Exception as e:
                print(f"发送数据错误: {e}")



# 缓存类 - 优化，使用预分配数组和循环缓冲区
class DataBuffer:
    """数据缓冲区（无限长度，动态列表）"""

    def __init__(self):
        self.buffer = []  # 用于存储所有数据
        self.lock = threading.Lock()

    def append_raw(self, data):
        with self.lock:
            self.buffer.extend(data)

    def get_raw_data(self, low_index=None, high_index=None):
        with self.lock:
            arr = np.array(self.buffer, dtype=np.uint16)
            if low_index is None and high_index is None:
                return arr
            elif high_index is not None and low_index is not None:
                if high_index > low_index and high_index <= len(arr):
                    return arr[low_index:high_index]
            return None

    def clear(self):
        with self.lock:
            self.buffer = []

    def has_data(self):
        with self.lock:
            return len(self.buffer) > 0


# GUI设置
class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("生物数据采集与处理系统")
        self.resize(1000, 700)

        # 数据结构和配置参数
        # 正式数据缓冲区
        self.ECG_1_data_ = DataBuffer()
        self.ECG_2_data_ = DataBuffer()
        self.EMG_1_data_ = DataBuffer()
        self.EMG_2_data_ = DataBuffer()
        self.GSR_1_data_ = DataBuffer()
        self.GSR_2_data_ = DataBuffer()

        # 预采集数据缓冲区
        self.pre_ECG_1_data_ = DataBuffer()
        self.pre_ECG_2_data_ = DataBuffer()
        self.pre_EMG_1_data_ = DataBuffer()
        self.pre_EMG_2_data_ = DataBuffer()
        self.pre_GSR_1_data_ = DataBuffer()
        self.pre_GSR_2_data_ = DataBuffer()

        self.serial_thread = None
        self.is_recording = False  # 是否正式记录数据
        self.initialized = False  # 是否已完成初始化

        # marker存储列表
        self.markers = []  # 存储格式: [(marker_name, elapsed_time), ...]
        self.current_cycle_markers = []  # 当前循环的marker存储
        self.current_phase = 1  # 当前实验阶段：1或2
        self.video_count = 0  # 视频计数器

        # 新增：实验人数设置（1或2）
        self.subject_count = 2  # 默认2人

        # 实验信号采集设置
        self.ecg_channel = False  # ECG通道
        self.emg_channel = False  # EMG通道
        self.gsr_channel = False  # GSR通道

        # 配置参数 - 与频率设置类似的方式处理实验编号
        self.sampling_rate = 1000  # 采样率属性，固定为1000Hz
        self.experiment_id = ""  # 实验编号属性

        # 优化：调整显示参数，减少UI更新频率以避免阻塞数据采集
        self.display_rate = 100  # Hz
        self.display_points = self.display_rate * 10  # 10秒数据
        self.update_interval = 100  # 毫秒，降低更新频率从50ms到100ms
        self.update_points = int(self.display_rate * (self.update_interval / 1000))  # 10点
        self.draw_index = 99  # 索引起始点

        # 创建UI
        self.setup_ui()

        # 初始化并启动套接字客户端
        self.init_marker_server()

        # 程序启动时自动连接串口，开始预采集
        self.init_serial_connection()

        # 定时器用于更新图像
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)

    def init_marker_server(self):
        """初始化并启动Marker服务器"""
        self.marker_server = MarkerServer()
        self.marker_server.marker_received.connect(self.handle_marker)
        self.marker_server.start_signal.connect(self.start_collection)
        self.marker_server.end_signal.connect(self.end_experiment)
        self.marker_server.save_signal.connect(self.save_intermediate_data)
        self.marker_server.restart_signal.connect(self.restart_collection)
        self.marker_server.start_Resting_state_signal.connect(self.start_resting_collection)
        self.marker_server.save_Resting_state_signal.connect(self.save_Resting_state_digital)
        self.marker_server.start()
        self.log_message(f"Marker服务器已启动，连接到 {self.marker_server.host}:{self.marker_server.port}", "info")

    def handle_marker(self, marker_code, elapsed):

        """处理接收到的marker数字，包含时间戳"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # 存储格式: (marker数字, elapsed_time, timestamp)
        marker_entry = (marker_code, elapsed, current_time)

        # 添加到当前循环的marker列表
        self.current_cycle_markers.append(marker_entry) 

        # 同时存储到总的marker列表（为了兼容现有逻辑）
        self.markers.append(marker_entry)
        
        self.log_message(f"收到Marker: {marker_code}, 时间: {elapsed:.3f}秒 (第{self.current_phase}阶段)", "info")
        

    def handle_end_marker(self):
        """处理结束marker，结束采集并保存数据"""
        self.stop_collection()
        # 延迟保存，确保数据处理完成
                # 创建实验编号文件夹
        data_dir = os.path.join("data", self.experiment_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = os.path.join(data_dir, f"{self.experiment_id}_{self.current_phase}.csv")


        # 保存当前正式采集的数据（第一阶段数据）
        if self._save_data_to_file(filename):
            self.log_message(f"第{self.current_phase}阶段数据已保存到 {filename}", "info")

            # 保存当前阶段marker日志
            self.save_cycle_markers(data_dir)
            
            # 清理所有缓存（生物数据、marker、串口）
            self.clear_all_data_and_markers()

    def save_intermediate_data(self):
        """保存中途数据为_f.csv文件并清理缓存"""
        if not self.experiment_id:
            self.log_message(f"错误: 实验编号未设置", "error")
            return
        
        # 确保将串口线程中的积累数据传回主线程，然后再保存数据
        if self.serial_thread and self.is_recording:
            self.serial_thread.send_accumulated_data()
            self.log_message("已传回串口线程积累数据", "info")
            
        self.serial_thread.is_recording = False
        self.is_recording = False
        
        # 创建实验编号文件夹
        data_dir = os.path.join("data", self.experiment_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = os.path.join(data_dir, f"{self.experiment_id}_{self.current_phase}.csv")


        # 保存当前正式采集的数据（第一阶段数据）
        if self._save_data_to_file(filename):
            self.log_message(f"第{self.current_phase}阶段数据已保存到 {filename}", "info")

            # 保存当前阶段marker日志
            self.save_cycle_markers(data_dir)
            
            # 清理所有缓存（生物数据、marker、串口）
            self.clear_all_data_and_markers()
            
            # 清理串口缓存并休眠30秒
            self.clean_serial_and_rest()

            self.current_phase += 1  # 进入下一个阶段
        else:
            self.log_message(f"第{self.current_phase}阶段数据保存失败", "error")



    def start_resting_collection(self):
        """静息态开始：丢弃尚未发送到主线程的积累数据，开启正式采集"""
        if not self.initialized:
            self.log_message("错误：未初始化，无法开始静息态", "error")
            return
        if self.serial_thread:
            # 直接清空积累区（不发送，按需求丢弃）
            self.serial_thread.reset_accumulator()
        # 只清正式缓冲，不动预采集历史（可根据需要调整）
        self.clear_formal_buffers()
        self.current_cycle_markers = []
        self.serial_thread.is_recording = True
        self.is_recording = True
        self.ecg_plot.setTitle('ECG信号 (静息态采集)')
        self.emg_plot.setTitle('EMG信号 (静息态采集)')
        self.gsr_plot.setTitle('GSR信号 (静息态采集)')
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_message("静息态正式采集开始", "info")

    def save_Resting_state_digital(self):
        """静息态结束：停止采集 -> flush 未发送数据 -> 保存 -> 清理"""
        if not self.experiment_id:
            self.log_message("错误: 实验编号未设置", "error")
            return

        # 立刻停止接收
        if self.is_recording:
            if self.serial_thread:
                self.serial_thread.is_recording = False
            self.is_recording = False

        # Flush 未发送积累数据
        if self.serial_thread:
            self.serial_thread.send_accumulated_data()
            self.serial_thread.reset_accumulator()
            self.log_message("静息态剩余缓存数据已刷新", "info")

        data_dir = os.path.join("data", self.experiment_id)
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.join(data_dir, f"{self.experiment_id}_resting_state.csv")

        if self._save_data_to_file(filename):
            self.log_message(f"静息态数据已保存到 {filename}", "info")
            self.save_resting_state_markers(data_dir)
            self.clear_all_data_and_markers()
        else:
            self.log_message("静息态数据保存失败", "error")

        # 恢复到预采集显示
        self.ecg_plot.setTitle('ECG信号 (预采集模式)')
        self.emg_plot.setTitle('EMG信号 (预采集模式)')
        self.gsr_plot.setTitle('GSR信号 (预采集模式)')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)




    def save_final_data(self):
        """保存最终数据为_a.csv文件"""
        if not self.experiment_id:
            self.log_message(f"错误: 实验编号未设置", "error")
            return
            
        # 创建实验编号文件夹
        data_dir = os.path.join("data", self.experiment_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        filename = os.path.join(data_dir, f"{self.experiment_id}_2.csv")
        
        # 保存当前正式采集的数据（第二阶段数据）
        if self._save_data_to_file(filename, phase="second"):
            self.log_message(f"第二阶段数据已保存到 {filename}", "info")
            # 保存第二阶段marker日志
            self.save_second_phase_markers(data_dir)
        else:
            self.log_message(f"第二阶段数据保存失败", "error")

    def save_cycle_markers(self, data_dir):
        """保存当前阶段marker日志"""
        if not self.current_cycle_markers:
            self.log_message("没有当前阶段marker信息可保存", "warning")
            return

        marker_filename = os.path.join(data_dir, f"{self.experiment_id}_{self.current_phase}_marker_log.txt")

        try:
            with open(marker_filename, 'w', encoding='utf-8') as f:
                for marker_code, elapsed, timestamp in self.current_cycle_markers:
                    # 格式：marker数字 | 时间差 | 时间戳
                    line = f"{marker_code} | {elapsed:.3f} | {timestamp}\n"
                    f.write(line)
            self.log_message(f"{self.current_phase}阶段的Marker信息已保存到 {marker_filename}", "info")
        except Exception as e:
            self.log_message(f"{self.current_phase}阶段的marker文件失败: {str(e)}", "error")


    def save_resting_state_markers(self, data_dir):
        """保存静息态marker日志"""
        if not self.current_cycle_markers:
            self.log_message("没有静息态marker信息可保存", "warning")
            return

        marker_filename = os.path.join(data_dir, f"{self.experiment_id}_resting_state_marker_log.txt")

        try:
            with open(marker_filename, 'w', encoding='utf-8') as f:
                for marker_code, elapsed, timestamp in self.current_cycle_markers:
                    # 格式：marker数字 | 时间差 | 时间戳
                    line = f"{marker_code} | {elapsed:.3f} | {timestamp}\n"
                    f.write(line)
            self.log_message(f"静息态的Marker信息已保存到 {marker_filename}", "info")
        except Exception as e:
            self.log_message(f"静息态的marker文件失败: {str(e)}", "error")

    def end_experiment(self):
        """结束实验：保存数据，断开串口连接，清理缓存"""
        self.log_message("实验结束", "info")

    # def save_second_phase_markers(self, data_dir):
    #     """保存第二阶段marker日志"""
    #     if not self.second_phase_markers:
    #         self.log_message("没有第二阶段marker信息可保存", "warning")
    #         return

    #     marker_filename = os.path.join(data_dir, f"{self.experiment_id}_2_marker_log.txt")

    #     try:
    #         with open(marker_filename, 'w', encoding='utf-8') as f:
    #             for marker_code, elapsed, timestamp in self.second_phase_markers:
    #                 # 格式：marker数字 | 时间戳
    #                 line = f"{marker_code} | {timestamp}\n"
    #                 f.write(line)
    #         self.log_message(f"第二阶段Marker信息已保存到 {marker_filename}", "info")
    #     except Exception as e:
    #         self.log_message(f"保存第二阶段marker文件失败: {str(e)}", "error")

    def restart_collection(self):
        """重新开始采集：清除预采集数据，开始正式采集下个阶段"""

        # 关键优化：清空串口线程的积累缓存，确保正式采集数据准确
        if self.serial_thread:
            self.serial_thread.reset_accumulator()

        # 开始正式采集
        self.serial_thread.is_recording = True
        self.is_recording = True

        # 清除预采集数据
        self.clear_pre_buffers()
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # 更新图表标题
        self.ecg_plot.setTitle(f'ECG信号 (正式采集模式 - 第{self.current_phase}阶段)')
        self.emg_plot.setTitle(f'EMG信号 (正式采集模式 - 第{self.current_phase}阶段)')
        self.gsr_plot.setTitle(f'GSR信号 (正式采集模式 - 第{self.current_phase}阶段)')

        self.log_message(f"{self.experiment_id} 实验第{self.current_phase}阶段开始", "info")

    def clean_serial_and_rest(self):
        """清理串口缓存并休眠30秒，然后重新开始预采集"""
        # 停止当前正式采集，但保持串口连接

        
        # 清理串口内部缓存
        if self.serial_thread and self.serial_thread.ser and self.serial_thread.ser.is_open:
            try:
                self.serial_thread.reset_accumulator()
                self.log_message("串口缓存已清理", "info")
            except Exception as e:
                self.log_message(f"清理串口缓存失败: {e}", "warning")
        
        # 更新图表标题为预采集模式
        self.ecg_plot.setTitle('ECG信号 (预采集模式 )')
        self.emg_plot.setTitle('EMG信号 (预采集模式 )')
        self.gsr_plot.setTitle('GSR信号 (预采集模式 )')
        
        
        



    # def save_markers_to_txt(self):
    #     """将marker数字信息保存到文件"""
    #     if not self.markers:
    #         self.log_message("没有marker信息可保存", "warning")
    #         return

    #     if not hasattr(self, 'experiment_id') or not self.experiment_id:
    #         self.log_message("实验编号未设置，无法保存marker日志", "error")
    #         return

    #     # 构建marker文件名：实验编号 + marker_log.txt
    #     marker_filename = f"{self.experiment_id}marker_log.txt"

    #     try:
    #         with open(marker_filename, 'w', encoding='utf-8') as f:
    #             for marker_code, _, timestamp in self.markers:
    #                 # 格式：marker数字 | 时间戳
    #                 line = f"{marker_code} | {timestamp}\n"
    #                 f.write(line)
    #         self.log_message(f"Marker信息已保存到 {marker_filename}", "info")
    #     except Exception as e:
    #         self.log_message(f"保存marker文件失败: {str(e)}", "error")

    # def save_markers_to_txt_in_folder(self, data_dir):
    #     """将marker数字信息保存到指定文件夹"""
    #     if not self.markers:
    #         self.log_message("没有marker信息可保存", "warning")
    #         return

    #     # 构建marker文件名：实验编号文件夹内的marker_log.txt
    #     marker_filename = os.path.join(data_dir, f"{self.experiment_id}_marker_log.txt")

    #     try:
    #         with open(marker_filename, 'w', encoding='utf-8') as f:
    #             for marker_code, _, timestamp in self.markers:
    #                 # 格式：marker数字 | 时间戳
    #                 line = f"{marker_code} | {timestamp}\n"
    #                 f.write(line)
    #         self.log_message(f"Marker信息已保存到 {marker_filename}", "info")
    #     except Exception as e:
    #         self.log_message(f"保存marker文件失败: {str(e)}", "error")

    def init_serial_connection(self):
        """初始化串口连接，启动后立即开始预采集"""
        port = self.port_select.currentText()
        # 使用优化后的较小阈值
        self.serial_thread = SerialThread(port, baudrate=500000, buffer_threshold=140)
        self.serial_thread.data_received.connect(self.handle_serial_data)
        self.serial_thread.error_occurred.connect(self.handle_serial_error)
        self.serial_thread.start()

    def handle_serial_error(self, msg):
        """处理串口错误错误信息"""
        level = "error" if "错误" in msg else "info"
        self.log_message(msg, level)
        # 检查是否已初始化，再决定是否启用开始按钮
        self.start_button.setEnabled("错误" not in msg and self.initialized)

    def setup_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # 串口控制区
        control_layout = QHBoxLayout()

        # 串口选择
        self.port_label = QLabel("选择串口:")
        self.port_select = QComboBox()
        self.port_select.addItem("COM10")
        self.port_select.addItem("COM9")
        self.port_select.addItem("COM8")
        self.port_select.addItem("COM7")
        self.port_select.addItem("COM5")
        self.port_select.addItem("COM4")
        self.port_select.addItem("COM3")
        self.port_select.addItem("COM2")
        self.port_select.addItem("COM2")
        self.port_select.addItem("COM1")
        self.port_select.setCurrentIndex(4)  # 默认选择第一个串口

        # 新增：实验人数选择
        self.subject_label = QLabel("采集人数:")
        self.subject_select = QComboBox()
        self.subject_select.addItem("1人")
        self.subject_select.addItem("2人")
        self.subject_select.setCurrentIndex(1)  # 默认选择2人

        # 实验编号输入框
        self.exp_id_label = QLabel("实验编号:")
        self.exp_id_edit = QLineEdit()
        self.exp_id_edit.setPlaceholderText("输入实验编号，如101、102")
        self.exp_id_edit.setFixedWidth(120)

        # 添加初始化按钮
        self.init_button = QPushButton("初始化设置")
        self.init_button.clicked.connect(self.initialize_settings)

        # 串口通信选择回调
        self.port_select.currentTextChanged.connect(self.select_port)

        # 按钮
        self.start_button = QPushButton("开始采集")
        self.start_button.clicked.connect(self.start_collection)
        self.start_button.setEnabled(False)  # 初始禁用，等待初始化和串口连接成功

        self.stop_button = QPushButton("停止采集")
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_button.setEnabled(False)

        self.save_button = QPushButton("保存数据")
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setEnabled(False)

        control_layout.addWidget(self.port_label)
        control_layout.addWidget(self.port_select)
        control_layout.addWidget(self.subject_label)  # 新增
        control_layout.addWidget(self.subject_select)  # 新增
        control_layout.addWidget(self.exp_id_label)
        control_layout.addWidget(self.exp_id_edit)
        control_layout.addWidget(self.init_button)  # 添加初始化按钮
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)




        # 模式选择勾选框
        module_layout = QHBoxLayout()

        self.ecg_checkbox = QCheckBox("ECG")
        self.emg_checkbox = QCheckBox("EMG")
        self.gsr_checkbox = QCheckBox("GSR")

        self.ecg_checkbox.setChecked(False)
        self.emg_checkbox.setChecked(False)
        self.gsr_checkbox.setChecked(False)

        self.ecg_checkbox.stateChanged.connect(self.ECG_channel_changed)
        self.emg_checkbox.stateChanged.connect(self.EMG_channel_changed)
        self.gsr_checkbox.stateChanged.connect(self.GSR_channel_changed)

        module_layout.addWidget(self.ecg_checkbox)
        module_layout.addWidget(self.emg_checkbox)
        module_layout.addWidget(self.gsr_checkbox)


        # 状态栏
        self.messagelabel = QTextBrowser()

        # 波形显示
        self.create_groupboxes()

        # 添加到主布局
        main_layout.addLayout(control_layout)
        main_layout.addLayout(module_layout)
        main_layout.addWidget(self.widget)
        main_layout.addWidget(self.messagelabel)

        self.log_message(f"应用已启动，开始预采集信号...", "info")
        self.log_message(f"采样频率: 1000Hz", "info")  # 启动时显示采样频率
        self.log_message(f"正在连接串口：{self.port_select.currentText()};波特率：500000", "info")
        self.log_message(f"请设置实验编号与实验人数，然后点击初始化按钮", "warning")

    # 新增：初始化设置方法
    def initialize_settings(self):
        """处理初始化设置，将文本框内容赋值到属性并验证"""
        # 获取实验编号
        exp_id = self.exp_id_edit.text().strip()
        if not exp_id:
            self.log_message(f"错误: 实验编号不能为空", "error")
            self.start_button.setEnabled(False)
            self.initialized = False
            return

        # 获取实验人数
        subject_count = self.subject_select.currentIndex() + 1  # 0对应1人，1对应2人

        # 保存到类属性
        self.experiment_id = exp_id
        self.subject_count = subject_count
        self.initialized = True

        # 禁用设置选项，防止运行时更改
        # self.ecg_checkbox.setEnabled(False)
        # self.emg_checkbox.setEnabled(False)
        # self.gsr_checkbox.setEnabled(False)

        # 更新曲线可见性
        self.update_curve_visibility()

        # 在状态栏显示信息
        self.log_message(f"初始化成功 - 实验编号: {self.experiment_id}, 实验人数: {self.subject_count}人", "info")

       # 新增：汇总并显示当前选择的通道
        selected_channels = []
        if self.ecg_checkbox.isChecked():
            selected_channels.append("ECG")
        if self.emg_checkbox.isChecked():
            selected_channels.append("EMG")
        if self.gsr_checkbox.isChecked():
            selected_channels.append("GSR")
        channels_text = "、".join(selected_channels) if selected_channels else "无"
        self.log_message(f"已选择通道: {channels_text}", "info")

        # 检查串口连接状态，启用开始按钮
        if hasattr(self, 'serial_thread') and self.serial_thread and self.serial_thread.running:
            self.start_button.setEnabled(True)

    # 新增：更新曲线可见性
    def update_curve_visibility(self):
        if self.subject_count == 1:
            # 1人模式：只显示第一组数据
            self.ecg_curve1.setVisible(True and self.ecg_channel)
            self.ecg_curve2.setVisible(False)
            self.emg_curve1.setVisible(True and self.emg_channel)
            self.emg_curve2.setVisible(False)
            self.gsr_curve1.setVisible(True and self.gsr_channel)
            self.gsr_curve2.setVisible(False)
        else:
            # 2人模式：显示所有数据
            self.ecg_curve1.setVisible(True and self.ecg_channel)
            self.ecg_curve2.setVisible(True and self.ecg_channel)
            self.emg_curve1.setVisible(True and self.emg_channel)
            self.emg_curve2.setVisible(True and self.emg_channel)
            self.gsr_curve1.setVisible(True and self.gsr_channel)
            self.gsr_curve2.setVisible(True and self.gsr_channel)

    def create_groupboxes(self):
        self.widget = QWidget()
        self.verticalLayout = QVBoxLayout(self.widget)

        # ECG GroupBox
        self.ecgBox = QGroupBox("ECG信号", self.widget)
        self.ecg_layout = QVBoxLayout(self.ecgBox)
        self.ecg_plot = pg.PlotWidget()
        self.ecg_plot.setRange(yRange=[200, 600])
        self.ecg_plot.setDownsampling(mode='peak')
        self.ecg_plot.setClipToView(True)
        self.ecg_plot.setBackground('w')
        self.ecg_plot.setLabel('left', '幅度', units='mV')
        self.ecg_plot.setLabel('bottom', '时间', units='s')
        self.ecg_plot.showGrid(x=True, y=True)
        self.ecg_plot.setTitle('ECG信号 (预采集模式)')
        self.ecg_layout.addWidget(self.ecg_plot)
        self.verticalLayout.addWidget(self.ecgBox)

        self.ecg_plot.addLegend()
        self.ecg_curve1 = self.ecg_plot.plot(pen=pg.mkPen(color='r', width=2), name='ECG_1')
        self.ecg_curve2 = self.ecg_plot.plot(pen=pg.mkPen(color='b', width=2), name='ECG_2')

        # EMG GroupBox
        self.emgBox = QGroupBox("EMG信号", self.widget)
        self.emg_layout = QVBoxLayout(self.emgBox)
        self.emg_plot = pg.PlotWidget()
        self.emg_plot.setBackground('w')
        self.emg_plot.setYRange(0, 600)
        self.emg_plot.setDownsampling(mode='peak')
        self.emg_plot.setClipToView(True)
        self.emg_plot.setLabel('left', '幅度', units='mV')
        self.emg_plot.setLabel('bottom', '时间', units='s')
        self.emg_plot.showGrid(x=True, y=True)
        self.emg_plot.setTitle('EMG信号 (预采集模式)')
        self.emg_layout.addWidget(self.emg_plot)
        self.verticalLayout.addWidget(self.emgBox)

        self.emg_plot.addLegend()
        self.emg_curve1 = self.emg_plot.plot(pen=pg.mkPen(color='r', width=2), name='EMG_1')
        self.emg_curve2 = self.emg_plot.plot(pen=pg.mkPen(color='b', width=2), name='EMG_2')

        # GSR GroupBox
        self.gsrBox = QGroupBox("GSR信号", self.widget)
        self.gsr_layout = QVBoxLayout(self.gsrBox)
        self.gsr_plot = pg.PlotWidget()
        self.gsr_plot.setBackground('w')
        self.gsr_plot.setYRange(0, 600)
        self.gsr_plot.setDownsampling(mode='mean')
        self.gsr_plot.setClipToView(True)
        self.gsr_plot.setLabel('left', '幅度', units='mV')
        self.gsr_plot.setLabel('bottom', '时间', units='s')
        self.gsr_plot.showGrid(x=True, y=True)
        self.gsr_plot.setTitle('GSR信号 (预采集模式)')
        self.gsr_layout.addWidget(self.gsr_plot)
        self.verticalLayout.addWidget(self.gsrBox)

        self.gsr_plot.addLegend()
        self.gsr_curve1 = self.gsr_plot.plot(pen=pg.mkPen(color='r', width=2), name='GSR_1')
        self.gsr_curve2 = self.gsr_plot.plot(pen=pg.mkPen(color='b', width=2), name='GSR_2')

        # 初始更新曲线可见性
        self.update_curve_visibility()


    # 通道选择回调
    def ECG_channel_changed(self, state):
        self.ecg_channel = (state == Qt.Checked)
        if self.ecg_channel:
            self.log_message("ECG通道已启用", "info")
        else:
            self.log_message("ECG通道已禁用", "info")
     
    def EMG_channel_changed(self, state):
        self.emg_channel = (state == Qt.Checked)
        if self.emg_channel:
            self.log_message("EMG通道已启用", "info")
        else:
            self.log_message("EMG通道已禁用", "info")

    def GSR_channel_changed(self, state):
        self.gsr_channel = (state == Qt.Checked)
        if self.gsr_channel:
            self.log_message("GSR通道已启用", "info")
        else:
            self.log_message("GSR通道已禁用", "info")



    # 串口选择回调
    def select_port(self):
        # 停止当前串口线程
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()

        # 连接新串口，继续预采集
        port = self.port_select.currentText()
        self.log_message(f"切换到串口：{port}，继续预采集", "info")
        # 优化：使用相同的优化设置
        self.serial_thread = SerialThread(port, baudrate=500000, buffer_threshold=140)
        self.serial_thread.data_received.connect(self.handle_serial_data)
        self.serial_thread.error_occurred.connect(self.handle_serial_error)
        self.serial_thread.start()
        # 检查是否已初始化
        self.start_button.setEnabled(self.initialized)

    # 在MainWindow类中修改start_collection方法
    def start_collection(self):
        # 新增：检查是否已初始化
        if not self.initialized:
            self.log_message("错误：请先完成初始化设置再开始采集", "error")
            return  # 未初始化则直接返回，不执行后续采集逻辑

        # 关键优化：清空串口线程的积累缓存，确保正式采集数据准确
        if self.serial_thread:
            self.serial_thread.reset_accumulator()

        # 清除缓冲区并开始记录数据
        self.clear_all_buffers()


        self.serial_thread.is_recording = True
        self.is_recording = True

        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)

        # 更新图表标题为正式采集模式
        self.ecg_plot.setTitle('ECG信号 (正式采集模式)')
        self.emg_plot.setTitle('EMG信号 (正式采集模式)')
        self.gsr_plot.setTitle('GSR信号 (正式采集模式)')

        # 显示实验编号开始信息
        self.log_message(f"{self.experiment_id} 实验正式开始 (实验人数: {self.subject_count}人)", "info")

    def handle_serial_data(self, ecg1, ecg2, emg1, emg2, gsr1, gsr2):
        """处理串口接收的数据，根据状态决定存入预采集还是正式数据缓冲区"""
        if self.is_recording:
            # 正式采集状态，存入正式数据缓冲区
            self.ECG_1_data_.append_raw(ecg1)
            self.EMG_1_data_.append_raw(emg1)
            self.GSR_1_data_.append_raw(gsr1)

            # 只有2人模式才保存第二组数据
            if self.subject_count == 2:
                self.ECG_2_data_.append_raw(ecg2)
                self.EMG_2_data_.append_raw(emg2)
                self.GSR_2_data_.append_raw(gsr2)
        else:
            # 预采集状态，存入预采集数据缓冲区
            self.pre_ECG_1_data_.append_raw(ecg1)
            self.pre_EMG_1_data_.append_raw(emg1)
            self.pre_GSR_1_data_.append_raw(gsr1)

            if self.subject_count == 2:
                self.pre_ECG_2_data_.append_raw(ecg2)
                self.pre_EMG_2_data_.append_raw(emg2)
                self.pre_GSR_2_data_.append_raw(gsr2)

    def stop_collection(self):
        """停止正式数据采集（保持串口连接，继续预采集）"""
        if self.is_recording:
            self.serial_thread.is_recording = False
            self.is_recording = False

            # 关键优化：采集结束时，将串口线程积累区剩余数据全部传回主线程
            if self.serial_thread:
                self.serial_thread.send_accumulated_data()

                # 显示异常包数量
                self.display_abnormal_packet_count(self.serial_thread.abnormal_packet_count)

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_button.setEnabled(self.ECG_1_data_.has_data())
            
            # 恢复图表标题为预采集模式
            self.ecg_plot.setTitle('ECG信号 (预采集模式)')
            self.emg_plot.setTitle('EMG信号 (预采集模式)')
            self.gsr_plot.setTitle('GSR信号 (预采集模式)')

            # 显示实验编号结束信息
            self.log_message(f"{self.experiment_id} 实验结束", "info")
            
    def display_abnormal_packet_count(self, count):
        """在状态栏显示异常包数量"""
        self.log_message(f"本次采集异常包数量: {count}", "warning")

    def clear_all_buffers(self):
        """清除所有数据缓冲区"""
        # 清除正式数据缓冲区
        self.ECG_1_data_.clear()
        self.ECG_2_data_.clear()
        self.EMG_1_data_.clear()
        self.EMG_2_data_.clear()
        self.GSR_1_data_.clear()
        self.GSR_2_data_.clear()

        # 清除预采集数据缓冲区
        self.pre_ECG_1_data_.clear()
        self.pre_ECG_2_data_.clear()
        self.pre_EMG_1_data_.clear()
        self.pre_EMG_2_data_.clear()
        self.pre_GSR_1_data_.clear()
        self.pre_GSR_2_data_.clear()

        # 清除marker
        self.markers = []
        self.first_phase_markers = []
        self.second_phase_markers = []

        # 重置绘图索引和实验阶段
        self.draw_index = 49
        self.current_phase = 1

    def clear_all_data_and_markers(self):
        """清除所有数据缓冲区和marker缓存（在第一阶段保存后调用）"""
        # 清除正式数据缓冲区
        self.clear_formal_buffers()
        self.current_cycle_markers = []
        
        self.log_message("所有数据和marker缓存已清理", "info")

    def clear_formal_buffers(self):
        """清除正式数据缓冲区"""
        self.ECG_1_data_.clear()
        self.ECG_2_data_.clear()
        self.EMG_1_data_.clear()
        self.EMG_2_data_.clear()
        self.GSR_1_data_.clear()
        self.GSR_2_data_.clear()
        self.log_message("正式数据缓存已清理", "info")

    def clear_pre_buffers(self):
        """清除预采集数据缓冲区"""
        self.pre_ECG_1_data_.clear()
        self.pre_ECG_2_data_.clear()
        self.pre_EMG_1_data_.clear()
        self.pre_EMG_2_data_.clear()
        self.pre_GSR_1_data_.clear()
        self.pre_GSR_2_data_.clear()
        self.log_message("预采集数据已清理", "info")

    def stop_current_cycle(self):
        """停止当前循环的采集"""
        self.serial_thread.is_recording = False
        self.is_recording = False
        
        # 更新图表标题
        self.ecg_plot.setTitle('ECG信号 (等待下一阶段)')
        self.emg_plot.setTitle('EMG信号 (等待下一阶段)')
        self.gsr_plot.setTitle('GSR信号 (等待下一阶段)')

        self.log_message(f"{self.current_phase}阶段采集已停止，等待下一阶段", "info")

    def experiment_complete(self):
        """实验完全结束"""
        self.stop_collection()
        self.is_experiment_started = False
        self.current_phase = 0
        self.log_message(f"{self.experiment_id} 实验完全结束", "info")

    def update_plot(self):
        # 根据当前状态选择要显示的数据缓冲区
        if self.is_recording:
            ecg1_data = self.ECG_1_data_
            emg1_data = self.EMG_1_data_
            gsr1_data = self.GSR_1_data_
            ecg2_data = self.ECG_2_data_
            emg2_data = self.EMG_2_data_
            gsr2_data = self.GSR_2_data_
        else:
            ecg1_data = self.pre_ECG_1_data_
            emg1_data = self.pre_EMG_1_data_
            gsr1_data = self.pre_GSR_1_data_
            ecg2_data = self.pre_ECG_2_data_
            emg2_data = self.pre_EMG_2_data_
            gsr2_data = self.pre_GSR_2_data_

        # 获取当前数据长度
        data_len = len(ecg1_data.get_raw_data())

        if data_len < self.display_points:
            return  # 数据太少，不进行绘图

        # 计算绘图起止索引
        if self.draw_index >= data_len:
            self.draw_index = data_len  # 防止越界

        if self.draw_index > self.display_points:
            start = self.draw_index - self.display_points
        else:
            start = 0

        # 获取时间轴
        time_axis = np.linspace(0, (self.draw_index - start) / self.display_rate, self.draw_index - start)

        # 更新第一组曲线（始终显示）
        self.ecg_curve1.setData(time_axis, ecg1_data.get_raw_data()[start:self.draw_index])
        self.emg_curve1.setData(time_axis, emg1_data.get_raw_data()[start:self.draw_index])
        self.gsr_curve1.setData(time_axis, gsr1_data.get_raw_data()[start:self.draw_index])

        # 只有2人模式才更新第二组曲线
        if self.subject_count == 2:
            self.ecg_curve2.setData(time_axis, ecg2_data.get_raw_data()[start:self.draw_index])
            self.emg_curve2.setData(time_axis, emg2_data.get_raw_data()[start:self.draw_index])
            self.gsr_curve2.setData(time_axis, gsr2_data.get_raw_data()[start:self.draw_index])

        # 递增索引
        self.draw_index += self.update_points

    def save_data(self):
        """保存原始数据到CSV文件（只保存正式采集的数据）- 保持原有功能不变"""
        # 使用类属性中的实验编号
        if not self.experiment_id:
            self.log_message(f"错误: 实验编号未设置", "error")
            return

        # 确保data文件夹存在
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # 使用类属性中的实验编号作为文件名
        default_filename = os.path.join(data_dir, f"bio_data_{self.experiment_id}")

        # 打开文件对话框
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存数据", default_filename, "CSV文件 (*.csv);;所有文件 (*)"
        )

        if filename:
            if self._save_data_to_file(filename):
                self.log_message(f"正式采集数据已保存到 {filename}", "info")
                self.save_markers_to_txt()

    def _save_data_to_file(self, filename):
        """内部方法：将数据保存到指定文件"""
        try:
            # === 第一步：根据 subject_count 和复选框状态，构建 (列名, buffer) 列表 ===
            channel_info = []  # 存储元组: (列名字符串, 数据buffer)

            if self.subject_count == 1:
                # 1人模式：每个信号最多1个通道
                mapping = [
                    (self.ecg_channel, "ECG_1", self.ECG_1_data_),
                    (self.emg_channel, "EMG_1", self.EMG_1_data_),
                    (self.gsr_channel, "GSR_1", self.GSR_1_data_),
                ]
                for is_selected, col_name, buffer in mapping:
                    if is_selected:
                        channel_info.append((col_name, buffer))
            else:
                # 2人模式：每个信号有2个通道
                mapping = [
                    (self.ecg_channel, ["ECG_1", "ECG_2"], [self.ECG_1_data_, self.ECG_2_data_]),
                    (self.emg_channel, ["EMG_1", "EMG_2"], [self.EMG_1_data_, self.EMG_2_data_]),
                    (self.gsr_channel, ["GSR_1", "GSR_2"], [self.GSR_1_data_, self.GSR_2_data_]),
                ]
                for is_selected, col_names, buffers in mapping:
                    if is_selected:
                        for name, buf in zip(col_names, buffers):
                            channel_info.append((name, buf))

            # 检查是否有选中通道
            if not channel_info:
                self.log_message("没有选择任何通道进行保存", "warning")
                return False

            # 提取 buffer 列表（用于计算长度）
            data_buffers = [buf for _, buf in channel_info]

            # === 第二步：计算最大长度并生成时间、marker ===
            data_lengths = [len(buf.get_raw_data()) for buf in data_buffers]
            max_length = max(data_lengths) if data_lengths else 0

            if max_length == 0:
                self.log_message("没有正式采集的数据可保存", "warning")
                return False

            # 生成时间列
            time = np.arange(max_length) / self.sampling_rate

            # 生成 marker 列
            marker_column = ["0"] * max_length
            for marker, elapsed, _ in self.current_cycle_markers:
                row_index = int(round(elapsed * self.sampling_rate))
                if 0 <= row_index < max_length:
                    marker_column[row_index] = str(marker)
                else:
                    self.log_message(f"Marker {marker} 位置超出数据范围", "warning")

            # === 第三步：动态构建 DataFrame ===
            df_dict = {'Time(s)': time}

            # 为每个选中的通道添加列
            for col_name, buffer in channel_info:
                raw_data = buffer.get_raw_data()
                # 补齐到 max_length（如果某些通道数据较短）
                if len(raw_data) < max_length:
                    raw_data = np.pad(raw_data, (0, max_length - len(raw_data)), constant_values=0)
                elif len(raw_data) > max_length:
                    raw_data = raw_data[:max_length]  # 截断（理论上不应发生）
                df_dict[col_name] = raw_data.astype(int)

            df_dict['marker'] = marker_column

            # 创建 DataFrame 并保存
            raw_df = pd.DataFrame(df_dict)
            if not filename.endswith('.csv'):
                filename += '.csv'
            raw_df.to_csv(filename, index=False)
            return True

        except Exception as e:
            self.log_message(f"保存文件失败: {str(e)}", "error")
            return False

    def log_message(self, message, level="info"):
        """记录消息到状态栏"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"

        # 根据日志级别设置不同的颜色
        if level == "info":
            self.set_text_color(QColor(0, 0, 0))  # 黑色
        elif level == "warning":
            self.set_text_color(QColor(255, 140, 0))  # 橙色
        elif level == "error":
            self.set_text_color(QColor(255, 0, 0))  # 红色

        # 添加消息到状态栏
        self.messagelabel.append(log_entry)

        # 滚动到底部
        self.messagelabel.verticalScrollBar().setValue(
            self.messagelabel.verticalScrollBar().maximum()
        )

    def set_text_color(self, color):
        """设置文本颜色"""
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor = self.messagelabel.textCursor()
        cursor.setCharFormat(fmt)
        self.messagelabel.setTextCursor(cursor)

    def closeEvent(self, event):
        """窗口关闭时确保串口线程正确停止"""
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait()
        if self.marker_server and self.marker_server.isRunning():
            self.marker_server.stop()
            self.marker_server.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())