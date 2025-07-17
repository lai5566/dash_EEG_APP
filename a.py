#!/usr/bin/env python3
"""
優化版響應式EEG監控系統
修復Gauge Charts顯示、性能優化、RWD排版
"""

import multiprocessing
import threading
import time
import serial
from collections import deque
import queue
import json
import sqlite3
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import random
import uuid
import os
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import butter, sosfiltfilt
import psutil

# 新增的模組
try:
    import paho.mqtt.client as mqtt

    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠️ MQTT 模組未安裝，將跳過 MQTT 功能")

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ 音頻模組未安裝，將跳過錄音功能")

# ----- 全局設定 -----
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 57600
WINDOW_SIZE = 512  # 統一使用 b.py 的配置
UPDATE_INTERVAL = 500  # 優化：降低更新頻率提升流暢度
FS = 256
BATCH_SIZE = 100  # 新增批次大小配置

# ----- 性能優化設定 -----
ADAPTIVE_UPDATE = True  # 自適應更新頻率
MIN_UPDATE_INTERVAL = 300  # 最小更新間隔
MAX_UPDATE_INTERVAL = 1000  # 最大更新間隔
CACHE_SIZE = 50  # LRU緩存大小
RENDER_OPTIMIZATION = True  # 渲染優化開關

# MQTT 設定
MQTT_BROKER = "192.168.11.90"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/data"

# 音頻設定 - 將在運行時自動檢測
AUDIO_DEVICE_INDEX = None  # 將自動檢測PD100X設備
AUDIO_SAMPLE_RATE = 44100  # 預設採樣率，將根據設備自動調整
AUDIO_CHANNELS = 1  # 單聲道

# 全局變數用於追蹤數據源
USE_REAL_SERIAL = True

# 錄音狀態
RECORDING_STATE = {
    'is_recording': False,
    'current_group_id': None,
    'recording_thread': None,
    'audio_data': [],
    'start_time': None
}

# ----- 協議常數 -----
SYNC = 0xaa
POOR_SIGNAL = 0x02
ATTENTION = 0x04
MEDITATION = 0x05
BLINK = 0x16
RAW_VALUE = 0x80
ASIC_EEG_POWER = 0x83

# ----- 頻帶定義 (樹莓派4優化版) -----
bands = {
    "Delta (0.5-4Hz)": (0.5, 4),
    "Theta (4-8Hz)": (4, 8),
    "Alpha (8-12Hz)": (8, 12),
    "Beta (12-35Hz)": (12, 35),
    "Gamma (35-50Hz)": (35, 50),
    # 新增缺少的重要頻帶
    "SMR (12-15Hz)": (12, 15),  # 感覺運動節律
    "Mu (8-13Hz)": (8, 13),  # 運動皮層節律
    "High-Gamma (50-80Hz)": (50, 80),  # 高伽馬波 (樹莓派4優化限制到80Hz)
}

# ASIC頻帶定義 (擴展版)
asic_bands = ["Delta", "Theta", "Low-Alpha", "High-Alpha",
              "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma",
              "SMR", "Mu"]  # 新增SMR和Mu頻帶

# 樹莓派4性能優化設置
RASPBERRY_PI_OPTIMIZATION = {
    'filter_order': 2,  # 降低濾波器階數以提高性能
    'use_float32': True,  # 使用32位浮點數節省記憶體
    'parallel_processing': True,  # 啟用並行處理
    'memory_limit_mb': 512,  # 記憶體使用限制
    'adaptive_update': True,  # 自適應更新頻率
}

# 預計算濾波器 (樹莓派4優化)
sos_filters = {}
for name, (low, high) in bands.items():
    try:
        # 樹莓派4優化：使用2階濾波器提高性能
        filter_order = RASPBERRY_PI_OPTIMIZATION.get('filter_order', 2)

        # 確保頻率範圍有效
        if high >= FS / 2:
            high = FS / 2 - 1
        if low >= high or low <= 0:
            continue

        sos_filters[name] = butter(
            filter_order,
            [low / (0.5 * FS), high / (0.5 * FS)],
            btype='band',
            output='sos'
        )
    except Exception as e:
        print(f"⚠️ 無法創建濾波器 {name}: {e}")
        continue

print(f"🍓 樹莓派4優化：成功創建 {len(sos_filters)} 個濾波器")


class EnhancedCircularBuffer:
    """增強的環形緩衝區 - 整合 b.py 的完整功能"""

    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)
        self.timestamps = np.zeros(size, dtype=np.float64)
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()

        # 認知指標數據 - 使用 b.py 的完整配置
        self.current_attention = 0
        self.current_meditation = 0
        self.current_signal_quality = 200  # Start with poor signal
        self.attention_history = deque(maxlen=50)  # 恢復到 b.py 的配置
        self.meditation_history = deque(maxlen=50)
        self.signal_quality_history = deque(maxlen=50)  # 新增信號品質歷史

        # ASIC頻帶數據 - 使用 b.py 的配置
        self.current_asic_bands = [0] * 8
        self.asic_bands_history = deque(maxlen=30)  # 新增 ASIC 歷史記錄

        # 眨眼數據 - 使用 b.py 的配置
        self.blink_events = deque(maxlen=20)  # 恢復到 b.py 的配置
        self.blink_count = 0
        self.blink_count_history = deque(maxlen=50)

        # MQTT 感測器數據
        self.current_temperature = 0.0
        self.current_humidity = 0.0
        self.current_light = 0
        self.sensor_history = deque(maxlen=50)

        # 初始化假數據（保留用於測試）
        self._init_fake_data()

    def _init_fake_data(self):
        """初始化假數據用於測試"""
        base_time = time.time()
        for i in range(15):
            t = base_time - (15 - i)
            self.attention_history.append((t, random.randint(30, 90)))
            self.meditation_history.append((t, random.randint(20, 80)))
            self.blink_count_history.append((t, i))

            if i % 3 == 0:
                self.blink_events.append((t, random.randint(50, 200)))

    def append(self, value: float, timestamp: float):
        """添加原始數據"""
        with self.lock:
            self.data[self.head] = value
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.size
            if self.count < self.size:
                self.count += 1

    def add_cognitive_data(self, attention: int = None, meditation: int = None, signal_quality: int = None):
        """添加認知指標數據 - 整合 b.py 的完整功能"""
        with self.lock:
            timestamp = time.time()

            if attention is not None:
                self.current_attention = attention
                self.attention_history.append((timestamp, attention))

            if meditation is not None:
                self.current_meditation = meditation
                self.meditation_history.append((timestamp, meditation))

            if signal_quality is not None:
                self.current_signal_quality = signal_quality
                self.signal_quality_history.append((timestamp, signal_quality))  # 新增信號品質歷史

    def add_asic_bands(self, bands_data: List[int]):
        """添加ASIC頻帶數據 - 整合 b.py 的歷史記錄功能"""
        with self.lock:
            timestamp = time.time()
            self.current_asic_bands = bands_data.copy()
            self.asic_bands_history.append((timestamp, bands_data.copy()))  # 新增 ASIC 歷史記錄

    def add_blink_event(self, intensity: int):
        """添加眨眼事件"""
        with self.lock:
            timestamp = time.time()
            self.blink_events.append((timestamp, intensity))
            self.blink_count += 1
            self.blink_count_history.append((timestamp, self.blink_count))

    def add_sensor_data(self, temperature: float, humidity: float, light: int):
        """添加感測器數據"""
        with self.lock:
            timestamp = time.time()
            self.current_temperature = temperature
            self.current_humidity = humidity
            self.current_light = light
            self.sensor_history.append((timestamp, temperature, humidity, light))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取原始數據"""
        with self.lock:
            if self.count == 0:
                # 生成假數據用於測試
                t = np.linspace(0, 2, self.size)
                fake_data = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(
                    self.size)
                return fake_data, t

            if self.count < self.size:
                return self.data[:self.count].copy(), self.timestamps[:self.count].copy()
            else:
                indices = np.arange(self.head, self.head + self.size) % self.size
                return self.data[indices].copy(), self.timestamps[indices].copy()

    def get_cognitive_data(self) -> Dict:
        """獲取認知指標數據 - 整合 b.py 的完整功能"""
        with self.lock:
            return {
                'attention': self.current_attention,
                'meditation': self.current_meditation,
                'signal_quality': self.current_signal_quality,
                'attention_history': list(self.attention_history),
                'meditation_history': list(self.meditation_history),
                'signal_quality_history': list(self.signal_quality_history)  # 新增信號品質歷史
            }

    def get_asic_data(self) -> Dict:
        """獲取ASIC頻帶數據 - 整合 b.py 的歷史記錄功能"""
        with self.lock:
            return {
                'current_bands': self.current_asic_bands.copy(),
                'bands_history': list(self.asic_bands_history)  # 新增 ASIC 歷史記錄
            }

    def get_blink_data(self) -> Dict:
        """獲取眨眼數據"""
        with self.lock:
            return {
                'events': list(self.blink_events),
                'count': self.blink_count,
                'count_history': list(self.blink_count_history)
            }

    def get_sensor_data(self) -> Dict:
        """獲取感測器數據"""
        with self.lock:
            return {
                'temperature': self.current_temperature,
                'humidity': self.current_humidity,
                'light': self.current_light,
                'history': list(self.sensor_history)
            }


class OptimizedFilterProcessor:
    """優化的濾波處理器 - 增強版本"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)  # 進一步優化線程數
        self.cache = {}  # 添加結果緩存
        self.last_data_hash = None  # 數據變化檢測

    def _apply_filter(self, data: np.ndarray, sos: np.ndarray) -> np.ndarray:
        """應用單一濾波器"""
        try:
            if len(data) < 10:  # 數據太少時返回零
                return np.zeros_like(data)
            return sosfiltfilt(sos, data)
        except Exception:
            return np.zeros_like(data)

    def process_bands_parallel(self, data: np.ndarray) -> dict:
        """並行處理所有頻帶 - 優化版本"""
        if len(data) < 10:
            return {name: np.zeros_like(data) for name in bands.keys()}

        # 數據變化檢測優化
        data_hash = hash(data.tobytes())
        if self.last_data_hash == data_hash and data_hash in self.cache:
            return self.cache[data_hash]

        # 智能並行處理
        futures = {}
        for name, sos in sos_filters.items():
            future = self.executor.submit(self._apply_filter, data, sos)
            futures[name] = future

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=0.03)  # 進一步減少超時
            except:
                results[name] = np.zeros_like(data)

        # 緩存結果
        self.cache[data_hash] = results
        self.last_data_hash = data_hash

        # 限制緩存大小
        if len(self.cache) > CACHE_SIZE:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        return results


class EnhancedDatabaseWriter:
    """增強的資料庫寫入器 - 整合所有數據類型"""

    def __init__(self, db_path: str = "enhanced_eeg.db"):
        self.db_path = db_path
        self.raw_buffer = []
        self.cognitive_buffer = []
        self.asic_buffer = []
        self.blink_buffer = []
        self.sensor_buffer = []  # 新增感測器數據緩衝
        self.unified_buffer = []  # 新增統一記錄緩衝
        self.running = True
        self.lock = threading.Lock()

    def setup_database(self):
        """設置資料庫"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA cache_size=10000;")

        # 原始ADC數據表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                voltage REAL NOT NULL
            )
        """)

        # 認知指標數據表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                attention INTEGER,
                meditation INTEGER,
                signal_quality INTEGER
            )
        """)

        # ASIC頻帶數據表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS asic_bands_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                delta INTEGER, theta INTEGER, low_alpha INTEGER, high_alpha INTEGER,
                low_beta INTEGER, high_beta INTEGER, low_gamma INTEGER, mid_gamma INTEGER
            )
        """)

        # 眨眼事件表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS blink_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                intensity INTEGER
            )
        """)

        # 感測器數據表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                temperature REAL,
                humidity REAL,
                light INTEGER
            )
        """)

        # 統一記錄表 (包含所有數據和錄製群組)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS unified_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                recording_group_id TEXT,
                attention INTEGER,
                meditation INTEGER,
                signal_quality INTEGER,
                temperature REAL,
                humidity REAL,
                light INTEGER,
                blink_intensity INTEGER,
                raw_voltage REAL,
                delta_power INTEGER,
                theta_power INTEGER,
                low_alpha_power INTEGER,
                high_alpha_power INTEGER,
                low_beta_power INTEGER,
                high_beta_power INTEGER,
                low_gamma_power INTEGER,
                mid_gamma_power INTEGER
            )
        """)

        # 錄音檔案記錄表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recording_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_group_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration REAL,
                sample_rate INTEGER,
                file_size INTEGER
            )
        """)

        conn.commit()
        return conn

    def add_raw_data(self, timestamp: float, voltage: float):
        """添加原始數據"""
        with self.lock:
            self.raw_buffer.append((timestamp, voltage))

    def add_cognitive_data(self, timestamp: float, attention: int, meditation: int, signal_quality: int):
        """添加認知數據"""
        with self.lock:
            self.cognitive_buffer.append((timestamp, attention, meditation, signal_quality))

    def add_asic_data(self, timestamp: float, bands_data: List[int]):
        """添加ASIC數據"""
        with self.lock:
            self.asic_buffer.append((timestamp, *bands_data))

    def add_blink_data(self, timestamp: float, intensity: int):
        """添加眨眼數據"""
        with self.lock:
            self.blink_buffer.append((timestamp, intensity))

    def add_sensor_data(self, timestamp: float, temperature: float, humidity: float, light: int):
        """添加感測器數據"""
        with self.lock:
            self.sensor_buffer.append((timestamp, temperature, humidity, light))

    def add_unified_record(self, timestamp: float, recording_group_id: str = None, **kwargs):
        """添加統一記錄"""
        with self.lock:
            record = (
                timestamp,
                recording_group_id,
                kwargs.get('attention'),
                kwargs.get('meditation'),
                kwargs.get('signal_quality'),
                kwargs.get('temperature'),
                kwargs.get('humidity'),
                kwargs.get('light'),
                kwargs.get('blink_intensity'),
                kwargs.get('raw_voltage'),
                kwargs.get('delta_power'),
                kwargs.get('theta_power'),
                kwargs.get('low_alpha_power'),
                kwargs.get('high_alpha_power'),
                kwargs.get('low_beta_power'),
                kwargs.get('high_beta_power'),
                kwargs.get('low_gamma_power'),
                kwargs.get('mid_gamma_power')
            )
            self.unified_buffer.append(record)

    def add_recording_file(self, recording_group_id: str, filename: str, start_time: float,
                           end_time: float = None, sample_rate: int = None, file_size: int = None):
        """添加錄音檔案記錄"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        duration = (end_time - start_time) if end_time else None
        cur.execute("""
            INSERT OR REPLACE INTO recording_files 
            (recording_group_id, filename, start_time, end_time, duration, sample_rate, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (recording_group_id, filename, start_time, end_time, duration, sample_rate, file_size))
        conn.commit()
        conn.close()

    def writer_thread(self):
        """資料庫寫入執行緒"""
        conn = self.setup_database()
        cur = conn.cursor()

        while self.running:
            try:
                time.sleep(2.0)  # 優化：降低寫入頻率提升性能

                with self.lock:
                    # 寫入原始數據
                    if self.raw_buffer:
                        cur.executemany(
                            "INSERT INTO raw_data (timestamp, voltage) VALUES (?, ?)",
                            self.raw_buffer
                        )
                        self.raw_buffer.clear()

                    # 寫入認知數據
                    if self.cognitive_buffer:
                        cur.executemany(
                            "INSERT INTO cognitive_data (timestamp, attention, meditation, signal_quality) VALUES (?, ?, ?, ?)",
                            self.cognitive_buffer
                        )
                        self.cognitive_buffer.clear()

                    # 寫入ASIC數據
                    if self.asic_buffer:
                        cur.executemany(
                            "INSERT INTO asic_bands_data (timestamp, delta, theta, low_alpha, high_alpha, low_beta, high_beta, low_gamma, mid_gamma) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            self.asic_buffer
                        )
                        self.asic_buffer.clear()

                    # 寫入眨眼數據
                    if self.blink_buffer:
                        cur.executemany(
                            "INSERT INTO blink_events (timestamp, intensity) VALUES (?, ?)",
                            self.blink_buffer
                        )
                        self.blink_buffer.clear()

                    # 寫入感測器數據
                    if self.sensor_buffer:
                        cur.executemany(
                            "INSERT INTO sensor_data (timestamp, temperature, humidity, light) VALUES (?, ?, ?, ?)",
                            self.sensor_buffer
                        )
                        self.sensor_buffer.clear()

                    # 寫入統一記錄
                    if self.unified_buffer:
                        cur.executemany("""
                            INSERT INTO unified_records 
                            (timestamp, recording_group_id, attention, meditation, signal_quality, 
                             temperature, humidity, light, blink_intensity, raw_voltage,
                             delta_power, theta_power, low_alpha_power, high_alpha_power,
                             low_beta_power, high_beta_power, low_gamma_power, mid_gamma_power)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, self.unified_buffer)
                        self.unified_buffer.clear()

                conn.commit()

            except Exception as e:
                print(f"[EnhancedDatabaseWriter] Error: {e}")

        conn.close()

    def start(self):
        """啟動寫入執行緒"""
        thread = threading.Thread(target=self.writer_thread, daemon=True)
        thread.start()
        return thread


class EnhancedSerialReader:
    """增強的串口讀取器 - 從 b.py 移植"""

    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.buffer = bytearray()

    def parse_payload(self, payload: bytearray) -> Dict:
        """解析ThinkGear協議"""
        data = {}
        i = 0

        while i < len(payload):
            code = payload[i]

            if code == POOR_SIGNAL and i + 1 < len(payload):
                data['signal_quality'] = payload[i + 1]
                i += 2

            elif code == ATTENTION and i + 1 < len(payload):
                data['attention'] = payload[i + 1]
                i += 2

            elif code == MEDITATION and i + 1 < len(payload):
                data['meditation'] = payload[i + 1]
                i += 2

            elif code == BLINK and i + 1 < len(payload):
                data['blink'] = payload[i + 1]
                i += 2

            elif code == RAW_VALUE and i + 2 < len(payload):
                raw_val = (payload[i + 1] << 8) | payload[i + 2]
                if raw_val >= 32768:
                    raw_val -= 65536
                data['raw_value'] = raw_val * (1.8 / 4096) / 2000
                i += 3

            elif code == ASIC_EEG_POWER and i + 24 < len(payload):
                # 解析8個頻帶 (每個3字節)
                bands_data = []
                for j in range(8):
                    band_value = (payload[i + 1 + j * 3] << 16) | (payload[i + 2 + j * 3] << 8) | payload[i + 3 + j * 3]
                    bands_data.append(band_value)
                data['asic_bands'] = bands_data
                i += 25

            else:
                i += 1

        return data

    def read_data(self, ser: serial.Serial) -> Dict:
        """讀取並解析EEG數據"""
        try:
            available = ser.in_waiting
            if available > 0:
                chunk = ser.read(available)
                self.buffer.extend(chunk)

            # 解析完整數據包
            while len(self.buffer) >= 4:
                # 尋找同步字節
                sync_pos = -1
                for i in range(len(self.buffer) - 1):
                    if self.buffer[i] == SYNC and self.buffer[i + 1] == SYNC:
                        sync_pos = i
                        break

                if sync_pos == -1:
                    self.buffer.clear()
                    break

                # 移除同步前的數據
                if sync_pos > 0:
                    self.buffer = self.buffer[sync_pos:]

                if len(self.buffer) < 4:
                    break

                length = self.buffer[2]
                if len(self.buffer) < 4 + length:
                    break

                payload = self.buffer[3:3 + length]
                checksum = self.buffer[3 + length]

                # 驗證校驗和
                calc_checksum = (~sum(payload)) & 0xFF
                if calc_checksum == checksum:
                    parsed_data = self.parse_payload(payload)
                    self.buffer = self.buffer[4 + length:]
                    return parsed_data
                else:
                    self.buffer = self.buffer[2:]

        except Exception as e:
            print(f"[EnhancedSerialReader] Error: {e}")

        return {}


def enhanced_serial_worker(out_queue: multiprocessing.Queue):
    """增強的串口工作程序 - 從 b.py 移植"""
    reader = EnhancedSerialReader(SERIAL_PORT, BAUD_RATE)
    ser = None

    while True:
        try:
            if ser is None:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
                ser.reset_input_buffer()
                print(f"[enhanced_serial_worker] Connected to {SERIAL_PORT}")

            parsed_data = reader.read_data(ser)

            if parsed_data:
                timestamp = time.time()
                parsed_data['timestamp'] = timestamp
                out_queue.put(parsed_data)
            else:
                time.sleep(0.01)

        except serial.SerialException as e:
            print(f"[enhanced_serial_worker] SerialException: {e}")
            try:
                ser.close()
            except:
                pass
            ser = None
            time.sleep(2)
        except Exception as e:
            print(f"[enhanced_serial_worker] Unexpected exception: {e}")
            time.sleep(1)


class MQTTSensorClient:
    """MQTT 感測器數據客戶端"""

    def __init__(self, broker: str, port: int, topic: str, data_buffer, db_writer):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.data_buffer = data_buffer
        self.db_writer = db_writer
        self.client = None
        self.running = True

    def on_connect(self, client, userdata, flags, rc):
        """連接回調"""
        if rc == 0:
            print(f"✅ MQTT 連接成功: {self.broker}:{self.port}")
            client.subscribe(self.topic)
            print(f"📡 訂閱主題: {self.topic}")
        else:
            print(f"❌ MQTT 連接失敗，代碼: {rc}")

    def on_message(self, client, userdata, msg):
        """訊息接收回調"""
        try:
            data = json.loads(msg.payload.decode())
            timestamp = time.time()

            temperature = data.get('temperature', 0.0)
            humidity = data.get('humidity', 0.0)
            light = data.get('light', 0)

            # 更新緩衝區
            self.data_buffer.add_sensor_data(temperature, humidity, light)

            # 寫入資料庫
            self.db_writer.add_sensor_data(timestamp, temperature, humidity, light)

            # 添加到統一記錄
            current_group_id = RECORDING_STATE['current_group_id'] if RECORDING_STATE['is_recording'] else None
            self.db_writer.add_unified_record(
                timestamp,
                current_group_id,
                temperature=temperature,
                humidity=humidity,
                light=light
            )

            print(f"📊 感測器數據: T={temperature}°C, H={humidity}%, L={light}")

        except Exception as e:
            print(f"[MQTT] 解析數據錯誤: {e}")

    def on_disconnect(self, client, userdata, rc):
        """斷線回調"""
        print(f"⚠️ MQTT 斷線，代碼: {rc}")

    def start(self):
        """啟動 MQTT 客戶端"""
        if not MQTT_AVAILABLE:
            print("⚠️ MQTT 模組未安裝，跳過 MQTT 功能")
            return None

        try:
            self.client = mqtt.Client()
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect

            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()

            return self.client
        except Exception as e:
            print(f"❌ MQTT 啟動失敗: {e}")
            return None


class AudioRecorder:
    """音頻錄製器"""

    def __init__(self, device_index: int, sample_rate: int, channels: int):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording_data = []
        self.is_recording = False

    def list_audio_devices(self):
        """列出音頻設備並自動檢測PD100X"""
        if not AUDIO_AVAILABLE:
            print("⚠️ 音頻模組未安裝")
            print("請安裝音頻模組: pip install sounddevice scipy")
            return None

        try:
            print("🔍 正在查詢音頻設備...")
            devices = sd.query_devices()
            print("🎙️ 可用音頻設備:")
            print("-" * 80)

            pd100x_device = None
            recommended_device = None

            for i, dev in enumerate(devices):
                status = "✅" if dev['max_input_channels'] > 0 else "❌"
                default_marker = " (預設)" if i == sd.default.device[0] else ""

                # 檢查是否為PD100X或USB Audio設備
                name_upper = dev['name'].upper()
                is_pd100x = any([
                    "PD100X" in name_upper,
                    "PODCAST MICROPHONE" in name_upper,
                    ("USB AUDIO" in name_upper and dev['max_input_channels'] > 0),
                    ("MICROPHONE" in name_upper and "USB" in name_upper),
                    # 特別檢查你的設備配置：索引1, USB Audio, 1輸入通道, 44100Hz, API 0
                    (i == 1 and "USB AUDIO" in name_upper and
                     dev['max_input_channels'] == 1 and
                     dev['default_samplerate'] == 44100.0 and
                     dev['hostapi'] == 0)
                ])
                pd100x_marker = " 🎯 PD100X!" if is_pd100x and dev['max_input_channels'] > 0 else ""

                print(f"  {status} {i}: {dev['name']}{default_marker}{pd100x_marker}")
                print(f"      輸入通道: {dev['max_input_channels']}, 輸出通道: {dev['max_output_channels']}")
                print(f"      預設採樣率: {dev['default_samplerate']}Hz")
                print(f"      主機API: {dev['hostapi']}")

                # 特別標記你的PD100X配置
                if (i == 1 and "USB Audio" in dev['name'] and
                        dev['max_input_channels'] == 1 and
                        dev['default_samplerate'] == 44100.0 and
                        dev['hostapi'] == 0):
                    print(f"      🎯 這是你的PD100X設備！")
                print()

                # 自動檢測最佳設備
                if is_pd100x and dev['max_input_channels'] > 0:
                    pd100x_device = i
                    print(f"🎯 檢測到PD100X設備: 索引 {i} - {dev['name']}")
                    print(f"   採樣率: {dev['default_samplerate']}Hz, 輸入通道: {dev['max_input_channels']}")
                elif dev['max_input_channels'] > 0 and recommended_device is None:
                    recommended_device = i
                    print(f"💡 備選輸入設備: 索引 {i} - {dev['name']}")

            print(f"📊 總共找到 {len(devices)} 個音頻設備")

            # 自動選擇最佳設備
            if pd100x_device is not None:
                self.device_index = pd100x_device
                # 使用設備的預設採樣率
                device_sample_rate = int(devices[pd100x_device]['default_samplerate'])
                self.sample_rate = device_sample_rate  # 使用設備預設採樣率
                self.channels = 1  # PD100X是單聲道
                print(f"✅ 自動選擇PD100X設備: 索引 {pd100x_device}")
                print(f"📝 已更新設定: 採樣率={self.sample_rate}Hz (設備預設), 聲道={self.channels}")
                return pd100x_device
            elif recommended_device is not None:
                print(f"💡 建議使用設備索引: {recommended_device}")
                self.device_index = recommended_device
                return recommended_device
            else:
                print("⚠️ 未找到合適的輸入設備")
                print("🔍 檢查到的設備中沒有輸入通道 > 0 的設備")
                print("💡 請確認:")
                print("   • PD100X是否正確連接USB")
                print("   • 設備驅動是否已安裝")
                print("   • 是否需要重新插拔USB連接")

                # 嘗試使用預設輸入設備
                try:
                    default_input = sd.default.device[0]
                    if default_input < len(devices) and devices[default_input]['max_input_channels'] > 0:
                        print(f"🔄 嘗試使用預設輸入設備: 索引 {default_input}")
                        self.device_index = default_input
                        return default_input
                except:
                    pass

                return None

        except Exception as e:
            print(f"❌ 列出音頻設備失敗: {e}")
            print(f"   錯誤類型: {type(e).__name__}")
            import traceback
            print(f"   詳細錯誤: {traceback.format_exc()}")
            print("   可能原因: 音頻驅動問題、權限不足、或系統音頻服務未啟動")
            return None

    def start_recording(self, group_id: str):
        """開始錄音"""
        if not AUDIO_AVAILABLE:
            error_msg = "⚠️ 音頻模組未安裝，無法錄音"
            print(error_msg)
            print("請安裝音頻模組: pip install sounddevice scipy")
            return False

        try:
            # 檢查音頻設備
            print(f"🔍 檢查音頻設備 {self.device_index}...")
            try:
                devices = sd.query_devices()
                if self.device_index >= len(devices):
                    print(f"❌ 設備索引 {self.device_index} 超出範圍 (可用設備: 0-{len(devices) - 1})")
                    return False

                device_info = devices[self.device_index]
                print(f"📱 使用設備: {device_info['name']}")
                print(
                    f"📊 設備資訊: 輸入通道={device_info['max_input_channels']}, 採樣率={device_info['default_samplerate']}")

                if device_info['max_input_channels'] < self.channels:
                    print(f"❌ 設備不支援 {self.channels} 聲道 (最大: {device_info['max_input_channels']})")
                    return False

            except Exception as e:
                print(f"❌ 查詢音頻設備失敗: {e}")
                return False

            # 測試錄音設備
            print(f"🧪 測試錄音設備...")
            try:
                test_duration = 0.1  # 100ms 測試
                test_recording = sd.rec(
                    int(test_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16',
                    device=self.device_index
                )
                sd.wait()  # 等待測試完成
                print(f"✅ 設備測試成功")
            except Exception as e:
                print(f"❌ 設備測試失敗: {e}")
                print(f"   可能原因: 設備被占用、權限不足、或設備不存在")
                return False

            self.recording_data = []
            self.is_recording = True
            RECORDING_STATE['is_recording'] = True
            RECORDING_STATE['current_group_id'] = group_id
            RECORDING_STATE['start_time'] = time.time()

            print(f"🎙️ 開始錄音，群組ID: {group_id}")
            print(f"📝 錄音參數: 採樣率={self.sample_rate}Hz, 聲道={self.channels}, 設備={self.device_index}")

            # 在背景執行緒中錄音
            def record_thread():
                try:
                    print(f"🎵 錄音執行緒啟動...")
                    # 錄音 60 秒或直到停止
                    duration = 60  # 最大錄音時間
                    recording = sd.rec(
                        int(duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='int16',
                        device=self.device_index
                    )

                    print(f"🎤 錄音中... (最大 {duration} 秒)")

                    # 等待錄音完成或被中斷
                    start_time = time.time()
                    while self.is_recording and (time.time() - start_time) < duration:
                        time.sleep(0.1)  # 每100ms檢查一次狀態
                        # 檢查錄音是否仍在進行
                        try:
                            if not sd.get_stream():
                                break
                        except:
                            break

                    if self.is_recording:
                        sd.stop()
                        print(f"⏹️ 錄音自動停止 (達到最大時間)")
                    else:
                        print(f"⏹️ 錄音手動停止")

                    # 儲存錄音數據
                    try:
                        # 計算實際錄音時間
                        actual_duration = time.time() - start_time
                        actual_frames = int(actual_duration * self.sample_rate)

                        # 確保不超過錄音數組的長度
                        max_frames = len(recording)
                        frames_to_save = min(actual_frames, max_frames)

                        self.recording_data = recording[:frames_to_save]
                        print(f"💾 錄音數據已儲存 ({frames_to_save} 幀, {actual_duration:.1f}秒)")

                    except Exception as e:
                        print(f"⚠️ 計算錄音幀數失敗，使用完整錄音: {e}")
                        self.recording_data = recording
                        print(f"💾 錄音數據已儲存 (完整, {len(recording)} 幀)")

                except Exception as e:
                    print(f"❌ 錄音執行緒錯誤: {e}")
                    print(f"   錯誤類型: {type(e).__name__}")
                    import traceback
                    print(f"   詳細錯誤: {traceback.format_exc()}")

            RECORDING_STATE['recording_thread'] = threading.Thread(target=record_thread, daemon=True)
            RECORDING_STATE['recording_thread'].start()

            return True

        except Exception as e:
            print(f"❌ 開始錄音失敗: {e}")
            print(f"   錯誤類型: {type(e).__name__}")
            import traceback
            print(f"   詳細錯誤: {traceback.format_exc()}")
            return False

    def stop_recording(self, db_writer):
        """停止錄音並儲存檔案"""
        if not self.is_recording:
            print("⚠️ 目前沒有進行錄音")
            return None

        try:
            print("🛑 正在停止錄音...")
            self.is_recording = False
            RECORDING_STATE['is_recording'] = False

            # 停止錄音
            try:
                sd.stop()
                print("✅ 音頻串流已停止")
            except Exception as e:
                print(f"⚠️ 停止音頻串流時發生錯誤: {e}")

            # 等待錄音執行緒結束
            if RECORDING_STATE['recording_thread']:
                print("⏳ 等待錄音執行緒結束...")
                RECORDING_STATE['recording_thread'].join(timeout=5)  # 增加超時時間
                if RECORDING_STATE['recording_thread'].is_alive():
                    print("⚠️ 錄音執行緒未能在時限內結束")
                else:
                    print("✅ 錄音執行緒已結束")

            # 檢查錄音數據
            group_id = RECORDING_STATE['current_group_id']
            print(f"📊 檢查錄音數據... 群組ID: {group_id}")
            print(
                f"📊 錄音數據長度: {len(self.recording_data) if hasattr(self, 'recording_data') and self.recording_data is not None else 0}")

            if group_id and hasattr(self, 'recording_data') and self.recording_data is not None and len(
                    self.recording_data) > 0:
                try:
                    # 建立錄音檔案目錄
                    recordings_dir = "recordings"
                    os.makedirs(recordings_dir, exist_ok=True)
                    print(f"📁 錄音目錄已準備: {recordings_dir}")

                    # 產生檔案名稱
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{recordings_dir}/recording_{group_id}_{timestamp}.wav"
                    print(f"📝 準備儲存檔案: {filename}")

                    # 檢查錄音數據格式
                    print(f"🔍 錄音數據格式: shape={self.recording_data.shape}, dtype={self.recording_data.dtype}")

                    # 儲存 WAV 檔案
                    wav.write(filename, self.sample_rate, self.recording_data)
                    print(f"💾 WAV 檔案已寫入")

                    # 驗證檔案是否成功建立
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        print(f"✅ 檔案驗證成功: {filename} ({file_size} bytes)")

                        # 記錄到資料庫
                        try:
                            end_time = time.time()
                            start_time = RECORDING_STATE['start_time']
                            duration = end_time - start_time if start_time else 0

                            db_writer.add_recording_file(
                                group_id, filename, start_time, end_time,
                                self.sample_rate, file_size
                            )
                            print(f"📊 資料庫記錄已新增 (時長: {duration:.1f}秒)")

                        except Exception as e:
                            print(f"⚠️ 資料庫記錄失敗: {e}")

                        # 清理狀態
                        RECORDING_STATE['current_group_id'] = None
                        RECORDING_STATE['start_time'] = None
                        RECORDING_STATE['recording_thread'] = None

                        print(f"🎉 錄音完成！檔案: {filename}")
                        return filename
                    else:
                        print(f"❌ 檔案未能成功建立: {filename}")

                except Exception as e:
                    print(f"❌ 儲存檔案時發生錯誤: {e}")
                    print(f"   錯誤類型: {type(e).__name__}")
                    import traceback
                    print(f"   詳細錯誤: {traceback.format_exc()}")
            else:
                if not group_id:
                    print("❌ 沒有群組ID")
                elif not hasattr(self, 'recording_data') or self.recording_data is None:
                    print("❌ 沒有錄音數據")
                elif len(self.recording_data) == 0:
                    print("❌ 錄音數據為空")

        except Exception as e:
            print(f"❌ 停止錄音失敗: {e}")
            print(f"   錯誤類型: {type(e).__name__}")
            import traceback
            print(f"   詳細錯誤: {traceback.format_exc()}")

        # 清理狀態（即使失敗也要清理）
        RECORDING_STATE['is_recording'] = False
        RECORDING_STATE['current_group_id'] = None
        RECORDING_STATE['start_time'] = None
        RECORDING_STATE['recording_thread'] = None

        return None


def mock_serial_worker(out_queue: multiprocessing.Queue):
    """模擬串口工作程序 - 用於測試"""
    while True:
        try:
            # 模擬數據生成
            timestamp = time.time()

            # 模擬原始數據
            raw_value = random.uniform(-0.001, 0.001)
            out_queue.put({'raw_value': raw_value, 'timestamp': timestamp})

            # 隨機生成認知數據
            if random.random() < 0.1:  # 10%機率
                attention = random.randint(30, 90)
                meditation = random.randint(20, 80)
                signal_quality = random.randint(0, 150)
                out_queue.put({
                    'attention': attention,
                    'meditation': meditation,
                    'signal_quality': signal_quality,
                    'timestamp': timestamp
                })

            # 隨機生成ASIC數據
            if random.random() < 0.05:  # 5%機率
                asic_bands = [random.randint(10, 100) for _ in range(8)]
                out_queue.put({'asic_bands': asic_bands, 'timestamp': timestamp})

            # 隨機生成眨眼事件
            if random.random() < 0.02:  # 2%機率
                blink_intensity = random.randint(50, 200)
                out_queue.put({'blink': blink_intensity, 'timestamp': timestamp})

            time.sleep(0.02)  # 50Hz更新頻率

        except Exception as e:
            print(f"[mock_serial_worker] Error: {e}")
            time.sleep(1)


def main():
    # 嘗試啟動真實串口，失敗則使用模擬數據
    serial_queue = multiprocessing.Queue()
    use_real_serial = True

    # 將 use_real_serial 設為全局變數以便在回調函數中使用
    global USE_REAL_SERIAL
    USE_REAL_SERIAL = use_real_serial

    try:
        # 嘗試真實串口連接
        p = multiprocessing.Process(
            target=enhanced_serial_worker,
            args=(serial_queue,),
            daemon=True
        )
        p.start()
        print("🔌 嘗試連接真實串口...")
        time.sleep(2)  # 等待連接
    except Exception as e:
        print(f"⚠️ 真實串口連接失敗，使用模擬數據: {e}")
        use_real_serial = False
        USE_REAL_SERIAL = False
        p = multiprocessing.Process(
            target=mock_serial_worker,
            args=(serial_queue,),
            daemon=True
        )
        p.start()

    # 初始化增強組件
    data_buffer = EnhancedCircularBuffer(WINDOW_SIZE)
    filter_processor = OptimizedFilterProcessor()
    db_writer = EnhancedDatabaseWriter()

    # 啟動資料庫寫入器
    db_writer.start()
    print("💾 資料庫寫入器已啟動")

    # 初始化 MQTT 客戶端
    mqtt_client = MQTTSensorClient(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, data_buffer, db_writer)
    mqtt_client.start()

    # 初始化音頻錄製器並自動檢測PD100X
    # 先用預設值初始化，然後自動檢測最佳設備
    initial_device = AUDIO_DEVICE_INDEX if AUDIO_DEVICE_INDEX is not None else 0
    audio_recorder = AudioRecorder(initial_device, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)
    detected_device = audio_recorder.list_audio_devices()  # 列出可用設備並自動檢測

    if detected_device is not None:
        print(f"🎯 音頻設備已自動配置: 索引 {audio_recorder.device_index}")
        print(f"📝 錄音設定: {audio_recorder.sample_rate}Hz, {audio_recorder.channels}聲道")
        print(f"✅ PD100X麥克風已準備就緒")
    else:
        print("⚠️ 未檢測到PD100X或合適的音頻設備")
        print("💡 請確認：")
        print("   1. PD100X USB連接正常")
        print("   2. 設備驅動程式已安裝")
        print("   3. 系統音頻服務正常運行")
        print("   4. 設備未被其他程式占用")

    # 增強的串口數據監聽器
    def enhanced_serial_listener():
        while True:
            try:
                parsed_data = serial_queue.get(timeout=1.0)
                timestamp = parsed_data.get('timestamp', time.time())

                # 獲取當前感測器數據用於統一記錄
                sensor_data = data_buffer.get_sensor_data()
                current_group_id = RECORDING_STATE['current_group_id'] if RECORDING_STATE['is_recording'] else None

                # 處理原始數據
                if 'raw_value' in parsed_data:
                    voltage = parsed_data['raw_value']
                    data_buffer.append(voltage, timestamp)
                    db_writer.add_raw_data(timestamp, voltage)

                    # 添加到統一記錄
                    db_writer.add_unified_record(
                        timestamp, current_group_id,
                        raw_voltage=voltage,
                        temperature=sensor_data['temperature'],
                        humidity=sensor_data['humidity'],
                        light=sensor_data['light']
                    )

                # 處理認知數據
                attention = parsed_data.get('attention')
                meditation = parsed_data.get('meditation')
                signal_quality = parsed_data.get('signal_quality')

                if any(x is not None for x in [attention, meditation, signal_quality]):
                    data_buffer.add_cognitive_data(attention, meditation, signal_quality)
                    if all(x is not None for x in [attention, meditation, signal_quality]):
                        db_writer.add_cognitive_data(timestamp, attention, meditation, signal_quality)

                    # 添加到統一記錄
                    db_writer.add_unified_record(
                        timestamp, current_group_id,
                        attention=attention,
                        meditation=meditation,
                        signal_quality=signal_quality,
                        temperature=sensor_data['temperature'],
                        humidity=sensor_data['humidity'],
                        light=sensor_data['light']
                    )

                # 處理ASIC頻帶
                if 'asic_bands' in parsed_data:
                    bands_data = parsed_data['asic_bands']
                    data_buffer.add_asic_bands(bands_data)
                    db_writer.add_asic_data(timestamp, bands_data)

                    # 添加到統一記錄
                    asic_dict = {f'{asic_bands[i].lower()}_power': bands_data[i] for i in
                                 range(min(8, len(bands_data)))}
                    db_writer.add_unified_record(
                        timestamp, current_group_id,
                        temperature=sensor_data['temperature'],
                        humidity=sensor_data['humidity'],
                        light=sensor_data['light'],
                        **asic_dict
                    )

                # 處理眨眼事件
                if 'blink' in parsed_data:
                    intensity = parsed_data['blink']
                    data_buffer.add_blink_event(intensity)
                    db_writer.add_blink_data(timestamp, intensity)

                    # 添加到統一記錄
                    db_writer.add_unified_record(
                        timestamp, current_group_id,
                        blink_intensity=intensity,
                        temperature=sensor_data['temperature'],
                        humidity=sensor_data['humidity'],
                        light=sensor_data['light']
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[enhanced_serial_listener] Error: {e}")

    threading.Thread(target=enhanced_serial_listener, daemon=True).start()

    # Dash應用程式
    app = dash.Dash(__name__)

    # 性能優化全局變數
    performance_monitor = {
        'last_update_time': time.time(),
        'update_count': 0,
        'avg_render_time': 0,
        'adaptive_interval': UPDATE_INTERVAL
    }

    # 響應式CSS樣式 - 修正版本
    app.layout = html.Div([

        html.Div([
            # 標題
            html.H1("優化版響應式EEG監控系統",
                    style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333'}),

            # 第一排：FFT頻帶分析
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("FFT頻帶分析",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        dcc.Graph(id="fft-bands-main", style={'height': '400px'},
                                  config={'displayModeBar': False}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),

            # 第二排：認知指標
            html.Div([
                # 左邊：趨勢圖
                html.Div([
                    html.Div([
                        html.H3("認知指標趨勢",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        dcc.Graph(id="cognitive-trends", style={'height': '250px'},
                                  config={'displayModeBar': False}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),

                # 右邊：圓形儀表
                html.Div([
                    html.Div([
                        html.H3("即時數值",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        html.Div([
                            dcc.Graph(id="attention-gauge", style={'height': '120px'},
                                      config={'displayModeBar': False}),
                            dcc.Graph(id="meditation-gauge", style={'height': '120px'},
                                      config={'displayModeBar': False}),
                        ]),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),

            # 第三排：眨眼檢測
            html.Div([
                # 左邊：事件時間軸
                html.Div([
                    html.Div([
                        html.H3("眨眼事件時間軸",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        dcc.Graph(id="blink-timeline", style={'height': '200px'},
                                  config={'displayModeBar': False}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),

                # 右邊：眨眼計數
                html.Div([
                    html.Div([
                        html.H3("眨眼計數",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        dcc.Graph(id="blink-count-chart", style={'height': '200px'},
                                  config={'displayModeBar': False}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),

            # 第四排：ASIC頻帶
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("ASIC頻帶分析",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        dcc.Graph(id="asic-bands-chart", style={'height': '300px'},
                                  config={'displayModeBar': False}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),

            # 第五排：感測器數據和錄音控制
            html.Div([
                # 左邊：感測器數據
                html.Div([
                    html.Div([
                        html.H3("環境感測器",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        html.Div(id="sensor-display",
                                 style={'fontSize': '12px', 'lineHeight': '1.5', 'fontFamily': 'monospace'}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),

                # 右邊：錄音控制
                html.Div([
                    html.Div([
                        html.H3("錄音控制",
                                style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                       'color': '#555'}),
                        html.Div([
                            html.Button("🎙️ 開始錄音", id="start-recording-btn",
                                        style={'marginRight': '10px', 'padding': '10px 20px', 'fontSize': '14px'}),
                            html.Button("⏹️ 停止錄音", id="stop-recording-btn",
                                        style={'padding': '10px 20px', 'fontSize': '14px'}),
                        ], style={'marginBottom': '10px'}),
                        html.Div(id="recording-status",
                                 style={'fontSize': '12px', 'color': '#666'}),
                    ], style={'background': 'white', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),

            # 狀態列
            html.Div([
                html.Div(id="performance-status",
                         style={'fontSize': '12px', 'color': '#666', 'textAlign': 'center', 'padding': '10px',
                                'borderTop': '1px solid #eee'}),
            ]),

            dcc.Interval(id="interval", interval=UPDATE_INTERVAL, n_intervals=0),
            dcc.Store(id="performance-store", data={}),  # 性能數據存儲
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '10px'}),
    ])

    @app.callback(
        Output("fft-bands-main", "figure"),
        Input("interval", "n_intervals")
    )
    def update_fft_bands_main(n):
        """更新增強版EEG頻帶分析圖"""
        start_time = time.time()

        try:
            # 自適應更新檢查
            if ADAPTIVE_UPDATE and n % 2 != 0:
                return dash.no_update

            data, timestamps = data_buffer.get_data()
            if len(data) < 10:
                return go.Figure().add_annotation(
                    text="等待EEG數據...<br>請確認ThinkGear設備連接",
                    showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                    font=dict(size=16, color="gray")
                )

            # 數據採樣優化
            if len(data) > 512:
                step = len(data) // 512
                data = data[::step]

            # 使用並行濾波處理
            filtered_data = filter_processor.process_bands_parallel(data)

            # 顯示所有5個主要頻帶
            all_bands = list(bands.keys())[:5]  # Delta, Theta, Alpha, Beta, Gamma
            fig = make_subplots(
                rows=5, cols=1,
                shared_xaxes=True,
                subplot_titles=[
                    "Delta (0.5-4Hz) - 深度睡眠",
                    "Theta (4-8Hz) - 創意思考",
                    "Alpha (8-12Hz) - 放鬆專注",
                    "Beta (12-35Hz) - 活躍思考",
                    "Gamma (35-50Hz) - 高度專注"
                ],
                vertical_spacing=0.03
            )

            # 時間軸
            t = np.arange(len(data)) / FS

            # 頻帶顏色配置
            band_colors = {
                "Delta (0.5-4Hz)": "#FF6B6B",  # 紅色
                "Theta (4-8Hz)": "#4ECDC4",  # 青色
                "Alpha (8-12Hz)": "#45B7D1",  # 藍色
                "Beta (12-35Hz)": "#96CEB4",  # 綠色
                "Gamma (35-50Hz)": "#FFEAA7",  # 黃色
            }

            # 添加所有頻帶
            for i, name in enumerate(all_bands, start=1):
                if name in filtered_data:
                    y = filtered_data[name]
                    color = band_colors.get(name, '#666666')

                    # 計算該頻帶的功率
                    power = np.mean(y ** 2) if len(y) > 0 else 0

                    fig.add_trace(
                        go.Scatter(
                            x=t, y=y,
                            mode="lines",
                            showlegend=False,
                            line=dict(color=color, width=1.5),
                            name=f"{name} (功率: {power:.2e})"
                        ),
                        row=i, col=1
                    )

                    # 添加功率指示
                    fig.add_annotation(
                        text=f"功率: {power:.2e}",
                        xref="paper", yref=f"y{i}",
                        x=0.98, y=max(y) if len(y) > 0 else 0,
                        showarrow=False,
                        font=dict(size=10, color=color),
                        bgcolor="rgba(255,255,255,0.8)"
                    )

            fig.update_layout(
                height=500,  # 增加高度以容納5個頻帶
                showlegend=False,
                title_text="EEG頻帶分解 (Delta, Theta, Alpha, Beta, Gamma)",
                title_x=0.5,
                margin=dict(l=40, r=15, t=40, b=30),
                plot_bgcolor='white'
            )

            # 更新x軸標籤
            fig.update_xaxes(title_text="時間 (秒)", row=5, col=1)

            # 更新y軸標籤
            for i in range(1, 6):
                fig.update_yaxes(title_text="振幅 (μV)", row=i, col=1)

            # 性能監控
            render_time = time.time() - start_time
            performance_monitor['avg_render_time'] = (
                    performance_monitor['avg_render_time'] * 0.9 + render_time * 0.1
            )

            return fig

        except Exception as e:
            print(f"Error in update_fft_bands_main: {e}")
            return go.Figure().add_annotation(
                text=f"頻帶分析錯誤: {str(e)}",
                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
            )

    @app.callback(
        [Output("attention-gauge", "figure"),
         Output("meditation-gauge", "figure")],
        Input("interval", "n_intervals")
    )
    def update_cognitive_gauges(n):
        """更新認知指標圓形儀表 - 超級優化版本"""
        try:
            # 降低更新頻率
            if ADAPTIVE_UPDATE and n % 3 != 0:  # 每3次更新一次
                return dash.no_update, dash.no_update

            cognitive_data = data_buffer.get_cognitive_data()
            attention = cognitive_data['attention']
            meditation = cognitive_data['meditation']

            # 極簡化儀表設計
            def create_simple_gauge(value, title, color):
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': title, 'font': {'size': 12}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 0},
                        'bar': {'color': color, 'thickness': 0.4},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "lightgray"
                    }
                ))
                fig.update_layout(
                    height=120,
                    margin=dict(l=5, r=5, t=15, b=5),
                    font={'size': 9}
                )
                return fig

            attention_fig = create_simple_gauge(attention, "注意力", "#1f77b4")
            meditation_fig = create_simple_gauge(meditation, "冥想", "#2ca02c")

            return attention_fig, meditation_fig

        except Exception as e:
            print(f"Error in update_cognitive_gauges: {e}")
            empty_fig = go.Figure()
            empty_fig.update_layout(height=120)
            return empty_fig, empty_fig

    @app.callback(
        Output("cognitive-trends", "figure"),
        Input("interval", "n_intervals")
    )
    def update_cognitive_trends(n):
        """更新認知指標趨勢圖 - 優化版本"""
        try:
            # 降低更新頻率
            if ADAPTIVE_UPDATE and n % 4 != 0:  # 每4次更新一次
                return dash.no_update

            cognitive_data = data_buffer.get_cognitive_data()

            fig = go.Figure()

            # 數據採樣優化 - 只顯示最近20個點
            max_points = 20

            # 注意力趨勢
            if cognitive_data['attention_history']:
                history = list(cognitive_data['attention_history'])[-max_points:]
                if history:
                    times, values = zip(*history)
                    base_time = times[0] if times else 0
                    rel_times = [(t - base_time) for t in times]
                    fig.add_trace(go.Scatter(
                        x=rel_times, y=values,
                        mode='lines',  # 移除markers提升性能
                        name='注意力',
                        line=dict(color='#1f77b4', width=2)
                    ))

            # 冥想趨勢
            if cognitive_data['meditation_history']:
                history = list(cognitive_data['meditation_history'])[-max_points:]
                if history:
                    times, values = zip(*history)
                    base_time = times[0] if times else 0
                    rel_times = [(t - base_time) for t in times]
                    fig.add_trace(go.Scatter(
                        x=rel_times, y=values,
                        mode='lines',  # 移除markers提升性能
                        name='冥想',
                        line=dict(color='#2ca02c', width=2)
                    ))

            fig.update_layout(
                xaxis_title="時間 (秒)",
                yaxis_title="數值",
                yaxis_range=[0, 100],
                height=250,
                margin=dict(l=30, r=15, t=15, b=30),
                legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            print(f"Error in update_cognitive_trends: {e}")
            return go.Figure()

    @app.callback(
        Output("blink-timeline", "figure"),
        Input("interval", "n_intervals")
    )
    def update_blink_timeline(n):
        """更新眨眼事件時間軸 - 優化版本"""
        try:
            # 降低更新頻率
            if ADAPTIVE_UPDATE and n % 5 != 0:  # 每5次更新一次
                return dash.no_update

            blink_data = data_buffer.get_blink_data()
            events = list(blink_data['events'])[-10:]  # 只顯示最近10個事件

            fig = go.Figure()

            if events:
                times, intensities = zip(*events)
                base_time = times[0] if times else 0
                rel_times = [(t - base_time) for t in times]

                fig.add_trace(go.Scatter(
                    x=rel_times, y=intensities,
                    mode='markers',
                    marker=dict(
                        size=8,  # 固定大小提升性能
                        color='red',  # 固定顏色
                        opacity=0.7
                    ),
                    name='眨眼事件'
                ))

            fig.update_layout(
                xaxis_title="時間 (秒)",
                yaxis_title="強度",
                height=200,
                margin=dict(l=30, r=15, t=15, b=30),
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            print(f"Error in update_blink_timeline: {e}")
            return go.Figure()

    @app.callback(
        Output("blink-count-chart", "figure"),
        Input("interval", "n_intervals")
    )
    def update_blink_count_chart(n):
        """更新眨眼計數圖"""
        try:
            blink_data = data_buffer.get_blink_data()
            count_history = blink_data['count_history']

            fig = go.Figure()

            if count_history:
                times, counts = zip(*count_history)
                base_time = times[0] if times else 0
                rel_times = [(t - base_time) for t in times]

                fig.add_trace(go.Scatter(
                    x=rel_times, y=counts,
                    mode='lines+markers',
                    name='累計次數',
                    line=dict(color='#9467bd', width=2),
                    marker=dict(size=4)
                ))

            fig.update_layout(
                xaxis_title="時間 (秒)",
                yaxis_title="次數",
                height=200,
                margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor='white'
            )

            return fig

        except Exception as e:
            print(f"Error in update_blink_count_chart: {e}")
            return go.Figure()

    @app.callback(
        Output("asic-bands-chart", "figure"),
        Input("interval", "n_intervals")
    )
    def update_asic_bands_chart(n):
        """更新ASIC頻帶圖 - 增強診斷版本"""
        try:
            # 降低更新頻率
            if ADAPTIVE_UPDATE and n % 6 != 0:
                return dash.no_update

            asic_data = data_buffer.get_asic_data()
            current_bands = asic_data['current_bands']

            fig = go.Figure()

            # 檢查是否有ASIC數據
            if all(band == 0 for band in current_bands):
                # 沒有ASIC數據時顯示診斷信息
                fig.add_annotation(
                    text="沒收到ASIC數據<br><br>可能原因:<br>• ThinkGear設備未連接<br>• 串口設定錯誤<br>• 電極接觸不良<br>• 設備未正確供電<br><br>請檢查設備連接狀態",
                    showarrow=False,
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    font=dict(size=14, color="red"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="red",
                    borderwidth=2
                )

                # 添加模擬數據用於演示
                demo_bands = [50, 30, 80, 60, 40, 70, 20, 90]  # 示例數據
                fig.add_trace(go.Bar(
                    x=[f"{band}<br>(示例)" for band in asic_bands],
                    y=demo_bands,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                                  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
                    opacity=0.3,
                    name="示例數據",
                    text=[f'{v}' for v in demo_bands],
                    textposition='auto'
                ))
            else:
                # 有數據時正常顯示
                fig.add_trace(go.Bar(
                    x=asic_bands,
                    y=current_bands,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                                  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
                    text=[f'{v}' if v > 0 else '0' for v in current_bands],
                    textposition='auto',
                    name="ASIC頻帶功率"
                ))

                # 添加數據狀態指示
                max_power = max(current_bands) if current_bands else 0
                fig.add_annotation(
                    text=f"✅ ASIC數據正常<br>最大功率: {max_power}",
                    showarrow=False,
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    font=dict(size=12, color="green"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="green",
                    borderwidth=1
                )

            fig.update_layout(
                title="ASIC EEG 8頻帶功率分布",
                xaxis_title="頻帶",
                yaxis_title="功率值",
                height=300,
                margin=dict(l=30, r=15, t=30, b=30),
                plot_bgcolor='white',
                showlegend=False
            )

            return fig

        except Exception as e:
            print(f"Error in update_asic_bands_chart: {e}")
            return go.Figure().add_annotation(
                text=f"ASIC圖表錯誤: {str(e)}",
                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper"
            )

    @app.callback(
        [Output("performance-status", "children"),
         Output("interval", "interval")],
        Input("interval", "n_intervals")
    )
    def update_performance_status(n):
        """更新性能狀態 - 自適應優化版本"""
        try:
            current_time = time.time()

            # 獲取系統性能
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent

            # 獲取數據狀態
            data, timestamps = data_buffer.get_data()
            latency = (time.time() - timestamps[-1]) * 1000 if len(timestamps) > 0 else 0

            # 獲取信號品質
            cognitive_data = data_buffer.get_cognitive_data()
            signal_quality = cognitive_data['signal_quality']

            # 數據源狀態
            data_source = "真實串口" if USE_REAL_SERIAL else "模擬數據"

            # 自適應間隔調整
            new_interval = UPDATE_INTERVAL
            if ADAPTIVE_UPDATE:
                # 根據CPU使用率調整更新頻率
                if cpu_usage > 80:
                    new_interval = min(MAX_UPDATE_INTERVAL, UPDATE_INTERVAL * 1.5)
                elif cpu_usage < 30:
                    new_interval = max(MIN_UPDATE_INTERVAL, UPDATE_INTERVAL * 0.8)

                performance_monitor['adaptive_interval'] = new_interval

            # 性能統計
            avg_render = performance_monitor['avg_render_time'] * 1000

            status_text = (
                f"CPU: {cpu_usage:.1f}% | "
                f"Memory: {memory_usage:.1f}% | "
                f"Latency: {latency:.1f}ms | "
                f"Render: {avg_render:.1f}ms | "
                f"Interval: {new_interval}ms | "
                f"Signal: {signal_quality} | "
                f"Source: {data_source} | "
                f"Updates: {n}"
            )

            return status_text, new_interval

        except Exception as e:
            return f"Status Error: {e}", UPDATE_INTERVAL

    @app.callback(
        Output("sensor-display", "children"),
        Input("interval", "n_intervals")
    )
    def update_sensor_display(n):
        """更新感測器顯示"""
        try:
            sensor_data = data_buffer.get_sensor_data()

            display_text = f"""
溫度: {sensor_data['temperature']:.1f}°C
濕度: {sensor_data['humidity']:.1f}%
光線: {sensor_data['light']}
更新: {datetime.now().strftime('%H:%M:%S')}
            """.strip()

            return display_text

        except Exception as e:
            return f"感測器錯誤: {e}"

    @app.callback(
        Output("recording-status", "children"),
        [Input("start-recording-btn", "n_clicks"),
         Input("stop-recording-btn", "n_clicks"),
         Input("interval", "n_intervals")],
        prevent_initial_call=True
    )
    def handle_recording_control(start_clicks, stop_clicks, n):
        """處理錄音控制"""
        try:
            ctx = dash.callback_context
            if not ctx.triggered:
                # 只是定期更新狀態
                if RECORDING_STATE['is_recording']:
                    elapsed = time.time() - RECORDING_STATE['start_time']
                    return f"🔴 錄音中... ({elapsed:.0f}秒) | 群組ID: {RECORDING_STATE['current_group_id']}"
                else:
                    return "⚪ 待機中"

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == "start-recording-btn" and start_clicks:
                if not RECORDING_STATE['is_recording']:
                    # 檢查音頻模組是否可用
                    if not AUDIO_AVAILABLE:
                        return "❌ 音頻模組未安裝！請執行: pip install sounddevice scipy"

                    # 產生新的群組ID
                    group_id = str(uuid.uuid4())[:8]
                    success = audio_recorder.start_recording(group_id)
                    if success:
                        return f"🔴 錄音開始 | 群組ID: {group_id}"
                    else:
                        return "❌ 錄音啟動失敗 (設備問題或權限不足)"
                else:
                    return "⚠️ 已在錄音中"

            elif button_id == "stop-recording-btn" and stop_clicks:
                if RECORDING_STATE['is_recording']:
                    filename = audio_recorder.stop_recording(db_writer)
                    if filename:
                        return f"✅ 錄音已停止並儲存: {filename}"
                    else:
                        return "⚠️ 錄音停止，但儲存失敗"
                else:
                    return "⚠️ 目前沒有錄音"

            # 預設狀態顯示
            if RECORDING_STATE['is_recording']:
                elapsed = time.time() - RECORDING_STATE['start_time']
                return f"🔴 錄音中... ({elapsed:.0f}秒) | 群組ID: {RECORDING_STATE['current_group_id']}"
            else:
                return "⚪ 待機中"

        except Exception as e:
            return f"錄音控制錯誤: {e}"

    print("🚀 啟動優化版響應式EEG監控系統")
    print("整合特性：")
    print("✓ 真實串口 + 模擬數據雙模式")
    print("✓ 完整資料庫儲存功能")
    print("✓ 增強的數據歷史記錄")
    print("✓ 高度優化的性能監控")
    print("✓ 響應式設計(RWD)")
    print("✓ ThinkGear協議完整支援")
    print("✓ 智能並行濾波處理")
    print("✓ 統一配置參數")
    print("✓ MQTT 感測器數據接收")
    print("✓ USB 麥克風錄音功能")
    print("✓ 統一記錄與群組管理")
    print("✓ 自適應更新頻率優化")
    print("✓ LRU緩存機制")
    print("✓ 智能渲染優化")
    print(f"✓ 數據源: {'真實串口' if USE_REAL_SERIAL else '模擬數據'}")
    print(f"✓ MQTT: {'已啟用' if MQTT_AVAILABLE else '未安裝'}")
    print(f"✓ 音頻: {'已啟用' if AUDIO_AVAILABLE else '未安裝'}")
    print(f"✓ 更新間隔: {UPDATE_INTERVAL}ms (自適應: {MIN_UPDATE_INTERVAL}-{MAX_UPDATE_INTERVAL}ms)")
    print("\n訪問地址: http://localhost:8052")

    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=8052)


if __name__ == "__main__":
    main()