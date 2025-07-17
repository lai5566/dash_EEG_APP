#!/usr/bin/env python3
"""
重新排序的單頁EEG監控系統
基於optimized_main.py的高效能架構，按照指定順序排列介面元素
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

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import butter, sosfiltfilt
import psutil

# ----- 全局設定 -----
SERIAL_PORT = "/dev/tty.usbserial-120"
BAUD_RATE = 57600
WINDOW_SIZE = 512
UPDATE_INTERVAL = 300  # 300ms 更新頻率
FS = 256
BATCH_SIZE = 100

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
    """增強的環形緩衝區 - 支援所有數據類型"""

    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)
        self.timestamps = np.zeros(size, dtype=np.float64)
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()

        # 認知指標數據
        self.current_attention = 0
        self.current_meditation = 0
        self.current_signal_quality = 200  # Start with poor signal
        self.attention_history = deque(maxlen=50)
        self.meditation_history = deque(maxlen=50)
        self.signal_quality_history = deque(maxlen=50)

        # ASIC頻帶數據
        self.current_asic_bands = [0] * 8
        self.asic_bands_history = deque(maxlen=30)

        # 眨眼數據
        self.blink_events = deque(maxlen=20)
        self.blink_count = 0
        self.blink_count_history = deque(maxlen=50)

    def append(self, value: float, timestamp: float):
        """添加原始數據"""
        with self.lock:
            self.data[self.head] = value
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.size
            if self.count < self.size:
                self.count += 1

    def add_cognitive_data(self, attention: int = None, meditation: int = None, signal_quality: int = None):
        """添加認知指標數據"""
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
                self.signal_quality_history.append((timestamp, signal_quality))

    def add_asic_bands(self, bands_data: List[int]):
        """添加ASIC頻帶數據"""
        with self.lock:
            timestamp = time.time()
            self.current_asic_bands = bands_data.copy()
            self.asic_bands_history.append((timestamp, bands_data.copy()))

    def add_blink_event(self, intensity: int):
        """添加眨眼事件"""
        with self.lock:
            timestamp = time.time()
            self.blink_events.append((timestamp, intensity))
            self.blink_count += 1
            self.blink_count_history.append((timestamp, self.blink_count))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """獲取原始數據"""
        with self.lock:
            if self.count == 0:
                return np.array([]), np.array([])

            if self.count < self.size:
                return self.data[:self.count].copy(), self.timestamps[:self.count].copy()
            else:
                indices = np.arange(self.head, self.head + self.size) % self.size
                return self.data[indices].copy(), self.timestamps[indices].copy()

    def get_cognitive_data(self) -> Dict:
        """獲取認知指標數據"""
        with self.lock:
            return {
                'attention': self.current_attention,
                'meditation': self.current_meditation,
                'signal_quality': self.current_signal_quality,
                'attention_history': list(self.attention_history),
                'meditation_history': list(self.meditation_history),
                'signal_quality_history': list(self.signal_quality_history)
            }

    def get_asic_data(self) -> Dict:
        """獲取ASIC頻帶數據"""
        with self.lock:
            return {
                'current_bands': self.current_asic_bands.copy(),
                'bands_history': list(self.asic_bands_history)
            }

    def get_blink_data(self) -> Dict:
        """獲取眨眼數據"""
        with self.lock:
            return {
                'events': list(self.blink_events),
                'count': self.blink_count,
                'count_history': list(self.blink_count_history)
            }


class EnhancedDatabaseWriter:
    """增強的資料庫寫入器"""

    def __init__(self, db_path: str = "ordered_eeg.db"):
        self.db_path = db_path
        self.raw_buffer = []
        self.cognitive_buffer = []
        self.asic_buffer = []
        self.blink_buffer = []
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

    def writer_thread(self):
        """資料庫寫入執行緒"""
        conn = self.setup_database()
        cur = conn.cursor()

        while self.running:
            try:
                time.sleep(1.0)  # 每秒寫入一次

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
    """增強的串口讀取器"""

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


class OptimizedFilterProcessor:
    """優化的濾波處理器"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=len(bands))

    def _apply_filter(self, data: np.ndarray, sos: np.ndarray) -> np.ndarray:
        """應用單一濾波器"""
        try:
            return sosfiltfilt(sos, data)
        except Exception:
            return np.zeros_like(data)

    def process_bands_parallel(self, data: np.ndarray) -> dict:
        """並行處理所有頻帶"""
        futures = {}

        for name, sos in sos_filters.items():
            future = self.executor.submit(self._apply_filter, data, sos)
            futures[name] = future

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=0.1)
            except:
                results[name] = np.zeros_like(data)

        return results

    def compute_fft_optimized(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """優化的FFT計算"""
        fft_vals = np.fft.rfft(data)
        freq = np.fft.rfftfreq(len(data), 1 / FS)
        return fft_vals, freq


def enhanced_serial_worker(out_queue: multiprocessing.Queue):
    """增強的串口工作程序"""
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


def main():
    # 啟動增強的串口讀取器
    serial_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=enhanced_serial_worker,
        args=(serial_queue,),
        daemon=True
    )
    p.start()

    # 初始化組件
    data_buffer = EnhancedCircularBuffer(WINDOW_SIZE)
    filter_processor = OptimizedFilterProcessor()
    db_writer = EnhancedDatabaseWriter()

    # 啟動資料庫寫入器
    db_writer.start()

    # 串口數據監聽器
    def enhanced_serial_listener():
        while True:
            try:
                parsed_data = serial_queue.get(timeout=1.0)
                timestamp = parsed_data.get('timestamp', time.time())

                # 處理原始數據
                if 'raw_value' in parsed_data:
                    voltage = parsed_data['raw_value']
                    data_buffer.append(voltage, timestamp)
                    db_writer.add_raw_data(timestamp, voltage)

                # 處理認知數據
                attention = parsed_data.get('attention')
                meditation = parsed_data.get('meditation')
                signal_quality = parsed_data.get('signal_quality')

                if any(x is not None for x in [attention, meditation, signal_quality]):
                    data_buffer.add_cognitive_data(attention, meditation, signal_quality)
                    if all(x is not None for x in [attention, meditation, signal_quality]):
                        db_writer.add_cognitive_data(timestamp, attention, meditation, signal_quality)

                # 處理ASIC頻帶
                if 'asic_bands' in parsed_data:
                    bands_data = parsed_data['asic_bands']
                    data_buffer.add_asic_bands(bands_data)
                    db_writer.add_asic_data(timestamp, bands_data)

                # 處理眨眼事件
                if 'blink' in parsed_data:
                    intensity = parsed_data['blink']
                    data_buffer.add_blink_event(intensity)
                    db_writer.add_blink_data(timestamp, intensity)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[enhanced_serial_listener] Error: {e}")

    threading.Thread(target=enhanced_serial_listener, daemon=True).start()

    # Dash應用程式
    app = dash.Dash(__name__)
    app.layout = html.Div([
        # 標題
        html.H1("重新排序的EEG監控儀表板",
                style={'textAlign': 'center', 'marginBottom': '20px'}),

        # 1. 主圖：FFT分頻顯示
        html.Div([
            html.H2("FFT頻帶分析", style={'textAlign': 'center'}),
            dcc.Graph(id="fft-bands-main", style={'height': '500px'}),
        ], style={'marginBottom': '30px'}),

        # 2. 認知指標：左邊趨勢圖 + 右邊圓形儀表
        html.Div([
            html.H2("認知指標", style={'textAlign': 'center'}),
            html.Div([
                # 左邊：趨勢圖 (60%)
                html.Div([
                    html.H3("認知指標趨勢"),
                    dcc.Graph(id="cognitive-trends", style={'height': '350px'}),
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # 右邊：圓形儀表 (40%)
                html.Div([
                    html.H3("即時數值"),
                    html.Div([
                        dcc.Graph(id="attention-gauge", style={'height': '175px'}),
                        dcc.Graph(id="meditation-gauge", style={'height': '175px'}),
                    ]),
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'}),
            ]),
        ], style={'marginBottom': '30px'}),

        # 3. 眨眼檢測：左邊事件時間軸 + 右邊眨眼計數圖
        html.Div([
            html.H2("眨眼檢測", style={'textAlign': 'center'}),
            html.Div([
                # 左邊：事件時間軸 (60%)
                html.Div([
                    html.H3("眨眼事件時間軸"),
                    dcc.Graph(id="blink-timeline", style={'height': '300px'}),
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # 右邊：眨眼計數圖 (40%)
                html.Div([
                    html.H3("眨眼計數統計"),
                    dcc.Graph(id="blink-count-chart", style={'height': '300px'}),
                ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'}),
            ]),
        ], style={'marginBottom': '30px'}),

        # 4. ASIC頻帶視覺化
        html.Div([
            html.H2("ASIC頻帶分析", style={'textAlign': 'center'}),
            dcc.Graph(id="asic-bands-chart", style={'height': '400px'}),
        ], style={'marginBottom': '30px'}),

        # 5. 機器性能 + 信號品質 (小字體10px)
        html.Div([
            html.Div(id="performance-status",
                     style={'fontSize': '10px', 'textAlign': 'center', 'color': '#666'}),
        ], style={'marginTop': '20px', 'borderTop': '1px solid #ddd', 'paddingTop': '10px'}),

        dcc.Interval(id="interval", interval=UPDATE_INTERVAL, n_intervals=0),
    ])

    @app.callback(
        Output("fft-bands-main", "figure"),
        Input("interval", "n_intervals")
    )
    def update_fft_bands_main(n):
        """更新主要的FFT頻帶圖 - 增強版"""
        data, timestamps = data_buffer.get_data()
        if len(data) == 0:
            return go.Figure().add_annotation(
                text="等待EEG數據...",
                showarrow=False,
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                font=dict(size=16, color="gray")
            )

        # 使用並行濾波處理
        filtered_data = filter_processor.process_bands_parallel(data)

        # 計算頻帶功率
        band_powers = {}
        for name, signal in filtered_data.items():
            if len(signal) > 0:
                # 計算RMS功率
                power = np.sqrt(np.mean(signal ** 2))
                band_powers[name] = power
            else:
                band_powers[name] = 0

        # 創建增強的子圖布局：左邊時域信號，右邊功率條形圖
        fig = make_subplots(
            rows=len(bands), cols=2,
            shared_xaxes=True,
            subplot_titles=list(bands.keys()) + ["頻帶功率分布"],
            vertical_spacing=0.08,
            horizontal_spacing=0.15,
            column_widths=[0.7, 0.3],  # 左邊70%，右邊30%
            specs=[[{"secondary_y": False}, {"rowspan": len(bands)}]] +
                  [[{"secondary_y": False}, None] for _ in range(len(bands) - 1)]
        )

        # 準備時間軸
        t = np.arange(len(data)) / FS

        # 定義更好的顏色方案
        band_colors = {
            "Delta (0.5-4Hz)": "#FF6B6B",  # 紅色 - 深度睡眠
            "Theta (4-8Hz)": "#4ECDC4",  # 青色 - 創意思考
            "Alpha (8-12Hz)": "#45B7D1",  # 藍色 - 放鬆專注
            "Beta (12-35Hz)": "#96CEB4",  # 綠色 - 活躍思考
            "Gamma (35-50Hz)": "#FFEAA7",  # 黃色 - 高度專注
            "SMR (12-15Hz)": "#DDA0DD",  # 紫色 - 感覺運動
            "Mu (8-13Hz)": "#98D8C8",  # 薄荷綠 - 運動皮層
            "High-Gamma (50-80Hz)": "#F7DC6F"  # 金色 - 高伽馬
        }

        # 添加每個頻帶的時域信號
        for i, (name, y) in enumerate(filtered_data.items(), start=1):
            color = band_colors.get(name, f"hsl({i * 40}, 70%, 50%)")

            fig.add_trace(
                go.Scatter(
                    x=t, y=y,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=color, width=1.5),
                    name=name
                ),
                row=i, col=1
            )

            # 添加功率信息到標題
            power = band_powers.get(name, 0)
            fig.layout.annotations[i - 1].text = f"{name}<br>功率: {power:.4f}"

        # 添加功率條形圖（右側）
        band_names_short = [name.split(' ')[0] for name in bands.keys()]
        powers = [band_powers.get(name, 0) for name in bands.keys()]
        colors_list = [band_colors.get(name, "#888888") for name in bands.keys()]

        fig.add_trace(
            go.Bar(
                x=powers,
                y=band_names_short,
                orientation='h',
                marker_color=colors_list,
                showlegend=False,
                text=[f'{p:.4f}' for p in powers],
                textposition='auto',
                name="頻帶功率"
            ),
            row=1, col=2
        )

        # 更新布局
        fig.update_layout(
            height=120 * len(bands),
            showlegend=False,
            title_text="EEG頻帶分解分析 - Delta, Theta, Alpha, Beta, Gamma",
            title_x=0.5,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # 更新x軸標籤
        fig.update_xaxes(title_text="時間 (秒)", row=len(bands), col=1)
        fig.update_xaxes(title_text="功率", row=len(bands), col=2)

        # 更新y軸標籤
        for i in range(1, len(bands) + 1):
            fig.update_yaxes(title_text="μV", row=i, col=1)

        return fig

    @app.callback(
        [Output("attention-gauge", "figure"),
         Output("meditation-gauge", "figure")],
        Input("interval", "n_intervals")
    )
    def update_cognitive_gauges(n):
        """更新認知指標圓形儀表"""
        cognitive_data = data_buffer.get_cognitive_data()
        attention = cognitive_data['attention']
        meditation = cognitive_data['meditation']

        # 注意力儀表
        attention_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attention,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "注意力"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        # 冥想儀表
        meditation_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=meditation,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "冥想"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        return attention_fig, meditation_fig

    @app.callback(
        Output("cognitive-trends", "figure"),
        Input("interval", "n_intervals")
    )
    def update_cognitive_trends(n):
        """更新認知指標趨勢圖"""
        cognitive_data = data_buffer.get_cognitive_data()

        fig = go.Figure()

        # 注意力趨勢
        if cognitive_data['attention_history']:
            times, values = zip(*cognitive_data['attention_history'])
            base_time = times[0] if times else 0
            rel_times = [(t - base_time) for t in times]
            fig.add_trace(go.Scatter(
                x=rel_times, y=values,
                mode='lines+markers',
                name='注意力',
                line=dict(color='blue')
            ))

        # 冥想趨勢
        if cognitive_data['meditation_history']:
            times, values = zip(*cognitive_data['meditation_history'])
            base_time = times[0] if times else 0
            rel_times = [(t - base_time) for t in times]
            fig.add_trace(go.Scatter(
                x=rel_times, y=values,
                mode='lines+markers',
                name='冥想',
                line=dict(color='green')
            ))

        fig.update_layout(
            title="認知指標歷史趨勢",
            xaxis_title="時間 (秒)",
            yaxis_title="數值 (0-100)",
            yaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    @app.callback(
        Output("blink-timeline", "figure"),
        Input("interval", "n_intervals")
    )
    def update_blink_timeline(n):
        """更新眨眼事件時間軸"""
        blink_data = data_buffer.get_blink_data()
        events = blink_data['events']

        fig = go.Figure()

        if events:
            times, intensities = zip(*events)
            base_time = times[0] if times else 0
            rel_times = [(t - base_time) for t in times]

            fig.add_trace(go.Scatter(
                x=rel_times, y=intensities,
                mode='markers',
                marker=dict(
                    size=[max(8, i / 3) for i in intensities],
                    color=intensities,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="強度")
                ),
                name='眨眼事件'
            ))

        fig.update_layout(
            title="眨眼事件時間軸",
            xaxis_title="時間 (秒)",
            yaxis_title="眨眼強度"
        )

        return fig

    @app.callback(
        Output("blink-count-chart", "figure"),
        Input("interval", "n_intervals")
    )
    def update_blink_count_chart(n):
        """更新眨眼計數圖"""
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
                name='累計眨眼次數',
                line=dict(color='purple')
            ))

        fig.update_layout(
            title="眨眼累計統計",
            xaxis_title="時間 (秒)",
            yaxis_title="累計次數"
        )

        return fig

    @app.callback(
        Output("asic-bands-chart", "figure"),
        Input("interval", "n_intervals")
    )
    def update_asic_bands_chart(n):
        """更新ASIC頻帶圖"""
        asic_data = data_buffer.get_asic_data()
        current_bands = asic_data['current_bands']

        fig = go.Figure(go.Bar(
            x=asic_bands,
            y=current_bands,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                          '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        ))

        fig.update_layout(
            title="ASIC EEG 8頻帶功率分布",
            xaxis_title="頻帶",
            yaxis_title="功率值"
        )

        return fig

    @app.callback(
        Output("performance-status", "children"),
        Input("interval", "n_intervals")
    )
    def update_performance_status(n):
        """更新性能狀態 (小字體)"""
        # 獲取系統性能
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent

        # 獲取數據狀態
        data, timestamps = data_buffer.get_data()
        latency = (time.time() - timestamps[-1]) * 1000 if len(timestamps) > 0 else 0

        # 獲取信號品質
        cognitive_data = data_buffer.get_cognitive_data()
        signal_quality = cognitive_data['signal_quality']

        status_text = (f"CPU: {cpu_usage:.1f}% | "
                       f"Memory: {memory_usage:.1f}% | "
                       f"Latency: {latency:.1f}ms | "
                       f"Samples: {len(data)} | "
                       f"Signal Quality: {signal_quality} | "
                       f"Update: {n}")

        return status_text

    print("🚀 啟動重新排序的EEG監控儀表板")
    print("介面順序：")
    print("1. FFT頻帶分析 (主圖)")
    print("2. 認知指標 (趨勢圖 + 圓形儀表)")
    print("3. 眨眼檢測 (時間軸 + 計數)")
    print("4. ASIC頻帶視覺化")
    print("5. 系統性能 + 信號品質")
    print("\n訪問地址: http://localhost:8050")

    app.run(debug=False, use_reloader=False, host='0.0.0.0')


if __name__ == "__main__":
    main()