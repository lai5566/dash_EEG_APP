"""EEG資料循環緩衝區"""

import threading
import time
import numpy as np
from collections import deque
from typing import Tuple, List, Dict
import random


class EnhancedCircularBuffer:
    """EEG資料循環緩衝區，包含認知指標"""

    def __init__(self, size: int):
        self.size = size
        self.data = np.zeros(size, dtype=np.float32)
        self.timestamps = np.zeros(size, dtype=np.float64)
        self.head = 0
        self.count = 0
        self.lock = threading.Lock()

        # 認知指標資料
        self.current_attention = 0
        self.current_meditation = 0
        self.current_signal_quality = 200  # 從信號品質不佳開始
        self.attention_history = deque(maxlen=50)
        self.meditation_history = deque(maxlen=50)
        self.signal_quality_history = deque(maxlen=50)

        # ASIC頻帶資料
        self.current_asic_bands = [0] * 8
        self.asic_bands_history = deque(maxlen=30)

        # 眨眼資料
        self.blink_events = deque(maxlen=20)
        self.blink_count = 0
        self.blink_count_history = deque(maxlen=50)

        # FFT頻帶功率歷史資料 (用於流動效果)
        self.fft_band_history = {
            'delta': deque(maxlen=100),
            'theta': deque(maxlen=100), 
            'alpha': deque(maxlen=100),
            'beta': deque(maxlen=100),
            'gamma': deque(maxlen=100)
        }
        self.current_fft_bands = {
            'delta': 0.0,
            'theta': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.0
        }

        # 完整頻譜資料 (用於瀑布圖/頻譜圖顯示)
        self.spectral_history = deque(maxlen=60)  # 保存60個時間點的頻譜數據
        self.current_spectrum_freqs = np.array([])
        self.current_spectrum_powers = np.array([])

        # MQTT感測器資料
        self.current_temperature = 0.0
        self.current_humidity = 0.0
        self.current_light = 0
        self.sensor_history = deque(maxlen=50)

        # 初始化測試用的假資料
        self._init_fake_data()

    def _init_fake_data(self):
        """初始化測試用的假資料"""
        base_time = time.time()
        for i in range(15):
            t = base_time - (15 - i)
            self.attention_history.append((t, random.randint(30, 90)))
            self.meditation_history.append((t, random.randint(20, 80)))
            self.blink_count_history.append((t, i))

            if i % 3 == 0:
                self.blink_events.append((t, random.randint(50, 200)))
                
        # 初始化 FFT 頻帶測試數據 - 增加數據點密度以創造流動效果
        for j in range(100):
            t = base_time - (100 - j) * 0.05  # 每0.05秒一個數據點，增加數據密度
            # 使用正弦波來模擬真實的波動效果，而非隨機數據
            phase_offset = j * 0.1  # 為每個時間點添加相位偏移
            
            # 為每個頻帶創建特定的波動模式
            self.fft_band_history['delta'].append((t, 0.3 + 0.2 * np.sin(phase_offset * 0.5) + 0.05 * random.random()))
            self.fft_band_history['theta'].append((t, 0.5 + 0.3 * np.sin(phase_offset * 0.8) + 0.1 * random.random()))
            self.fft_band_history['alpha'].append((t, 0.7 + 0.5 * np.sin(phase_offset * 1.2) + 0.1 * random.random()))
            self.fft_band_history['beta'].append((t, 0.35 + 0.25 * np.sin(phase_offset * 1.5) + 0.05 * random.random()))
            self.fft_band_history['gamma'].append((t, 0.15 + 0.15 * np.sin(phase_offset * 2.0) + 0.03 * random.random()))

    def append(self, value: float, timestamp: float):
        """新增原始資料"""
        with self.lock:
            self.data[self.head] = value
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.size
            if self.count < self.size:
                self.count += 1

    def add_cognitive_data(self, attention: int = None, meditation: int = None, signal_quality: int = None):
        """新增認知指標資料"""
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
        """新增ASIC頻帶資料"""
        with self.lock:
            timestamp = time.time()
            self.current_asic_bands = bands_data.copy()
            self.asic_bands_history.append((timestamp, bands_data.copy()))
            print(f"[ASIC DEBUG] DataBuffer: Added ASIC bands: {bands_data}, History length: {len(self.asic_bands_history)}")

    def add_blink_event(self, intensity: int):
        """新增眨眼事件"""
        with self.lock:
            timestamp = time.time()
            self.blink_events.append((timestamp, intensity))
            self.blink_count += 1
            self.blink_count_history.append((timestamp, self.blink_count))

    def add_fft_band_powers(self, band_powers: Dict[str, float]):
        """新增FFT頻帶功率資料"""
        with self.lock:
            timestamp = time.time()
            for band_name, power in band_powers.items():
                if band_name in self.current_fft_bands:
                    self.current_fft_bands[band_name] = power
                    self.fft_band_history[band_name].append((timestamp, power))

    def add_spectral_data(self, freqs: np.ndarray, powers: np.ndarray):
        """新增完整頻譜資料用於瀑布圖顯示"""
        with self.lock:
            timestamp = time.time()
            
            # 更新當前頻譜資料
            self.current_spectrum_freqs = freqs.copy()
            self.current_spectrum_powers = powers.copy()
            
            # 添加到歷史資料 (timestamp, freqs, powers)
            self.spectral_history.append((timestamp, freqs.copy(), powers.copy()))

    def add_sensor_data(self, temperature: float, humidity: float, light: int):
        """新增感測器資料"""
        with self.lock:
            timestamp = time.time()
            self.current_temperature = temperature
            self.current_humidity = humidity
            self.current_light = light
            self.sensor_history.append((timestamp, temperature, humidity, light))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """取得原始資料"""
        with self.lock:
            if self.count == 0:
                # 產生測試用的假資料
                t = np.linspace(0, 2, self.size)
                fake_data = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(self.size)
                return fake_data, t

            if self.count < self.size:
                return self.data[:self.count].copy(), self.timestamps[:self.count].copy()
            else:
                indices = np.arange(self.head, self.head + self.size) % self.size
                return self.data[indices].copy(), self.timestamps[indices].copy()

    def get_cognitive_data(self) -> Dict:
        """取得認知指標資料"""
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
        """取得ASIC頻帶資料"""
        with self.lock:
            result = {
                'current_bands': self.current_asic_bands.copy(),
                'bands_history': list(self.asic_bands_history)
            }
            print(f"[ASIC DEBUG] DataBuffer: Retrieved ASIC data - Current: {self.current_asic_bands}, History: {len(self.asic_bands_history)} items")
            return result

    def get_blink_data(self) -> Dict:
        """取得眨眼資料"""
        with self.lock:
            return {
                'events': list(self.blink_events),
                'count': self.blink_count,
                'count_history': list(self.blink_count_history)
            }

    def get_fft_band_data(self) -> Dict:
        """取得FFT頻帶功率資料"""
        with self.lock:
            return {
                'current_bands': self.current_fft_bands.copy(),
                'band_history': {
                    band_name: list(history) 
                    for band_name, history in self.fft_band_history.items()
                }
            }

    def get_spectral_data(self) -> Dict:
        """取得完整頻譜資料用於瀑布圖顯示"""
        with self.lock:
            return {
                'current_freqs': self.current_spectrum_freqs.copy(),
                'current_powers': self.current_spectrum_powers.copy(),
                'spectral_history': list(self.spectral_history)
            }

    def get_sensor_data(self) -> Dict:
        """取得感測器資料"""
        with self.lock:
            return {
                'temperature': self.current_temperature,
                'humidity': self.current_humidity,
                'light': self.current_light,
                'history': list(self.sensor_history)
            }