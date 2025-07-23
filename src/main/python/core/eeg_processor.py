"""核心EEG信號處理模組 - 整合Numba優化"""

import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
import logging
import sys
import os

# 添加配置文件路徑
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'config'))
from app_config import FFT_TEST_DATA_CONFIG

# 導入Numba優化函數
try:
    from .numba_optimized import (
        hanning_window_numba, power_spectrum_numba, band_power_extraction_numba,
        signal_quality_z_score_numba, spectral_features_numba, NUMBA_AVAILABLE
    )
    USE_NUMBA = True
    logger = logging.getLogger(__name__)
    logger.info("🚀 Numba optimizations loaded for EEG processing")
except ImportError as e:
    USE_NUMBA = False
    NUMBA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Numba optimizations not available: {e}")
    logger.info("📊 Falling back to standard NumPy implementations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    """核心EEG信號處理類別"""
    
    def __init__(self, sample_rate: int = 512, window_size: int = 1024):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # 頻率帶 (Hz)
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # 初始化濾波器
        self._init_filters()
        
    def _init_filters(self):
        """初始化數位濾波器"""
        # 巴特沃茲帶通濾波器 (1-50 Hz)
        self.sos_bandpass = signal.butter(4, [1, 50], 
                                        btype='band', 
                                        fs=self.sample_rate, 
                                        output='sos')
        
        # 50Hz/60Hz電力線干擾陷波濾波器
        self.sos_notch_50 = signal.butter(2, [49, 51], 
                                        btype='bandstop', 
                                        fs=self.sample_rate, 
                                        output='sos')
        
        self.sos_notch_60 = signal.butter(2, [59, 61], 
                                        btype='bandstop', 
                                        fs=self.sample_rate, 
                                        output='sos')
        
    def preprocess_signal(self, raw_data: np.ndarray) -> np.ndarray:
        """預處理原始EEG信號"""
        with self.lock:
            try:
                # 套用帶通濾波器
                filtered = signal.sosfilt(self.sos_bandpass, raw_data)
                
                # 套用陷波濾波器
                filtered = signal.sosfilt(self.sos_notch_50, filtered)
                filtered = signal.sosfilt(self.sos_notch_60, filtered)
                
                # 正規化
                filtered = (filtered - np.mean(filtered)) / np.std(filtered)
                
                return filtered
                
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                return raw_data
    
    def compute_power_spectrum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用FFT計算功率譜 - Numba優化版本"""
        try:
            if USE_NUMBA and NUMBA_AVAILABLE:
                # 使用Numba優化的窗函數
                windowed = data * hanning_window_numba(len(data))
                
                # 計算FFT (仍使用SciPy，因為它已經高度優化)
                fft_data = fft(windowed)
                freqs = fftfreq(len(data), 1/self.sample_rate)
                
                # 使用Numba優化的功率譜計算
                psd = power_spectrum_numba(fft_data)
                
                # 只取正頻率
                positive_freqs = freqs[:len(freqs)//2]
                positive_psd = psd[:len(psd)//2]
                
                return positive_freqs, positive_psd
            else:
                # 回退到標準NumPy實現
                windowed = data * np.hanning(len(data))
                fft_data = fft(windowed)
                freqs = fftfreq(len(data), 1/self.sample_rate)
                psd = np.abs(fft_data) ** 2
                
                positive_freqs = freqs[:len(freqs)//2]
                positive_psd = psd[:len(psd)//2]
                
                return positive_freqs, positive_psd
            
        except Exception as e:
            logger.error(f"Error computing power spectrum: {e}")
            return np.array([]), np.array([])
    
    def extract_band_powers(self, data: np.ndarray) -> Dict[str, float]:
        """提取不同頻率帶的功率 - Numba優化版本"""
        try:
            freqs, psd = self.compute_power_spectrum(data)
            
            if len(freqs) == 0:
                return {band: 0.0 for band in self.frequency_bands.keys()}
            
            band_powers = {}
            
            if USE_NUMBA and NUMBA_AVAILABLE:
                # 使用Numba優化的頻帶功率提取
                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    band_power = band_power_extraction_numba(psd, freqs, low_freq, high_freq)
                    band_powers[band_name] = float(band_power)
            else:
                # 回退到標準NumPy實現
                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    band_indices = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                    
                    if len(band_indices) > 0:
                        band_power = np.mean(psd[band_indices])
                        band_powers[band_name] = float(band_power)
                    else:
                        band_powers[band_name] = 0.0
            
            return band_powers
            
        except Exception as e:
            logger.error(f"Error extracting band powers: {e}")
            return {band: 0.0 for band in self.frequency_bands.keys()}
    
    def extract_fft_bands(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """提取FFT頻帶的時域信號 (用於折線圖顯示)"""
        try:
            # 計算FFT
            fft_vals = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1/self.sample_rate)
            
            if len(freqs) == 0:
                return {band: np.array([]) for band in self.frequency_bands.keys()}
            
            band_signals = {}
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # 創建頻率掩模
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                # 應用掩模到FFT結果
                filtered_fft = fft_vals.copy()
                filtered_fft[~mask] = 0
                
                # 逆FFT得到時域信號
                band_signal = np.fft.irfft(filtered_fft, n=len(data))
                band_signals[band_name] = band_signal
            
            return band_signals
            
        except Exception as e:
            logger.error(f"Error extracting FFT bands: {e}")
            return {band: np.array([]) for band in self.frequency_bands.keys()}
    
    def detect_artifacts(self, data: np.ndarray, threshold: float = 3.0) -> List[int]:
        """使用統計閾值檢測偽訊"""
        try:
            # 計算z分數
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            
            # 找出偽訊
            artifact_indices = np.where(z_scores > threshold)[0]
            
            return artifact_indices.tolist()
            
        except Exception as e:
            logger.error(f"Error detecting artifacts: {e}")
            return []
    
    def calculate_spectral_features(self, data: np.ndarray) -> Dict[str, float]:
        """計算頻譜特徵 - Numba優化版本"""
        try:
            freqs, psd = self.compute_power_spectrum(data)
            
            if len(freqs) == 0:
                return {'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0}
            
            if USE_NUMBA and NUMBA_AVAILABLE:
                # 使用Numba優化的頻譜特徵計算
                spectral_centroid, spectral_bandwidth = spectral_features_numba(freqs, psd)
                
                return {
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_bandwidth': float(spectral_bandwidth)
                }
            else:
                # 回退到標準NumPy實現
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
                
                return {
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_bandwidth': float(spectral_bandwidth)
                }
            
        except Exception as e:
            logger.error(f"Error calculating spectral features: {e}")
            return {'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0}
    
    def process_eeg_window(self, raw_data: np.ndarray) -> Dict:
        """處理EEG資料視窗"""
        try:
            # 預處理
            processed_data = self.preprocess_signal(raw_data)
            
            # 提取特徵
            band_powers = self.extract_band_powers(processed_data)
            fft_bands = self.extract_fft_bands(processed_data)
            spectral_features = self.calculate_spectral_features(processed_data)
            artifacts = self.detect_artifacts(processed_data)
            
            # 計算信號品質指標
            signal_quality = self._calculate_signal_quality(processed_data)
            
            return {
                'processed_data': processed_data,
                'band_powers': band_powers,
                'fft_bands': fft_bands,
                'spectral_features': spectral_features,
                'artifacts': artifacts,
                'signal_quality': signal_quality,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing EEG window: {e}")
            return {}
    
    def _calculate_signal_quality(self, data: np.ndarray) -> float:
        """計算信號品質分數 (0-100) - Numba優化版本"""
        try:
            if USE_NUMBA and NUMBA_AVAILABLE:
                # 使用Numba優化的信號品質計算
                quality = signal_quality_z_score_numba(data)
                return float(quality)
            else:
                # 回退到標準NumPy實現
                signal_power = np.var(data)
                noise_estimate = np.var(np.diff(data))  # 高頻雜訊
                
                if noise_estimate > 0:
                    snr = 10 * np.log10(signal_power / noise_estimate)
                    # 將信噪比映射到0-100範圍
                    quality = min(100, max(0, (snr + 10) * 5))
                else:
                    quality = 100
                
                return float(quality)
            
        except Exception as e:
            logger.error(f"Error calculating signal quality: {e}")
            return 0.0


class RealTimeEEGProcessor:
    """具有循環緩衝區的即時EEG處理器"""
    
    def __init__(self, sample_rate: int = 512, window_size: int = 1024):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.processor = EEGProcessor(sample_rate, window_size)
        self.buffer = np.zeros(window_size)
        self.buffer_index = 0
        self.is_buffer_full = False
        self.lock = threading.Lock()
        
    def add_sample(self, sample: float):
        """將新樣本添加到循環緩衝區"""
        with self.lock:
            self.buffer[self.buffer_index] = sample
            self.buffer_index = (self.buffer_index + 1) % len(self.buffer)
            
            if self.buffer_index == 0:
                self.is_buffer_full = True
    
    def get_current_window(self) -> np.ndarray:
        """取得目前視窗資料"""
        with self.lock:
            # 總是確保有足夠的數據進行處理
            if not self.is_buffer_full and self.buffer_index < 512:
                # 生成測試用的假資料 (當沒有足夠真實資料時)
                # 確保至少有足夠的樣本數進行處理
                min_samples = max(self.window_size, 512)  # 至少512個樣本
                return self._generate_test_data(min_samples)
            
            if not self.is_buffer_full:
                # 如果緩衝區未滿但有一些數據，仍然返回足夠大小的測試數據
                if self.buffer_index < 512:
                    return self._generate_test_data(max(self.buffer_index, 512))
                return self.buffer[:self.buffer_index]
            
            # 返回正確排序的循環緩衝區
            return np.concatenate([
                self.buffer[self.buffer_index:],
                self.buffer[:self.buffer_index]
            ])
    
    def _generate_test_data(self, length: int) -> np.ndarray:
        """生成指定長度的測試數據"""
        duration = length / self.sample_rate
        t = np.linspace(0, duration, length)
        
        # 使用配置文件中的振幅和頻率設定
        amps = FFT_TEST_DATA_CONFIG['amplitudes']
        freqs = FFT_TEST_DATA_CONFIG['frequencies']
        
        test_data = (
            amps['delta'] * np.sin(2 * np.pi * freqs['delta'] * t) +     # Delta波
            amps['theta'] * np.sin(2 * np.pi * freqs['theta'] * t) +     # Theta波  
            amps['alpha'] * np.sin(2 * np.pi * freqs['alpha'] * t) +     # Alpha波
            amps['beta'] * np.sin(2 * np.pi * freqs['beta'] * t) +       # Beta波
            amps['gamma'] * np.sin(2 * np.pi * freqs['gamma'] * t) +     # Gamma波
            amps['noise'] * np.random.randn(length)                      # 雜訊
        )
        return test_data
    
    def process_current_window(self) -> Dict:
        """處理目前視窗"""
        current_data = self.get_current_window()
        
        # 確保我們有足夠的數據進行處理
        if len(current_data) < 50:  # 降低最小樣本要求
            logger.warning(f"Insufficient data for processing: {len(current_data)} samples")
            # 生成足夠的測試數據
            current_data = self._generate_test_data(512)
        
        # 進行完整的EEG處理
        try:
            result = self.processor.process_eeg_window(current_data)
            if result and 'fft_bands' in result:
                # 確保 FFT 頻段數據不為空
                for band_name in result['fft_bands']:
                    if len(result['fft_bands'][band_name]) == 0:
                        result['fft_bands'][band_name] = np.zeros(len(current_data))
            return result
        except Exception as e:
            logger.error(f"Error in process_eeg_window: {e}")
            # 返回默認結構確保圖表有數據顯示
            return {
                'processed_data': current_data,
                'band_powers': {band: 0.0 for band in self.processor.frequency_bands.keys()},
                'fft_bands': {band: np.zeros(len(current_data)) for band in self.processor.frequency_bands.keys()},
                'spectral_features': {'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0},
                'artifacts': [],
                'signal_quality': 50.0,
                'timestamp': time.time()
            }