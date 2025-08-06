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
from app_config import FFT_TEST_DATA_CONFIG, USE_MOCK_DATA, FFT_CALCULATION_CONFIG, PREPROCESSING_CONFIG

# 導入Numba優化函數
try:
    from .numba_optimized import (
        hanning_window_numba, power_spectrum_numba, band_power_extraction_numba,
        signal_quality_z_score_numba, spectral_features_numba, NUMBA_AVAILABLE
    )
    USE_NUMBA = True
    logger = logging.getLogger(__name__)
    logger.info("Numba optimizations loaded for EEG processing")
except ImportError as e:
    USE_NUMBA = False
    NUMBA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"WARNING: Numba optimizations not available: {e}")
    logger.info("Falling back to standard NumPy implementations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    """核心EEG信號處理類別"""
    
    def __init__(self, sample_rate: int = 512, window_size: int = 1024):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.lock = threading.Lock()
        
        # 根據配置選擇計算方法和頻率帶
        self.calculation_mode = FFT_CALCULATION_CONFIG['mode']
        method_config = FFT_CALCULATION_CONFIG[f'{self.calculation_mode}_method']
        self.frequency_bands = method_config['frequency_bands']
        self.data_scaling = method_config['data_scaling']
        
        # 載入預處理配置
        self.preprocessing_mode = PREPROCESSING_CONFIG['mode']
        self.preprocessing_config = PREPROCESSING_CONFIG[self.preprocessing_mode]
        
        logger.info(f"EEG Processor initialized with {self.calculation_mode} mode")
        logger.info(f"Preprocessing mode: {self.preprocessing_mode}")
        logger.info(f"Frequency bands: {self.frequency_bands}")
        
        # 根據配置初始化濾波器 (延遲初始化)
        self.filters_initialized = False
        self._filters = {}
        
        # 如果需要任何濾波器，立即初始化
        if (self.preprocessing_config.get('dc_removal', False) or 
            self.preprocessing_config.get('powerline_notch', False) or 
            self.preprocessing_config.get('bandpass_filter', False)):
            self._init_filters()
        
    def _init_filters(self):
        """根據配置初始化所需的數位濾波器"""
        if self.filters_initialized:
            return
            
        filter_params = PREPROCESSING_CONFIG['filter_params']
        butter_order = filter_params['butter_order']
        notch_order = filter_params['notch_order']
        
        # DC移除 (高通濾波器)
        if self.preprocessing_config.get('dc_removal', False):
            cutoff = self.preprocessing_config.get('highpass_cutoff', 0.5)
            self._filters['highpass'] = signal.butter(
                butter_order, cutoff, 
                btype='high', 
                fs=self.sample_rate, 
                output='sos'
            )
            logger.info(f"Initialized highpass filter: {cutoff} Hz")
        
        # 帶通濾波器
        if self.preprocessing_config.get('bandpass_filter', False):
            low = self.preprocessing_config.get('bandpass_low', 0.5)
            high = self.preprocessing_config.get('bandpass_high', 50.0)
            self._filters['bandpass'] = signal.butter(
                butter_order, [low, high], 
                btype='band', 
                fs=self.sample_rate, 
                output='sos'
            )
            logger.info(f"Initialized bandpass filter: {low}-{high} Hz")
        
        # 電力線陷波濾波器 
        if self.preprocessing_config.get('powerline_notch', False):
            # 50Hz陷波
            self._filters['notch_50'] = signal.butter(
                notch_order, [49, 51], 
                btype='bandstop', 
                fs=self.sample_rate, 
                output='sos'
            )
            # 60Hz陷波
            self._filters['notch_60'] = signal.butter(
                notch_order, [59, 61], 
                btype='bandstop', 
                fs=self.sample_rate, 
                output='sos'
            )
            logger.info("Initialized powerline notch filters: 50Hz, 60Hz")
        
        self.filters_initialized = True
        logger.info(f"Filter initialization complete. Active filters: {list(self._filters.keys())}")
        
    def preprocess_signal(self, raw_data: np.ndarray) -> np.ndarray:
        """根據配置進行信號預處理，支持無預處理模式"""
        with self.lock:
            try:
                # 如果是無預處理模式，直接返回原始數據
                if self.preprocessing_mode == 'none':
                    logger.debug("No preprocessing applied - using raw signal (none mode)")
                    return raw_data.copy()
                
                processed_data = raw_data.copy()
                applied_filters = []
                
                # 1. DC移除 (高通濾波) - 推薦保留以移除DC偏移
                if 'highpass' in self._filters:
                    processed_data = signal.sosfilt(self._filters['highpass'], processed_data)
                    applied_filters.append('highpass')
                
                # 2. 帶通濾波 - 可選，會改變信號特性
                if 'bandpass' in self._filters:
                    processed_data = signal.sosfilt(self._filters['bandpass'], processed_data)
                    applied_filters.append('bandpass')
                
                # 3. 電力線干擾陷波濾波 - 可選，在環境乾淨時可關閉
                if 'notch_50' in self._filters:
                    processed_data = signal.sosfilt(self._filters['notch_50'], processed_data)
                    applied_filters.append('notch_50')
                    
                if 'notch_60' in self._filters:
                    processed_data = signal.sosfilt(self._filters['notch_60'], processed_data)
                    applied_filters.append('notch_60')
                
                # 4. 標準化 - 只在完整模式下啟用，會失去絕對電壓信息
                if self.preprocessing_config.get('normalization', False):
                    if len(processed_data) > 1:  # 避免除零錯誤
                        std_val = np.std(processed_data)
                        if std_val > 0:  # 確保標準差不為零
                            processed_data = (processed_data - np.mean(processed_data)) / std_val
                            applied_filters.append('normalization')
                        else:
                            logger.warning("Signal standard deviation is zero, skipping normalization")
                
                # 記錄應用的預處理步驟
                if applied_filters:
                    logger.debug(f"Applied preprocessing filters: {applied_filters}")
                else:
                    logger.debug("No preprocessing applied - using raw signal")
                
                return processed_data
                
            except Exception as e:
                logger.error(f"Error in preprocessing: {e}")
                logger.info("Returning raw signal due to preprocessing error")
                return raw_data
    
    def compute_power_spectrum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """使用FFT計算功率譜 - 帶窗函數能量補償的Numba優化版本"""
        try:
            # 獲取窗函數設定
            windowing_config = PREPROCESSING_CONFIG['windowing']
            window_type = windowing_config.get('type', 'hanning')
            compensation_enabled = self.preprocessing_config.get('window_compensation', True)
            
            if USE_NUMBA and NUMBA_AVAILABLE:
                # 選擇窗函數
                if window_type == 'hanning':
                    windowed = data * hanning_window_numba(len(data))
                    compensation_factor = windowing_config.get('compensation_factor', 2.0) if compensation_enabled else 1.0
                else:
                    # 回退到numpy窗函數
                    if window_type == 'hamming':
                        window = np.hamming(len(data))
                        compensation_factor = 1.85 if compensation_enabled else 1.0
                    elif window_type == 'blackman':
                        window = np.blackman(len(data))
                        compensation_factor = 2.8 if compensation_enabled else 1.0
                    elif window_type == 'rectangular':
                        window = np.ones(len(data))
                        compensation_factor = 1.0  # 矩形窗不需要補償
                    else:
                        window = np.hanning(len(data))
                        compensation_factor = 2.0 if compensation_enabled else 1.0
                    
                    windowed = data * window
                
                # 計算FFT
                fft_data = fft(windowed)
                freqs = fftfreq(len(data), 1/self.sample_rate)
                
                # 使用Numba優化的功率譜計算並應用補償
                psd = power_spectrum_numba(fft_data) * compensation_factor
                
                # 只取正頻率
                positive_freqs = freqs[:len(freqs)//2]
                positive_psd = psd[:len(psd)//2]
                
                return positive_freqs, positive_psd
            else:
                # 回退到標準NumPy實現
                if window_type == 'hanning':
                    windowed = data * np.hanning(len(data))
                    compensation_factor = 2.0 if compensation_enabled else 1.0
                elif window_type == 'hamming':
                    windowed = data * np.hamming(len(data))
                    compensation_factor = 1.85 if compensation_enabled else 1.0
                elif window_type == 'blackman':
                    windowed = data * np.blackman(len(data))
                    compensation_factor = 2.8 if compensation_enabled else 1.0
                elif window_type == 'rectangular':
                    windowed = data  # 無窗函數
                    compensation_factor = 1.0
                else:
                    windowed = data * np.hanning(len(data))
                    compensation_factor = 2.0 if compensation_enabled else 1.0
                
                fft_data = fft(windowed)
                freqs = fftfreq(len(data), 1/self.sample_rate)
                psd = (np.abs(fft_data) ** 2) * compensation_factor
                
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
                # 如果沒有計算出頻譜數據，返回空字典
                return {}
            
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
            return {}
    
    def extract_fft_bands(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """提取FFT頻帶的時域信號 (用於折線圖顯示)"""
        try:
            # 計算FFT - 與main_old.py完全一致的方法
            fft_vals = np.fft.rfft(data)
            freqs = np.fft.rfftfreq(len(data), 1/self.sample_rate)
            
            if len(freqs) == 0:
                return {band: np.array([]) for band in self.frequency_bands.keys()}
            
            band_signals = {}
            
            for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                # 創建頻率掩模 - 與main_old.py一致
                mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                # 應用掩模到FFT結果 - 與main_old.py一致
                filtered_fft = fft_vals * mask
                
                # 逆FFT得到時域信號 - 與main_old.py一致
                band_signal = np.fft.irfft(filtered_fft, n=len(data))
                band_signals[band_name] = band_signal
            
            return band_signals
            
        except Exception as e:
            logger.error(f"Error extracting FFT bands: {e}")
            return {band: np.array([]) for band in self.frequency_bands.keys()}
    
    def compute_full_spectrum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """計算完整頻譜數據用於瀑布圖/頻譜圖顯示"""
        try:
            # 使用窗函數進行FFT計算
            if USE_NUMBA and NUMBA_AVAILABLE:
                windowed = data * hanning_window_numba(len(data))
                fft_data = fft(windowed)
                freqs = fftfreq(len(data), 1/self.sample_rate)
                psd = power_spectrum_numba(fft_data)
            else:
                windowed = data * np.hanning(len(data))
                fft_data = fft(windowed)
                freqs = fftfreq(len(data), 1/self.sample_rate)
                psd = np.abs(fft_data) ** 2
            
            # 只取正頻率部分
            positive_freqs = freqs[:len(freqs)//2]
            positive_psd = psd[:len(psd)//2]
            
            # 限制頻率範圍到 1-50 Hz (EEG的主要頻率範圍)
            freq_mask = (positive_freqs >= 1) & (positive_freqs <= 50)
            spectrum_freqs = positive_freqs[freq_mask]
            spectrum_powers = positive_psd[freq_mask]
            
            # 對功率譜進行對數變換以便更好地顯示
            spectrum_powers_db = 10 * np.log10(spectrum_powers + 1e-12)  # 避免log(0)
            
            return spectrum_freqs, spectrum_powers_db
            
        except Exception as e:
            logger.error(f"Error computing full spectrum: {e}")
            return np.array([]), np.array([])
    
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
            
            # 根據配置選擇計算方法
            if self.calculation_mode == 'power':
                # 功率模式：計算頻帶功率值
                band_powers = self.extract_band_powers(processed_data)
                # 為了保持向後兼容，仍然計算fft_bands但不使用
                fft_bands = self.extract_fft_bands(processed_data)
            elif self.calculation_mode == 'simple_fft_bands':
                # 簡化FFT頻帶模式：與main_old.py一致，直接顯示時域波形
                fft_bands = self.extract_fft_bands(processed_data)
                # 不計算band_powers，直接使用時域波形數據
                band_powers = {}
                for band_name, waveform in fft_bands.items():
                    if len(waveform) > 0:
                        # 使用波形的平均絕對值作為顯示參考，不做過度縮放
                        avg_amplitude = np.mean(np.abs(waveform))
                        band_powers[band_name] = float(avg_amplitude)
                    else:
                        band_powers[band_name] = 0.0
            else:  # waveform mode
                # 波形模式：計算頻帶濾波波形，並從中計算代表性數值
                fft_bands = self.extract_fft_bands(processed_data)
                # 將波形的RMS值作為功率指標，並應用縮放
                band_powers = {}
                for band_name, waveform in fft_bands.items():
                    if len(waveform) > 0:
                        # 計算RMS值並應用縮放
                        rms_value = np.sqrt(np.mean(waveform**2)) * self.data_scaling
                        band_powers[band_name] = float(rms_value)
                    else:
                        band_powers[band_name] = 0.0
            
            spectral_features = self.calculate_spectral_features(processed_data)
            artifacts = self.detect_artifacts(processed_data)
            
            # 計算完整頻譜用於瀑布圖顯示
            spectrum_freqs, spectrum_powers = self.compute_full_spectrum(processed_data)
            
            # 計算信號品質指標
            signal_quality = self._calculate_signal_quality(processed_data)
            
            return {
                'processed_data': processed_data,
                'band_powers': band_powers,
                'fft_bands': fft_bands,
                'spectrum_freqs': spectrum_freqs,
                'spectrum_powers': spectrum_powers,
                'spectral_features': spectral_features,
                'artifacts': artifacts,
                'signal_quality': signal_quality,
                'timestamp': time.time(),
                'calculation_mode': self.calculation_mode
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
        
        # 添加配置模式訪問 - 從內部處理器獲取
        self.calculation_mode = self.processor.calculation_mode
        self.frequency_bands = self.processor.frequency_bands
        self.data_scaling = self.processor.data_scaling
        
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
        """生成指定長度的測試數據（只有在啟用模擬數據時）"""
        if not USE_MOCK_DATA:
            # 如果禁用模擬數據，返回空數據
            return np.array([])
            
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
            # 只有在啟用模擬數據時才生成測試數據
            if USE_MOCK_DATA:
                current_data = self._generate_test_data(512)
            else:
                # 數據不足且未啟用模擬數據，返回空結果
                return {}
        
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
            # 返回空結構
            return {}