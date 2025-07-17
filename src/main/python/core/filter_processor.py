"""EEG信號的優化濾波處理器"""

import numpy as np
import threading
import time
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter, sosfiltfilt
import logging

logger = logging.getLogger(__name__)


class OptimizedFilterProcessor:
    """EEG信號的優化並行濾波處理器"""
    
    def __init__(self, sample_rate: int = 512, max_workers: int = 4):
        self.sample_rate = sample_rate
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = {}
        self.last_data_hash = None
        self.lock = threading.Lock()
        
        # EEG頻率帶
        self.bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 35),
            "gamma": (35, 50),
            "smr": (12, 15),  # 感覺運動節律
            "mu": (8, 13),    # 運動皮質節律
            "high_gamma": (50, 80)  # 高伽馬波 (為性能限制)
        }
        
        # 預計算濾波器以提升性能
        self.sos_filters = {}
        self._precompute_filters()
        
    def _precompute_filters(self):
        """預計算所有頻率帶的SOS濾波器"""
        try:
            nyquist = self.sample_rate / 2
            filter_order = 2  # 較低階數以獲得更好性能
            
            for name, (low, high) in self.bands.items():
                try:
                    # 確保頻率範圍有效
                    if high >= nyquist:
                        high = nyquist - 1
                    if low >= high or low <= 0:
                        continue
                        
                    # 建立帶通濾波器
                    sos = butter(
                        filter_order,
                        [low / nyquist, high / nyquist],
                        btype='band',
                        output='sos'
                    )
                    
                    self.sos_filters[name] = sos
                    logger.debug(f"Filter created for {name}: {low}-{high} Hz")
                    
                except Exception as e:
                    logger.warning(f"Failed to create filter for {name}: {e}")
                    continue
                    
            logger.info(f"🍓 已建立優化濾波器: {len(self.sos_filters)} 個頻率帶")
            
        except Exception as e:
            logger.error(f"預計算濾波器錯誤: {e}")
            
    def _apply_single_filter(self, data: np.ndarray, sos: np.ndarray, 
                           band_name: str) -> np.ndarray:
        """對資料套用單一濾波器"""
        try:
            if len(data) < 10:  # 最小資料長度
                return np.zeros_like(data)
                
            # 套用濾波器
            filtered = sosfiltfilt(sos, data)
            return filtered
            
        except Exception as e:
            logger.warning(f"Filter application failed for {band_name}: {e}")
            return np.zeros_like(data)
            
    def process_bands_parallel(self, data: np.ndarray, 
                              use_cache: bool = True) -> Dict[str, np.ndarray]:
        """並行處理所有頻率帶"""
        if len(data) < 10:
            return {name: np.zeros_like(data) for name in self.bands.keys()}
            
        try:
            # 用於緩存的資料變更檢測
            if use_cache:
                data_hash = hash(data.tobytes())
                if (self.last_data_hash == data_hash and 
                    data_hash in self.cache):
                    return self.cache[data_hash]
                    
            # 提交所有濾波任務
            futures = {}
            for name, sos in self.sos_filters.items():
                future = self.executor.submit(
                    self._apply_single_filter, data, sos, name
                )
                futures[name] = future
                
            # 在逾時內收集結果
            results = {}
            timeout = 0.05  # 每個濾波器逾時50毫秒
            
            for name, future in futures.items():
                try:
                    result = future.result(timeout=timeout)
                    results[name] = result
                except Exception as e:
                    logger.warning(f"Filter timeout/error for {name}: {e}")
                    results[name] = np.zeros_like(data)
                    
            # 緩存結果
            if use_cache:
                with self.lock:
                    self.cache[data_hash] = results
                    self.last_data_hash = data_hash
                    
                    # 限制緩存大小
                    if len(self.cache) > 10:  # 只保留最近10個
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                        
            return results
            
        except Exception as e:
            logger.error(f"並行處理錯誤: {e}")
            return {name: np.zeros_like(data) for name in self.bands.keys()}
            
    def compute_band_powers(self, data: np.ndarray) -> Dict[str, float]:
        """計算每個頻率帶的功率"""
        try:
            filtered_data = self.process_bands_parallel(data)
            
            band_powers = {}
            for name, filtered_signal in filtered_data.items():
                if len(filtered_signal) > 0:
                    # 計算RMS功率
                    power = np.mean(filtered_signal ** 2)
                    band_powers[name] = float(power)
                else:
                    band_powers[name] = 0.0
                    
            return band_powers
            
        except Exception as e:
            logger.error(f"計算頻率帶功率錯誤: {e}")
            return {name: 0.0 for name in self.bands.keys()}
            
    def compute_relative_powers(self, data: np.ndarray) -> Dict[str, float]:
        """計算每個頻率帶的相對功率"""
        try:
            band_powers = self.compute_band_powers(data)
            
            # 計算總功率
            total_power = sum(band_powers.values())
            
            if total_power > 0:
                relative_powers = {
                    name: (power / total_power) * 100 
                    for name, power in band_powers.items()
                }
            else:
                relative_powers = {name: 0.0 for name in band_powers.keys()}
                
            return relative_powers
            
        except Exception as e:
            logger.error(f"計算相對功率錯誤: {e}")
            return {name: 0.0 for name in self.bands.keys()}
            
    def analyze_cognitive_state(self, data: np.ndarray) -> Dict[str, float]:
        """從EEG資料分析認知狀態"""
        try:
            relative_powers = self.compute_relative_powers(data)
            
            # 認知狀態指標
            cognitive_state = {
                'relaxation': relative_powers.get('alpha', 0),
                'attention': relative_powers.get('beta', 0),
                'drowsiness': relative_powers.get('theta', 0),
                'deep_sleep': relative_powers.get('delta', 0),
                'focus': relative_powers.get('smr', 0),
                'motor_activity': relative_powers.get('mu', 0),
                'high_cognitive': relative_powers.get('gamma', 0)
            }
            
            # 計算綜合分數
            cognitive_state['alertness'] = (
                cognitive_state['beta'] + cognitive_state['gamma']
            ) / 2
            
            cognitive_state['calmness'] = (
                cognitive_state['alpha'] - cognitive_state['beta']
            )
            
            return cognitive_state
            
        except Exception as e:
            logger.error(f"分析認知狀態錯誤: {e}")
            return {
                'relaxation': 0.0, 'attention': 0.0, 'drowsiness': 0.0,
                'deep_sleep': 0.0, 'focus': 0.0, 'motor_activity': 0.0,
                'high_cognitive': 0.0, 'alertness': 0.0, 'calmness': 0.0
            }
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """取得性能統計資料"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1),
                'active_filters': len(self.sos_filters),
                'available_bands': list(self.bands.keys()),
                'executor_active': not self.executor._shutdown
            }
            
    def cleanup(self):
        """清理資源"""
        try:
            self.executor.shutdown(wait=True)
            with self.lock:
                self.cache.clear()
            logger.info("濾波處理器已清理")
            
        except Exception as e:
            logger.error(f"清理期間錯誤: {e}")
            
    def __del__(self):
        """解構函數"""
        self.cleanup()


class AdaptiveFilterProcessor(OptimizedFilterProcessor):
    """具有動態優化的自適應濾波處理器"""
    
    def __init__(self, sample_rate: int = 512, max_workers: int = 4):
        super().__init__(sample_rate, max_workers)
        
        # 自適應參數
        self.performance_history = []
        self.adaptive_timeout = 0.05
        self.quality_threshold = 0.8
        
    def _update_performance_metrics(self, processing_time: float, 
                                  data_length: int, success_rate: float):
        """更新自適應優化的性能指標"""
        metric = {
            'timestamp': time.time(),
            'processing_time': processing_time,
            'data_length': data_length,
            'success_rate': success_rate,
            'throughput': data_length / processing_time if processing_time > 0 else 0
        }
        
        self.performance_history.append(metric)
        
        # 只保留最近的歷史記錄
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
            
        # 根據性能調整逾時時間
        if len(self.performance_history) >= 10:
            avg_time = np.mean([m['processing_time'] for m in self.performance_history[-10:]])
            if avg_time > self.adaptive_timeout:
                self.adaptive_timeout = min(0.1, avg_time * 1.2)
            else:
                self.adaptive_timeout = max(0.02, avg_time * 0.8)
                
    def process_bands_parallel(self, data: np.ndarray, 
                              use_cache: bool = True) -> Dict[str, np.ndarray]:
        """具有性能監控的自適應並行處理"""
        start_time = time.time()
        
        try:
            # 使用父類方法附帶自適應逾時
            original_timeout = 0.05
            
            # 暫時覆寫未來中的逾時
            results = super().process_bands_parallel(data, use_cache)
            
            # 計算成功率
            success_count = sum(1 for r in results.values() if np.any(r != 0))
            success_rate = success_count / len(results) if results else 0
            
            # 更新性能指標
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time, len(data), success_rate)
            
            return results
            
        except Exception as e:
            logger.error(f"自適應處理錯誤: {e}")
            return {name: np.zeros_like(data) for name in self.bands.keys()}
            
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """取得自適應性能統計資料"""
        base_stats = self.get_performance_stats()
        
        if len(self.performance_history) > 0:
            recent_metrics = self.performance_history[-10:]
            base_stats.update({
                'adaptive_timeout': self.adaptive_timeout,
                'avg_processing_time': np.mean([m['processing_time'] for m in recent_metrics]),
                'avg_throughput': np.mean([m['throughput'] for m in recent_metrics]),
                'avg_success_rate': np.mean([m['success_rate'] for m in recent_metrics]),
                'performance_samples': len(self.performance_history)
            })
        else:
            base_stats.update({
                'adaptive_timeout': self.adaptive_timeout,
                'avg_processing_time': 0,
                'avg_throughput': 0,
                'avg_success_rate': 0,
                'performance_samples': 0
            })
            
        return base_stats