"""Numba性能基準測試模組

此模組用於測試和比較Numba優化函數與標準NumPy實現的性能差異。
提供詳細的基準測試結果，用於驗證優化效果。
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Callable
from scipy.fft import fft, fftfreq
import statistics

# 導入標準和優化版本的函數
try:
    from .numba_optimized import (
        hanning_window_numba, power_spectrum_numba, band_power_extraction_numba,
        signal_quality_z_score_numba, spectral_features_numba, 
        filter_power_calculation_numba, NUMBA_AVAILABLE
    )
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

logger = logging.getLogger(__name__)


class NumbaPerformanceBenchmark:
    """Numba性能基準測試類別"""
    
    def __init__(self, warmup_runs: int = 3, test_runs: int = 10):
        """
        初始化基準測試
        
        Args:
            warmup_runs: 預熱運行次數（讓JIT編譯）
            test_runs: 正式測試運行次數
        """
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
        self.results = {}
        
        # 測試數據大小
        self.test_sizes = [256, 512, 1024, 2048]
        
        # 生成測試數據
        self.test_data = {}
        for size in self.test_sizes:
            np.random.seed(42)  # 固定種子確保可重複性
            self.test_data[size] = {
                'raw_signal': np.random.randn(size).astype(np.float64),
                'fft_data': None,
                'freqs': None,
                'psd': None
            }
            
            # 預計算FFT相關數據
            windowed = self.test_data[size]['raw_signal'] * np.hanning(size)
            fft_data = fft(windowed)
            freqs = fftfreq(size, 1/512)
            psd = np.abs(fft_data) ** 2
            
            self.test_data[size]['fft_data'] = fft_data
            self.test_data[size]['freqs'] = freqs[:size//2]
            self.test_data[size]['psd'] = psd[:size//2]
    
    def time_function(self, func: Callable, *args, **kwargs) -> Tuple[float, any]:
        """測量函數執行時間"""
        # 預熱運行
        for _ in range(self.warmup_runs):
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"預熱運行失敗: {e}")
                return float('inf'), None
        
        # 正式測試
        times = []
        result = None
        
        for _ in range(self.test_runs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"測試運行失敗: {e}")
                return float('inf'), None
        
        # 返回平均執行時間
        avg_time = statistics.mean(times)
        return avg_time, result
    
    def benchmark_hanning_window(self) -> Dict:
        """基準測試Hanning窗函數"""
        results = {'numpy': {}, 'numba': {}, 'speedup': {}}
        
        for size in self.test_sizes:
            # NumPy版本
            numpy_time, numpy_result = self.time_function(np.hanning, size)
            results['numpy'][size] = numpy_time
            
            if NUMBA_AVAILABLE:
                # Numba版本
                numba_time, numba_result = self.time_function(hanning_window_numba, size)
                results['numba'][size] = numba_time
                
                # 計算加速比
                if numba_time > 0:
                    speedup = numpy_time / numba_time
                    results['speedup'][size] = speedup
                else:
                    results['speedup'][size] = float('inf')
            else:
                results['numba'][size] = float('inf')
                results['speedup'][size] = 0.0
        
        return results
    
    def benchmark_power_spectrum(self) -> Dict:
        """基準測試功率譜計算"""
        results = {'numpy': {}, 'numba': {}, 'speedup': {}}
        
        for size in self.test_sizes:
            fft_data = self.test_data[size]['fft_data']
            
            # NumPy版本
            numpy_func = lambda x: np.abs(x) ** 2
            numpy_time, numpy_result = self.time_function(numpy_func, fft_data)
            results['numpy'][size] = numpy_time
            
            if NUMBA_AVAILABLE:
                # Numba版本
                numba_time, numba_result = self.time_function(power_spectrum_numba, fft_data)
                results['numba'][size] = numba_time
                
                # 計算加速比
                if numba_time > 0:
                    speedup = numpy_time / numba_time
                    results['speedup'][size] = speedup
                else:
                    results['speedup'][size] = float('inf')
            else:
                results['numba'][size] = float('inf')
                results['speedup'][size] = 0.0
        
        return results
    
    def benchmark_band_power_extraction(self) -> Dict:
        """基準測試頻帶功率提取"""
        results = {'numpy': {}, 'numba': {}, 'speedup': {}}
        
        for size in self.test_sizes:
            psd = self.test_data[size]['psd']
            freqs = self.test_data[size]['freqs']
            low_freq, high_freq = 8.0, 12.0  # Alpha頻帶
            
            # NumPy版本
            def numpy_band_power(psd, freqs, low, high):
                band_indices = np.where((freqs >= low) & (freqs <= high))[0]
                return np.mean(psd[band_indices]) if len(band_indices) > 0 else 0.0
            
            numpy_time, numpy_result = self.time_function(
                numpy_band_power, psd, freqs, low_freq, high_freq
            )
            results['numpy'][size] = numpy_time
            
            if NUMBA_AVAILABLE:
                # Numba版本
                numba_time, numba_result = self.time_function(
                    band_power_extraction_numba, psd, freqs, low_freq, high_freq
                )
                results['numba'][size] = numba_time
                
                # 計算加速比
                if numba_time > 0:
                    speedup = numpy_time / numba_time
                    results['speedup'][size] = speedup
                else:
                    results['speedup'][size] = float('inf')
            else:
                results['numba'][size] = float('inf')
                results['speedup'][size] = 0.0
        
        return results
    
    def benchmark_signal_quality(self) -> Dict:
        """基準測試信號品質計算"""
        results = {'numpy': {}, 'numba': {}, 'speedup': {}}
        
        for size in self.test_sizes:
            data = self.test_data[size]['raw_signal']
            
            # NumPy版本
            def numpy_signal_quality(data):
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                outlier_ratio = np.sum(z_scores > 3.0) / len(data)
                return max(0.0, 100.0 * (1.0 - outlier_ratio * 2.0))
            
            numpy_time, numpy_result = self.time_function(numpy_signal_quality, data)
            results['numpy'][size] = numpy_time
            
            if NUMBA_AVAILABLE:
                # Numba版本
                numba_time, numba_result = self.time_function(signal_quality_z_score_numba, data)
                results['numba'][size] = numba_time
                
                # 計算加速比
                if numba_time > 0:
                    speedup = numpy_time / numba_time
                    results['speedup'][size] = speedup
                else:
                    results['speedup'][size] = float('inf')
            else:
                results['numba'][size] = float('inf')
                results['speedup'][size] = 0.0
        
        return results
    
    def benchmark_spectral_features(self) -> Dict:
        """基準測試頻譜特徵計算"""
        results = {'numpy': {}, 'numba': {}, 'speedup': {}}
        
        for size in self.test_sizes:
            freqs = self.test_data[size]['freqs']
            psd = self.test_data[size]['psd']
            
            # NumPy版本
            def numpy_spectral_features(freqs, psd):
                if len(freqs) == 0 or len(psd) == 0:
                    return 0.0, 0.0
                
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_bandwidth = np.sqrt(
                    np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd)
                )
                return spectral_centroid, spectral_bandwidth
            
            numpy_time, numpy_result = self.time_function(numpy_spectral_features, freqs, psd)
            results['numpy'][size] = numpy_time
            
            if NUMBA_AVAILABLE:
                # Numba版本
                numba_time, numba_result = self.time_function(spectral_features_numba, freqs, psd)
                results['numba'][size] = numba_time
                
                # 計算加速比
                if numba_time > 0:
                    speedup = numpy_time / numba_time
                    results['speedup'][size] = speedup
                else:
                    results['speedup'][size] = float('inf')
            else:
                results['numba'][size] = float('inf')
                results['speedup'][size] = 0.0
        
        return results
    
    def benchmark_filter_power(self) -> Dict:
        """基準測試濾波器功率計算"""
        results = {'numpy': {}, 'numba': {}, 'speedup': {}}
        
        for size in self.test_sizes:
            filtered_signal = self.test_data[size]['raw_signal']
            
            # NumPy版本
            numpy_func = lambda x: np.mean(x ** 2)
            numpy_time, numpy_result = self.time_function(numpy_func, filtered_signal)
            results['numpy'][size] = numpy_time
            
            if NUMBA_AVAILABLE:
                # Numba版本
                numba_time, numba_result = self.time_function(
                    filter_power_calculation_numba, filtered_signal
                )
                results['numba'][size] = numba_time
                
                # 計算加速比
                if numba_time > 0:
                    speedup = numpy_time / numba_time
                    results['speedup'][size] = speedup
                else:
                    results['speedup'][size] = float('inf')
            else:
                results['numba'][size] = float('inf')
                results['speedup'][size] = 0.0
        
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """運行所有基準測試"""
        if not BENCHMARK_AVAILABLE:
            logger.error("❌ Numba benchmark not available - missing dependencies")
            return {}
        
        logger.info("🚀 開始Numba性能基準測試...")
        
        benchmarks = {
            'hanning_window': self.benchmark_hanning_window,
            'power_spectrum': self.benchmark_power_spectrum,
            'band_power_extraction': self.benchmark_band_power_extraction,
            'signal_quality': self.benchmark_signal_quality,
            'spectral_features': self.benchmark_spectral_features,
            'filter_power': self.benchmark_filter_power,
        }
        
        results = {}
        for name, benchmark_func in benchmarks.items():
            logger.info(f"📊 測試 {name}...")
            results[name] = benchmark_func()
        
        self.results = results
        return results
    
    def print_results(self):
        """打印基準測試結果"""
        if not self.results:
            logger.warning("⚠️ 沒有基準測試結果可顯示")
            return
        
        print("\n🎯 Numba性能基準測試結果")
        print("=" * 80)
        
        for func_name, func_results in self.results.items():
            print(f"\n📈 {func_name.replace('_', ' ').title()}")
            print("-" * 60)
            
            print(f"{'數據大小':<10} {'NumPy(ms)':<12} {'Numba(ms)':<12} {'加速比':<10} {'狀態'}")
            print("-" * 60)
            
            for size in self.test_sizes:
                numpy_time = func_results['numpy'].get(size, 0) * 1000  # 轉換為毫秒
                numba_time = func_results['numba'].get(size, 0) * 1000
                speedup = func_results['speedup'].get(size, 0)
                
                if speedup == float('inf'):
                    status = "❌ 錯誤"
                    speedup_str = "N/A"
                elif speedup > 1.5:
                    status = "🚀 優化"
                    speedup_str = f"{speedup:.2f}x"
                elif speedup > 0.8:
                    status = "➡️ 相近"
                    speedup_str = f"{speedup:.2f}x"
                else:
                    status = "🐌 較慢"
                    speedup_str = f"{speedup:.2f}x"
                
                print(f"{size:<10} {numpy_time:<12.3f} {numba_time:<12.3f} {speedup_str:<10} {status}")
        
        # 計算整體統計
        print(f"\n📊 整體性能統計")
        print("=" * 40)
        
        all_speedups = []
        for func_results in self.results.values():
            for speedup in func_results['speedup'].values():
                if speedup != float('inf') and speedup > 0:
                    all_speedups.append(speedup)
        
        if all_speedups:
            avg_speedup = statistics.mean(all_speedups)
            max_speedup = max(all_speedups)
            min_speedup = min(all_speedups)
            
            print(f"平均加速比: {avg_speedup:.2f}x")
            print(f"最大加速比: {max_speedup:.2f}x") 
            print(f"最小加速比: {min_speedup:.2f}x")
            
            improved_count = sum(1 for s in all_speedups if s > 1.0)
            total_count = len(all_speedups)
            improvement_rate = (improved_count / total_count) * 100
            
            print(f"改善率: {improvement_rate:.1f}% ({improved_count}/{total_count})")
        
        print("\n✅ 基準測試完成")


def run_benchmark():
    """運行基準測試的便捷函數"""
    benchmark = NumbaPerformanceBenchmark(warmup_runs=5, test_runs=15)
    results = benchmark.run_all_benchmarks()
    benchmark.print_results()
    return results


if __name__ == "__main__":
    # 直接運行基準測試
    run_benchmark()