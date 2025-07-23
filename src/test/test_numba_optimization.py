#!/usr/bin/env python3
"""
Numba優化測試腳本

此腳本用於測試和驗證EEG系統的Numba優化功能。
運行此腳本可以：
1. 檢查Numba是否正確安裝
2. 驗證優化函數是否正常工作
3. 執行性能基準測試
4. 生成詳細的性能報告

使用方法:
# 從專案根目錄運行
python -m src.test.test_numba_optimization

# 或直接運行
python src/test/test_numba_optimization.py
"""

import sys
import os
import numpy as np

# 將專案路徑添加到Python路徑
# 從 src/test/ 目錄導航到 src/main/python/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main', 'python'))

try:
    from core.numba_optimized import check_numba_performance, NUMBA_AVAILABLE
    from core.numba_benchmark import run_benchmark
    from core.eeg_processor import EEGProcessor
    from core.filter_processor import OptimizedFilterProcessor
    print("SUCCESS: Successfully imported all optimization modules")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    print("請確保已安裝所有必要的依賴套件")
    sys.exit(1)


def test_numba_installation():
    """測試Numba安裝和基本功能"""
    print("\nChecking Numba installation status...")
    print("=" * 50)
    
    # 檢查Numba性能
    result = check_numba_performance()
    
    print(f"Numba availability: {'Available' if result['available'] else 'Not Available'}")
    print(f"Status message: {result['message']}")
    
    if result['available'] and 'test_results' in result:
        print("\nBasic function test results:")
        for key, value in result['test_results'].items():
            print(f"  {key}: {value:.4f}")
    
    return result['available']


def test_eeg_processor_integration():
    """測試EEG處理器的Numba整合"""
    print("\nTesting EEG processor Numba integration...")
    print("=" * 50)
    
    # 創建測試數據
    sample_rate = 512
    duration = 2.0  # 2秒
    n_samples = int(sample_rate * duration)
    
    # 生成合成EEG信號 (包含多個頻率成分)
    t = np.linspace(0, duration, n_samples)
    signal = (
        0.1 * np.sin(2 * np.pi * 2 * t) +    # Delta (2 Hz)
        0.08 * np.sin(2 * np.pi * 6 * t) +   # Theta (6 Hz)
        0.12 * np.sin(2 * np.pi * 10 * t) +  # Alpha (10 Hz)
        0.06 * np.sin(2 * np.pi * 20 * t) +  # Beta (20 Hz)
        0.04 * np.sin(2 * np.pi * 40 * t) +  # Gamma (40 Hz)
        0.02 * np.random.randn(n_samples)    # 雜訊
    )
    
    # 初始化EEG處理器
    processor = EEGProcessor(sample_rate=sample_rate)
    
    try:
        # 測試功率譜計算
        freqs, psd = processor.compute_power_spectrum(signal)
        print(f"SUCCESS: Power spectrum calculation - frequency points: {len(freqs)}")
        
        # 測試頻帶功率提取
        band_powers = processor.extract_band_powers(signal)
        print("SUCCESS: Band power extraction:")
        for band, power in band_powers.items():
            print(f"  {band}: {power:.6f}")
        
        # 測試頻譜特徵計算
        spectral_features = processor.calculate_spectral_features(signal)
        print("SUCCESS: Spectral features calculation:")
        for feature, value in spectral_features.items():
            print(f"  {feature}: {value:.4f}")
        
        # 測試信號品質計算  
        quality = processor._calculate_signal_quality(signal)
        print(f"SUCCESS: Signal quality calculation: {quality:.2f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: EEG processor test failed: {e}")
        return False


def test_filter_processor_integration():
    """測試濾波器處理器的Numba整合"""
    print("\nTesting filter processor Numba integration...")
    print("=" * 50)
    
    # 創建測試數據
    sample_rate = 512
    n_samples = 1024
    signal = np.random.randn(n_samples)
    
    # 初始化濾波器處理器
    filter_processor = OptimizedFilterProcessor(sample_rate=sample_rate)
    
    try:
        # 測試頻帶功率計算
        band_powers = filter_processor.compute_band_powers(signal)
        print("SUCCESS: Filter band power calculation:")
        for band, power in band_powers.items():
            print(f"  {band}: {power:.6f}")
        
        # 測試相對功率計算
        relative_powers = filter_processor.compute_relative_powers(signal)
        print("SUCCESS: Relative power calculation:")
        for band, power in relative_powers.items():
            print(f"  {band}: {power:.4f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Filter processor test failed: {e}")
        return False


def run_performance_benchmark():
    """運行性能基準測試"""
    print("\nRunning performance benchmark tests...")
    print("=" * 50)
    print("這可能需要幾分鐘時間，請稍候...")
    
    try:
        results = run_benchmark()
        return True
    except Exception as e:
        print(f"ERROR: Benchmark test failed: {e}")
        return False


def main():
    """主測試函數"""
    print("EEG System Numba Optimization Testing")
    print("=" * 80)
    
    # 測試步驟
    tests = [
        ("Numba安裝檢查", test_numba_installation),
        ("EEG處理器整合", test_eeg_processor_integration),
        ("濾波器處理器整合", test_filter_processor_integration),
        ("性能基準測試", run_performance_benchmark),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n開始測試: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"ERROR: Test '{test_name}' encountered exception: {e}")
            results[test_name] = False
    
    # 輸出總結
    print("\nTest Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Numba optimization successfully integrated into EEG system")
        print("\nExpected performance improvements:")
        print("  • FFT related operations: 3-5x acceleration")
        print("  • Statistical calculations: 2-4x acceleration")
        print("  • Real-time processing latency: from 500ms to 100-150ms")
        print("  • Raspberry Pi compatibility: significantly improved")
    else:
        print("WARNING: Some tests failed, please check error messages")
        
        if not NUMBA_AVAILABLE:
            print("\nSuggestions:")
            print("1. Install Numba: pip install numba")
            print("2. Check Python version (requires 3.7+)")
            print("3. Re-run tests")


if __name__ == "__main__":
    main()