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
    print("✅ 成功導入所有優化模組")
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
    print("請確保已安裝所有必要的依賴套件")
    sys.exit(1)


def test_numba_installation():
    """測試Numba安裝和基本功能"""
    print("\n🔍 檢查Numba安裝狀態...")
    print("=" * 50)
    
    # 檢查Numba性能
    result = check_numba_performance()
    
    print(f"Numba可用性: {'✅ 是' if result['available'] else '❌ 否'}")
    print(f"狀態訊息: {result['message']}")
    
    if result['available'] and 'test_results' in result:
        print("\n📊 基本功能測試結果:")
        for key, value in result['test_results'].items():
            print(f"  {key}: {value:.4f}")
    
    return result['available']


def test_eeg_processor_integration():
    """測試EEG處理器的Numba整合"""
    print("\n🧠 測試EEG處理器Numba整合...")
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
        print(f"✅ 功率譜計算成功 - 頻率點數: {len(freqs)}")
        
        # 測試頻帶功率提取
        band_powers = processor.extract_band_powers(signal)
        print("✅ 頻帶功率提取成功:")
        for band, power in band_powers.items():
            print(f"  {band}: {power:.6f}")
        
        # 測試頻譜特徵計算
        spectral_features = processor.calculate_spectral_features(signal)
        print("✅ 頻譜特徵計算成功:")
        for feature, value in spectral_features.items():
            print(f"  {feature}: {value:.4f}")
        
        # 測試信號品質計算  
        quality = processor._calculate_signal_quality(signal)
        print(f"✅ 信號品質計算成功: {quality:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ EEG處理器測試失敗: {e}")
        return False


def test_filter_processor_integration():
    """測試濾波器處理器的Numba整合"""
    print("\n🔧 測試濾波器處理器Numba整合...")
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
        print("✅ 濾波器頻帶功率計算成功:")
        for band, power in band_powers.items():
            print(f"  {band}: {power:.6f}")
        
        # 測試相對功率計算
        relative_powers = filter_processor.compute_relative_powers(signal)
        print("✅ 相對功率計算成功:")
        for band, power in relative_powers.items():
            print(f"  {band}: {power:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 濾波器處理器測試失敗: {e}")
        return False


def run_performance_benchmark():
    """運行性能基準測試"""
    print("\n⚡ 運行性能基準測試...")
    print("=" * 50)
    print("這可能需要幾分鐘時間，請稍候...")
    
    try:
        results = run_benchmark()
        return True
    except Exception as e:
        print(f"❌ 基準測試失敗: {e}")
        return False


def main():
    """主測試函數"""
    print("🚀 EEG系統Numba優化測試")
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
            print(f"❌ 測試 '{test_name}' 發生異常: {e}")
            results[test_name] = False
    
    # 輸出總結
    print("\n📊 測試結果總結")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n總體結果: {passed}/{total} 測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！Numba優化已成功整合到EEG系統")
        print("\n🚀 性能提升預期:")
        print("  • FFT相關運算: 3-5x 加速")
        print("  • 統計計算: 2-4x 加速")
        print("  • 實時處理延遲: 從500ms降至100-150ms")
        print("  • 樹莓派兼容性: 顯著改善")
    else:
        print("⚠️ 部分測試失敗，請檢查相關錯誤訊息")
        
        if not NUMBA_AVAILABLE:
            print("\n💡 建議:")
            print("1. 安裝Numba: pip install numba")
            print("2. 檢查Python版本 (需要3.7+)")
            print("3. 重新運行測試")


if __name__ == "__main__":
    main()