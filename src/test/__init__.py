"""EEG系統測試模組

此模組包含EEG腦波監控系統的各種測試，包括：
- 單元測試 (unit/)
- 整合測試 (integration/)  
- Numba性能優化測試

主要測試文件：
- test_numba_optimization.py: Numba JIT優化功能測試
"""

import sys
import os

# 將主程式路徑添加到Python路徑，方便測試模組導入
main_python_path = os.path.join(os.path.dirname(__file__), '..', 'main', 'python')
if main_python_path not in sys.path:
    sys.path.insert(0, main_python_path)

__version__ = "1.0.0"
__author__ = "EEG Development Team"

# 快速測試函數
def run_numba_tests():
    """快速運行Numba優化測試"""
    try:
        from .test_numba_optimization import main
        main()
    except ImportError as e:
        print(f"無法運行Numba測試: {e}")
        print("請確保已安裝所有必要的依賴套件")

# 模組導出
__all__ = ['run_numba_tests']