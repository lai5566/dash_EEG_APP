#!/usr/bin/env python3
"""
Numba優化測試運行器

這是一個方便的腳本，可以從專案根目錄直接運行Numba優化測試。

使用方法:
python run_numba_test.py
"""

import sys
import os

# 確保可以找到測試模組
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
test_path = os.path.join(src_path, 'test')

if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    """執行Numba優化測試"""
    print("EEG System Numba Optimization Testing")
    print("=" * 50)
    print(f"專案根目錄: {project_root}")
    print(f"測試目錄: {test_path}")
    print()
    
    try:
        # 導入並運行測試
        from test.test_numba_optimization import main as run_tests
        run_tests()
    except ImportError as e:
        print(f"ERROR: Unable to import test module: {e}")
        print()
        print("請確保:")
        print("1. 專案結構正確")
        print("2. 已安裝所有必要的依賴套件")
        print("3. 在專案根目錄中運行此腳本")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()