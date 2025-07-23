"""核心EEG處理模組 - 整合Numba優化"""

from .eeg_processor import EEGProcessor, RealTimeEEGProcessor
from .filter_processor import OptimizedFilterProcessor, AdaptiveFilterProcessor

# 導入Numba優化支援
try:
    from .numba_optimized import check_numba_performance, NUMBA_AVAILABLE
    from .numba_benchmark import run_benchmark
    
    # 執行Numba性能檢查
    numba_status = check_numba_performance()
    
    if numba_status['available']:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("🚀 EEG核心模組已整合Numba優化")
        logger.info(f"📊 Numba狀態: {numba_status['message']}")
        
        # 導出優化標識
        __numba_enabled__ = True
        __optimization_level__ = "high"
    else:
        __numba_enabled__ = False
        __optimization_level__ = "standard"
        
except ImportError:
    __numba_enabled__ = False
    __optimization_level__ = "standard"
    NUMBA_AVAILABLE = False

# 模組級別的優化資訊
__all__ = [
    'EEGProcessor', 
    'RealTimeEEGProcessor', 
    'OptimizedFilterProcessor', 
    'AdaptiveFilterProcessor',
    '__numba_enabled__',
    '__optimization_level__'
]

# 如果Numba可用，添加基準測試函數
if __numba_enabled__:
    __all__.extend(['run_benchmark', 'check_numba_performance'])