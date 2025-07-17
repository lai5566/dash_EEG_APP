"應用程式配置設定"""

import os

# 應用程式設定
APP_CONFIG = {
    'name': 'EEG Dashboard Application',
    'version': '1.0.0',
    'buffer_size': 2048,
    'sample_rate': 512,
    'window_size': 1024,
    'max_recording_duration': 3600,  # 1小時
}

# 網頁伺服器設定
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8052,
    'debug': False,
    'threaded': True,
    'buffer_size': 2048,
    'sample_rate': 512,
    'window_size': 1024,
}

# 資料庫設定
DATABASE_PATH = os.path.join(os.getcwd(), "enhanced_eeg.db")
DATABASE_BATCH_SIZE = 100
DATABASE_WRITE_INTERVAL = 2.0  # 秒

# 使用者介面設定
UI_CONFIG = {
    'title': "EEG監控系統",
    'max_points': 20,
    'chart_height': 400,
    'update_interval': 300,  # 毫秒 (aligned with B.py)
    'theme': 'light',
}

# EEG處理設定
PROCESSING_CONFIG = {
    'artifact_threshold': 3.0,
    'filter_low_cutoff': 1.0,
    'filter_high_cutoff': 50.0,
    'notch_frequencies': [50, 60],
    'window_functions': ['hanning', 'hamming', 'blackman'],
    'default_window': 'hanning',
    'signal_quality_good_threshold': 70.0,  # 大於此值為良好品質
    'signal_quality_poor_threshold': 50.0,  # 小於此值為差品質
}

# FFT測試資料設定
FFT_TEST_DATA_CONFIG = {
    'amplitudes': {
        'delta': 0.15,    # 2Hz Delta波振幅
        'theta': 0.12,    # 6Hz Theta波振幅
        'alpha': 0.10,    # 10Hz Alpha波振幅
        'beta': 0.08,     # 20Hz Beta波振幅
        'gamma': 0.05,    # 40Hz Gamma波振幅
        'noise': 0.02     # 雜訊振幅
    },
    'frequencies': {
        'delta': 2.0,     # Delta波頻率
        'theta': 6.0,     # Theta波頻率
        'alpha': 10.0,    # Alpha波頻率
        'beta': 20.0,     # Beta波頻率
        'gamma': 40.0     # Gamma波頻率
    }
}

# 資料匯出設定
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'mat', 'edf'],
    'default_format': 'csv',
    'output_directory': os.path.join(os.getcwd(), 'output'),
    'max_export_duration': 3600,  # 1小時
}

# 全域狀態
USE_MOCK_DATA = True  # 暫時啟用以測試ASIC功能
RECORDING_STATE = {
    'is_recording': False,
    'current_group_id': None,
    'recording_thread': None,
    'audio_data': [],
    'start_time': None
}

# 日誌配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(os.getcwd(), 'logs', 'eeg_app.log'),
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# 建立目錄（如果不存在）
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(EXPORT_CONFIG['output_directory'], exist_ok=True)
os.makedirs(os.path.dirname(LOGGING_CONFIG['file']), exist_ok=True)