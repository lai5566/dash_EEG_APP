"應用程式配置設定"""

import os

# 應用程式設定
APP_CONFIG = {
    'name': 'EEG Dashboard Application',
    'version': '1.0.0',
    'buffer_size': 512,  # 減少緩衝區大小以適應Pi4
    'sample_rate': 512,
    'window_size': 256,   # 減少窗口大小以提高性能
    'max_recording_duration': 3600,  # 1小時
}

# 網頁伺服器設定
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8052,
    'debug': False,
    'threaded': True,
    'buffer_size': 512,  # 與APP_CONFIG保持一致
    'sample_rate': 512,
    'window_size': 512,   # 與APP_CONFIG保持一致
}

# 資料庫設定
DATABASE_PATH = os.path.join(os.getcwd(), "enhanced_eeg.db")
DATABASE_BATCH_SIZE = 100
DATABASE_WRITE_INTERVAL = 2.0  # 秒

# 使用者介面設定
UI_CONFIG = {
    'title': "EEG Monitoring System",
    'max_points': 30,        # 減少最大顯示點數
    'chart_height': 500,     # 減少圖表高度以節省渲染資源
    'update_interval': 200,  # 增加更新間隔（毫秒）適應Pi4性能
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

# 信號預處理配置 - 更貼近原始EEG信號的設定
PREPROCESSING_CONFIG = {
    # 預處理模式: 'minimal', 'standard', 'full'
    'mode': 'minimal',
    
    # 最小預處理模式 - 保持最接近原始信號
    'minimal': {
        'description': 'Minimal preprocessing for raw EEG signal analysis',
        'dc_removal': True,              # 移除DC偏移 (推薦保留)
        'highpass_cutoff': 0.5,          # 高通濾波截止頻率 (Hz)
        'powerline_notch': False,        # 電力線陷波濾波 (可選)
        'bandpass_filter': False,        # 帶通濾波 (關閉以保持原始性)
        'normalization': False,          # Z-score標準化 (關閉以保持絕對電壓值)
        'artifact_removal': False,       # 偽影自動移除 (關閉)
        'window_compensation': True,     # 窗函數能量補償
        'preserve_units': True           # 保持μV單位
    },
    
    # 標準預處理模式 - 平衡原始性與噪聲抑制
    'standard': {
        'description': 'Standard preprocessing with basic noise reduction',
        'dc_removal': True,
        'highpass_cutoff': 0.5,
        'powerline_notch': True,         # 啟用電力線濾波
        'bandpass_filter': True,
        'bandpass_low': 0.5,
        'bandpass_high': 50.0,
        'normalization': False,          # 仍不標準化
        'artifact_removal': False,
        'window_compensation': True,
        'preserve_units': True
    },
    
    # 完整預處理模式 - 最大噪聲抑制 (保留原有邏輯)
    'full': {
        'description': 'Full preprocessing with maximum noise reduction',
        'dc_removal': True,
        'highpass_cutoff': 1.0,
        'powerline_notch': True,
        'bandpass_filter': True,
        'bandpass_low': 1.0,
        'bandpass_high': 50.0,
        'normalization': True,           # 只在此模式啟用標準化
        'artifact_removal': True,
        'window_compensation': True,
        'preserve_units': False          # 標準化後單位會改變
    },
    
    # 濾波器參數
    'filter_params': {
        'butter_order': 4,               # 巴特沃茲濾波器階數
        'notch_order': 2,                # 陷波濾波器階數
        'notch_q_factor': 30             # 陷波濾波器品質因子
    },
    
    # 窗函數設定 
    'windowing': {
        'type': 'hanning',               # 窗函數類型
        'compensation_factor': 2.0,      # Hanning窗能量補償係數
        'alternative_windows': ['hamming', 'blackman', 'rectangular']
    }
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

# FFT計算方法配置
FFT_CALCULATION_CONFIG = {
    'mode': 'power',  # 'power' 或 'waveform'
    'power_method': {
        'description': 'Display frequency band power values',
        'frequency_bands': {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        },
        'y_axis_label': 'Power (μV²)',
        'data_scaling': 1.0,
        'chart_title': 'FFT Band Power Analysis'
    },
    'waveform_method': {
        'description': 'Display frequency band filtered waveforms',
        'frequency_bands': {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12), 
            'beta': (12, 35),
            'gamma': (35, 50)
        },
        'y_axis_label': 'Voltage (μV)',
        'data_scaling': 1000000.0,  # Convert V to μV for display
        'chart_title': 'FFT Band Waveform Analysis'
    }
}

# 資料匯出設定
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'mat', 'edf'],
    'default_format': 'csv',
    'output_directory': os.path.join(os.getcwd(), 'output'),
    'max_export_duration': 3600,  # 1小時
}

# 平台檢測和優化設定
PLATFORM_CONFIG = {
    'is_raspberry_pi': os.path.exists('/proc/device-tree/model'),
    'raspberry_pi_optimizations': {
        'buffer_size': 512,        # 更小的緩衝區
        'window_size': 512,        # 更小的窗口
        'update_interval': 750,    # 更長的更新間隔
        'chart_height': 300,       # 更小的圖表
        'max_points': 10,          # 更少的顯示點
        'reduced_processing': True  # 啟用簡化處理
    }
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

# 平台優化應用函數
def apply_platform_optimizations():
    """根據平台自動應用優化設定"""
    if PLATFORM_CONFIG['is_raspberry_pi']:
        print("🍓 檢測到樹莓派平台，應用性能優化...")
        
        # 應用樹莓派優化設定
        optimizations = PLATFORM_CONFIG['raspberry_pi_optimizations']
        
        # 更新應用配置
        APP_CONFIG['buffer_size'] = optimizations['buffer_size']
        APP_CONFIG['window_size'] = optimizations['window_size']
        
        # 更新API配置
        API_CONFIG['buffer_size'] = optimizations['buffer_size']
        API_CONFIG['window_size'] = optimizations['window_size']
        
        # 更新UI配置
        UI_CONFIG['update_interval'] = optimizations['update_interval']
        UI_CONFIG['chart_height'] = optimizations['chart_height']
        UI_CONFIG['max_points'] = optimizations['max_points']
        
        print(f"已應用樹莓派優化設定:")
        print(f"   - 緩衝區大小: {optimizations['buffer_size']}")
        print(f"   - 窗口大小: {optimizations['window_size']}")
        print(f"   - 更新間隔: {optimizations['update_interval']}ms")
        print(f"   - 圖表高度: {optimizations['chart_height']}")
        print(f"   - 最大顯示點: {optimizations['max_points']}")
    else:
        print("檢測到標準平台，使用默認設定")

# 自動應用平台優化
apply_platform_optimizations()