"""音頻配置設定"""

import os

# 音頻錄製設定
AUDIO_CONFIG = {
    'device_index': None,  # 自動偵測
    'sample_rate': 44100,
    'channels': 1,  # 單聲道
    'max_duration': 60,  # 秒
    'output_directory': os.path.join(os.getcwd(), 'recordings'),
    'file_format': 'wav',
    'bit_depth': 16,
    'buffer_size': 1024,
}

# 設備偵測設定
DEVICE_DETECTION = {
    'auto_detect_pd100x': True,
    'preferred_devices': [
        'PD100X',
        'PODCAST MICROPHONE',
        'USB AUDIO',
        'USB MICROPHONE'
    ],
    'fallback_to_default': True,
    'test_device_on_startup': True,
}

# 錄製品質設定
QUALITY_SETTINGS = {
    'noise_reduction': False,
    'auto_gain_control': False,
    'echo_cancellation': False,
    'volume_normalization': True,
    'silence_detection': False,
}

# 檔案管理設定
FILE_MANAGEMENT = {
    'auto_cleanup': True,
    'max_file_age_days': 30,
    'max_storage_size_mb': 1000,
    'compress_old_files': True,
    'backup_to_cloud': False,
}

# 建立輸出目錄（如果不存在）
os.makedirs(AUDIO_CONFIG['output_directory'], exist_ok=True)