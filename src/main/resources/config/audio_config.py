"""音頻配置設定"""

# 音頻配置字典
AUDIO_CONFIG = {
    'device_index': None,  # 自動偵測 PD100X
    'sample_rate': 44100,
    'channels': 1,  # 單聲道
    'max_duration': 60,  # 秒
    'format': 'int16',
    'output_directory': 'recordings'
}

# 設備偵測
PD100X_DEVICE_NAMES = [
    "PD100X",
    "PODCAST MICROPHONE", 
    "USB AUDIO",
    "MICROPHONE",
]