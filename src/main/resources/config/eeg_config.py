"""EEG配置設定"""

# 串列埠配置
SERIAL_PORT = "/dev/tty.usbserial-11410"
BAUD_RATE = 57600

# 資料處理
WINDOW_SIZE = 512
FS = 512  # 取樣頻率 (修正為實際設備頻率 513.3Hz)
BATCH_SIZE = 100

# 效能設定
UPDATE_INTERVAL = 500  # 毫秒
ADAPTIVE_UPDATE = True
MIN_UPDATE_INTERVAL = 300
MAX_UPDATE_INTERVAL = 1000
CACHE_SIZE = 50
RENDER_OPTIMIZATION = True

# ThinkGear協定常數
SYNC = 0xaa
POOR_SIGNAL = 0x02
ATTENTION = 0x04
MEDITATION = 0x05
BLINK = 0x16
RAW_VALUE = 0x80
ASIC_EEG_POWER = 0x83

# EEG頻率頻帶
BANDS = {
    "Delta (0.5-4Hz)": (0.5, 4),
    "Theta (4-8Hz)": (4, 8),
    "Alpha (8-12Hz)": (8, 12),
    "Beta (12-35Hz)": (12, 35),
    "Gamma (35-50Hz)": (35, 50),
    "SMR (12-15Hz)": (12, 15),
    "Mu (8-13Hz)": (8, 13),
    "High-Gamma (50-80Hz)": (50, 80),
}

# ASIC頻帶名稱
ASIC_BANDS = ["Delta", "Theta", "Low-Alpha", "High-Alpha",
              "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma",
              "SMR", "Mu"]

# 樹莓派優化
RASPBERRY_PI_OPTIMIZATION = {
    'filter_order': 2,
    'use_float32': True,
    'parallel_processing': True,
    'memory_limit_mb': 512,
    'adaptive_update': True,
}