"""MQTT配置設定"""

# MQTT代理伺服器設定
MQTT_CONFIG = {
    'broker_host': '192.168.11.88',
    'broker_port': 1883,
    'username': None,
    'password': None,
    'client_id': 'eeg_dashboard',
    'keepalive': 60,
    'clean_session': True,
    'qos': 1,
    'retain': False,
}

# MQTT主題
MQTT_TOPICS = {
    'eeg_data': 'eeg/data',
    'cognitive_data': 'eeg/cognitive',
    'sensor_data': 'sensor/data',
    'system_status': 'system/status',
    'commands': 'system/commands',
    'recordings': 'eeg/recordings',
    'alerts': 'system/alerts',
}

# 感測器資料主題
SENSOR_TOPICS = {
    'temperature': 'sensors/temperature',
    'humidity': 'sensors/humidity',
    'light': 'sensors/light',
    'motion': 'sensors/motion',
    'sound': 'sensors/sound',
}

# 資料發佈設定
PUBLISH_CONFIG = {
    'enable_publishing': False,
    'publish_interval': 1.0,  # 秒
    'compress_data': True,
    'include_timestamp': True,
    'max_message_size': 1024,  # 位元組
    'buffer_size': 100,
}

# 訂閱設定
SUBSCRIPTION_CONFIG = {
    'enable_subscriptions': False,
    'auto_reconnect': True,
    'reconnect_delay': 5,  # 秒
    'max_reconnect_attempts': 10,
    'message_timeout': 30,  # 秒
}

# SSL/TLS設定
SSL_CONFIG = {
    'enable_ssl': False,
    'ca_cert_path': None,
    'client_cert_path': None,
    'client_key_path': None,
    'verify_mode': 'CERT_REQUIRED',
    'ciphers': None,
}

# 日誌配置
MQTT_LOGGING = {
    'enable_logging': True,
    'log_level': 'INFO',
    'log_file': 'mqtt_client.log',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 3,
}

# 舊版相容設定
MQTT_BROKER = MQTT_CONFIG['broker_host']
MQTT_PORT = MQTT_CONFIG['broker_port']
MQTT_TOPIC = MQTT_TOPICS['sensor_data']
MQTT_KEEPALIVE = MQTT_CONFIG['keepalive']
MQTT_TIMEOUT = 10
MQTT_RETRY_ATTEMPTS = 3
MQTT_RETRY_DELAY = 5