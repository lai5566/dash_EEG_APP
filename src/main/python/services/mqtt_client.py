"""MQTT感測器客戶端服務"""

import json
import time
import threading
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from typing import Dict, Any, Optional, Callable

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    # 為類型註解創建模擬mqtt對象
    class mqtt:
        class Client:
            pass

from resources.config.mqtt_config import MQTT_CONFIG, MQTT_TOPICS

logger = logging.getLogger(__name__)


class MQTTSensorClient:
    """MQTT感測器資料客戶端"""
    
    def __init__(self, data_buffer, db_writer, config: Dict[str, Any] = None):
        self.data_buffer = data_buffer
        self.db_writer = db_writer
        self.config = config or MQTT_CONFIG
        self.client = None
        self.running = False
        self.connected = False
        self.message_handlers = {}
        
        # 回調函數
        self.on_sensor_data_callback = None
        
    def set_sensor_data_callback(self, callback: Callable):
        """設定感測器資料回調"""
        self.on_sensor_data_callback = callback
        
    def on_connect(self, client, userdata, flags, rc):
        """連接回調"""
        if rc == 0:
            logger.info(f"✅ MQTT connected to {self.config['broker_host']}:{self.config['broker_port']}")
            self.connected = True
            
            # 訂閱所有感測器主題
            for topic_name, topic in MQTT_TOPICS.items():
                try:
                    client.subscribe(topic, qos=self.config['qos'])
                    logger.info(f"📡 Subscribed to topic: {topic}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {topic}: {e}")
                    
        else:
            logger.error(f"❌ MQTT connection failed with code: {rc}")
            self.connected = False
            
    def on_disconnect(self, client, userdata, rc):
        """斷線回調"""
        logger.warning(f"⚠️ MQTT disconnected with code: {rc}")
        self.connected = False
        
    def on_message(self, client, userdata, msg):
        """收到訊息回調"""
        try:
            topic = msg.topic
            payload = msg.payload.decode()
            timestamp = time.time()
            
            logger.debug(f"Received message on topic {topic}: {payload}")
            
            # 解析JSON資料
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in message: {payload}")
                return
                
            # 處理不同的訊息類型
            if topic == MQTT_TOPICS['sensor_data']:
                self._handle_sensor_data(data, timestamp)
            elif topic == MQTT_TOPICS['eeg_data']:
                self._handle_eeg_data(data, timestamp)
            elif topic == MQTT_TOPICS['system_status']:
                self._handle_system_status(data, timestamp)
            elif topic == MQTT_TOPICS['commands']:
                self._handle_commands(data, timestamp)
            else:
                logger.warning(f"Unknown topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    def _handle_sensor_data(self, data: Dict[str, Any], timestamp: float):
        """處理感測器資料"""
        try:
            # 提取感測器數值
            temperature = data.get('temperature', 0.0)
            humidity = data.get('humidity', 0.0)
            light = data.get('light', 0)
            
            # 驗證資料
            if not isinstance(temperature, (int, float)):
                temperature = 0.0
            if not isinstance(humidity, (int, float)):
                humidity = 0.0
            if not isinstance(light, (int, float)):
                light = 0
                
            # 更新資料緩衝區（即時顯示用，不需要會話）
            self.data_buffer.add_sensor_data(temperature, humidity, light)
            
            # 只在有活動會話時才儲存至資料庫
            if self.db_writer.current_session_id:
                # 儲存至資料庫
                self.db_writer.add_sensor_data(timestamp, temperature, humidity, light)
                
                # 將MQTT感測器資料也添加到統一記錄聚合器 (與main.py邏輯一致)
                mqtt_data = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'light': light
                }
                self.db_writer.add_data_to_aggregator(timestamp, None, **mqtt_data)
                logger.debug(f"MQTT sensor data recorded - Session: {self.db_writer.current_session_id}")
            else:
                logger.debug(f"📊 MQTT Sensor data updated (display only): T={temperature}°C, H={humidity}%, L={light}")
            
            # 如果設定則呼叫回調
            if self.on_sensor_data_callback:
                self.on_sensor_data_callback(temperature, humidity, light, timestamp)
            
        except Exception as e:
            logger.error(f"Error handling MQTT sensor data: {e}")
            
    def _handle_eeg_data(self, data: Dict[str, Any], timestamp: float):
        """處理來自MQTT的EEG資料"""
        try:
            # 這可以用於分散式EEG處理
            logger.debug(f"Received EEG data via MQTT: {data}")
            
        except Exception as e:
            logger.error(f"Error handling EEG data: {e}")
            
    def _handle_system_status(self, data: Dict[str, Any], timestamp: float):
        """處理系統狀態訊息"""
        try:
            logger.info(f"System status update: {data}")
            
        except Exception as e:
            logger.error(f"Error handling system status: {e}")
            
    def _handle_commands(self, data: Dict[str, Any], timestamp: float):
        """處理命令訊息"""
        try:
            command = data.get('command')
            params = data.get('params', {})
            
            if command == 'start_recording':
                logger.info("Received start recording command via MQTT")
            elif command == 'stop_recording':
                logger.info("Received stop recording command via MQTT")
            elif command == 'update_config':
                logger.info("Received config update command via MQTT")
            else:
                logger.warning(f"Unknown command: {command}")
                
        except Exception as e:
            logger.error(f"Error handling commands: {e}")
            
    def publish_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """發布訊息到MQTT代理伺服器"""
        try:
            if not self.connected or not self.client:
                logger.warning("MQTT client not connected")
                return False
                
            payload = json.dumps(message)
            result = self.client.publish(topic, payload, qos=self.config['qos'])
            
            if result.rc == 0:
                logger.debug(f"Published message to {topic}: {payload}")
                return True
            else:
                logger.error(f"Failed to publish message to {topic}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
            
    def start(self) -> Optional[mqtt.Client]:
        """啟動MQTT客戶端"""
        if not MQTT_AVAILABLE:
            logger.warning("⚠️ MQTT module not installed, skipping MQTT functionality")
            return None
            
        try:
            self.client = mqtt.Client(
                client_id=self.config['client_id'],
                clean_session=self.config['clean_session']
            )
            
            # 設定回調
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # 如果提供則設定用戶名和密碼
            if self.config['username'] and self.config['password']:
                self.client.username_pw_set(
                    self.config['username'], 
                    self.config['password']
                )
                
            # 連接到代理伺服器
            self.client.connect(
                self.config['broker_host'],
                self.config['broker_port'],
                self.config['keepalive']
            )
            
            # 啟動循環
            self.client.loop_start()
            self.running = True
            
            logger.info("MQTT client started successfully")
            return self.client
            
        except Exception as e:
            logger.error(f"❌ Failed to start MQTT client: {e}")
            return None
            
    def stop(self):
        """停止MQTT客戶端"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                
            self.running = False
            self.connected = False
            logger.info("MQTT client stopped")
            
        except Exception as e:
            logger.error(f"Error stopping MQTT client: {e}")
            
    def is_connected(self) -> bool:
        """檢查MQTT客戶端是否已連接"""
        return self.connected
        
    def get_status(self) -> Dict[str, Any]:
        """取得MQTT客戶端狀態"""
        return {
            'connected': self.connected,
            'running': self.running,
            'broker_host': self.config['broker_host'],
            'broker_port': self.config['broker_port'],
            'client_id': self.config['client_id'],
            'available': MQTT_AVAILABLE
        }