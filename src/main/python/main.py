"""主要應用程式進入點"""

import os
import sys
import time
import threading
import multiprocessing
import logging
from typing import Dict, Any

# 將專案根目錄加入路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.eeg_processor import RealTimeEEGProcessor
from core.filter_processor import AdaptiveFilterProcessor
from models.data_buffer import EnhancedCircularBuffer
from services.database_service import EnhancedDatabaseWriter
from services.serial_service import mock_serial_worker, enhanced_serial_worker
from services.mqtt_client import MQTTSensorClient
from services.audio_recorder import AudioRecorder
from api.eeg_api import create_app
from ui.dash_app import EEGDashboardApp
from utils.data_utils import DataValidator, DataExporter
from resources.config.app_config import (
    APP_CONFIG, API_CONFIG, DATABASE_PATH, USE_MOCK_DATA, PROCESSING_CONFIG
)
from resources.config.audio_config import AUDIO_CONFIG

# 配置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EEGApplication:
    """主要EEG應用程式控制器"""
    
    def __init__(self):
        self.eeg_buffer = None
        self.db_writer = None
        self.processor = None
        self.filter_processor = None
        self.mqtt_client = None
        self.audio_recorder = None
        self.data_validator = None
        self.data_exporter = None
        self.serial_queue = None
        self.serial_process = None
        self.web_app = None
        self.dash_app = None
        self.web_thread = None
        self.data_thread = None
        self.running = False
        
        # 日誌速率限制
        self.last_log_time = {}
        self.log_interval = 5.0  # 每5秒最多記錄一次相同警告
        
    def initialize(self):
        """初始化應用程式組件"""
        try:
            logger.info("Initializing EEG Application...")
            
            # 初始化資料緩衝區
            self.eeg_buffer = EnhancedCircularBuffer(APP_CONFIG['buffer_size'])
            logger.info(f"Buffer initialized with size: {APP_CONFIG['buffer_size']}")
            
            # 初始化資料庫寫入器
            self.db_writer = EnhancedDatabaseWriter(DATABASE_PATH)
            logger.info(f"Database writer initialized: {DATABASE_PATH}")
            
            # 初始化處理器
            self.processor = RealTimeEEGProcessor(
                sample_rate=APP_CONFIG['sample_rate'],
                window_size=APP_CONFIG['window_size']
            )
            logger.info("EEG processor initialized")
            
            # 初始化濾波器處理器
            self.filter_processor = AdaptiveFilterProcessor(
                sample_rate=APP_CONFIG['sample_rate'],
                max_workers=4
            )
            logger.info("Filter processor initialized")
            
            # 初始化MQTT客戶端
            self.mqtt_client = MQTTSensorClient(
                data_buffer=self.eeg_buffer,
                db_writer=self.db_writer
            )
            logger.info("MQTT client initialized")
            
            # 初始化音訊錄製器
            self.audio_recorder = AudioRecorder(
                device_index=AUDIO_CONFIG.get('device_index'),
                sample_rate=AUDIO_CONFIG.get('sample_rate', 44100),
                channels=AUDIO_CONFIG.get('channels', 1)
            )
            # 自動偵測音訊設備
            self.audio_recorder.list_audio_devices()
            logger.info("Audio recorder initialized")
            
            # 初始化工具類別
            self.data_validator = DataValidator()
            self.data_exporter = DataExporter()
            logger.info("Data utilities initialized")
            
            # 初始化串列通訊
            self.serial_queue = multiprocessing.Queue()
            
            # 初始化Web API
            self.web_app = create_app()
            logger.info("Web API initialized")
            
            # 初始化Dash應用程式
            self.dash_app = EEGDashboardApp(
                data_buffer=self.eeg_buffer,
                db_writer=self.db_writer,
                processor=self.processor,
                mqtt_client=self.mqtt_client,
                audio_recorder=self.audio_recorder
            )
            logger.info("Dash web interface initialized")
            
            logger.info("Application initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            return False
    
    def start_serial_communication(self):
        """啟動串列通訊程序"""
        try:
            if USE_MOCK_DATA:
                logger.info("Starting mock serial data worker")
                worker_func = mock_serial_worker
            else:
                logger.info("Starting real serial data worker")
                worker_func = enhanced_serial_worker
            
            self.serial_process = multiprocessing.Process(
                target=worker_func,
                args=(self.serial_queue,),
                daemon=True
            )
            self.serial_process.start()
            logger.info("Serial communication started")
            
        except Exception as e:
            logger.error(f"Failed to start serial communication: {e}")
    
    def start_data_processing(self):
        """啟動資料處理線程"""
        try:
            self.data_thread = threading.Thread(
                target=self._data_processing_loop,
                daemon=True
            )
            self.data_thread.start()
            logger.info("Data processing thread started")
            
        except Exception as e:
            logger.error(f"Failed to start data processing: {e}")
    
    def start_web_server(self):
        """啟動Web伺服器線程"""
        try:
            self.web_thread = threading.Thread(
                target=self._web_server_loop,
                daemon=True
            )
            self.web_thread.start()
            logger.info(f"Web server starting on {API_CONFIG['host']}:{API_CONFIG['port']}")
            
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            
    def start_mqtt_client(self):
        """啟動MQTT客戶端"""
        try:
            if self.mqtt_client:
                self.mqtt_client.start()
                logger.info("MQTT client started")
            
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")
    
    def start_database_writer(self):
        """啟動資料庫寫入器"""
        try:
            self.db_writer.start()
            logger.info("Database writer started")
            
        except Exception as e:
            logger.error(f"Failed to start database writer: {e}")
    
    def _data_processing_loop(self):
        """主要資料處理迴圈"""
        logger.info("Data processing loop started")
        
        while self.running:
            try:
                # 從串列佇列獲取資料
                if not self.serial_queue.empty():
                    data = self.serial_queue.get_nowait()
                    
                    if self.data_validator.validate_serial_data(data):
                        self._process_serial_data(data)
                    else:
                        logger.warning("Invalid serial data received")
                
                # 處理當前視窗
                if self.processor:
                    processed_result = self.processor.process_current_window()
                    if processed_result:
                        self._handle_processed_data(processed_result)
                
                time.sleep(0.001)  # 1毫秒延遲
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                time.sleep(0.1)
    
    def _process_serial_data(self, data: Dict[str, Any]):
        """處理傳入的串列資料"""
        try:
            timestamp = data.get('timestamp', time.time())
            
            # 處理原始EEG資料
            if 'raw_value' in data:
                raw_value = data['raw_value']
                self.eeg_buffer.append(raw_value, timestamp)
                self.processor.add_sample(raw_value)
                # 注意：已移除 add_raw_data() 調用，避免與 UnifiedRecordAggregator 重複
            
            # 處理認知資料
            if any(key in data for key in ['attention', 'meditation', 'signal_quality']):
                self.eeg_buffer.add_cognitive_data(
                    attention=data.get('attention'),
                    meditation=data.get('meditation'),
                    signal_quality=data.get('signal_quality')
                )
                
                self.db_writer.add_cognitive_data(
                    timestamp,
                    data.get('attention', 0),
                    data.get('meditation', 0),
                    data.get('signal_quality', 200)
                )
            
            # 處理ASIC頻帶資料
            if 'asic_bands' in data:
                bands_data = data['asic_bands']
                print(f"[ASIC DEBUG] MainApp: Processing ASIC bands: {bands_data}")
                self.eeg_buffer.add_asic_bands(bands_data)
                self.db_writer.add_asic_data(timestamp, bands_data)
            
            # 處理眨眼事件
            if 'blink' in data:
                blink_intensity = data['blink']
                self.eeg_buffer.add_blink_event(blink_intensity)
                self.db_writer.add_blink_data(timestamp, blink_intensity)
            
            # 處理感測器數據 (如果存在)
            if any(key in data for key in ['temperature', 'humidity', 'light']):
                temperature = data.get('temperature', 0.0)
                humidity = data.get('humidity', 0.0)
                light = data.get('light', 0)
                
                # 更新緩衝區
                self.eeg_buffer.add_sensor_data(temperature, humidity, light)
                
                # 寫入感測器資料表
                self.db_writer.add_sensor_data(timestamp, temperature, humidity, light)
            
            # 將所有數據添加到統一記錄聚合器
            # 獲取當前錄音群組ID (如果正在錄音)
            current_group_id = None
            if self.audio_recorder and hasattr(self.audio_recorder, 'current_group_id'):
                current_group_id = self.audio_recorder.current_group_id
                
            # 確保有當前會話ID並將數據添加到聚合器
            if self.db_writer.current_session_id:
                # 移除data中的timestamp以避免參數衝突
                data_copy = {k: v for k, v in data.items() if k != 'timestamp'}
                self.db_writer.add_data_to_aggregator(timestamp, current_group_id, **data_copy)
                logger.debug(f"數據已添加到聚合器 - Session: {self.db_writer.current_session_id}, Group: {current_group_id}")
            else:
                logger.warning(f"無活動會話 - 數據未添加到unified_records。數據鍵: {list(data.keys())}")
            
        except Exception as e:
            logger.error(f"Error processing serial data: {e}")
    
    def _should_log(self, message_key: str) -> bool:
        """檢查是否應該記錄訊息(速率限制)"""
        current_time = time.time()
        if message_key not in self.last_log_time:
            self.last_log_time[message_key] = current_time
            return True
        elif current_time - self.last_log_time[message_key] >= self.log_interval:
            self.last_log_time[message_key] = current_time
            return True
        return False

    def _handle_processed_data(self, processed_data: Dict[str, Any]):
        """處理已處理的EEG資料"""
        try:
            # 記錄信號品質 (修正邏輯: 高分數=好品質)
            if 'signal_quality' in processed_data:
                quality = processed_data['signal_quality']
                good_threshold = PROCESSING_CONFIG['signal_quality_good_threshold']
                poor_threshold = PROCESSING_CONFIG['signal_quality_poor_threshold']
                
                if quality > good_threshold:
                    if self._should_log('good_signal'):
                        logger.debug(f"Good signal quality: {quality:.1f}")
                elif quality < poor_threshold:
                    if self._should_log('poor_signal'):
                        logger.warning(f"Poor signal quality: {quality:.1f}")
                else:
                    if self._should_log('fair_signal'):
                        logger.info(f"Fair signal quality: {quality:.1f}")
            
            # 檢查偽影 (加入速率限制)
            if 'artifacts' in processed_data:
                artifacts = processed_data['artifacts']
                if len(artifacts) > 0:
                    if self._should_log('artifacts'):
                        logger.info(f"Artifacts detected: {len(artifacts)} points")
            
        except Exception as e:
            logger.error(f"Error handling processed data: {e}")
    
    def _web_server_loop(self):
        """Web伺服器迴圈"""
        try:
            # 啟動Dash應用程式而非Flask API
            self.dash_app.run(
                host=API_CONFIG['host'],
                port=API_CONFIG['port'],
                debug=API_CONFIG['debug']
            )
        except Exception as e:
            logger.error(f"Web server error: {e}")
    
    def start(self):
        """啟動應用程式"""
        try:
            logger.info("Starting EEG Application...")
            
            if not self.initialize():
                logger.error("Failed to initialize application")
                return False
            
            self.running = True
            
            # 啟動組件
            self.start_database_writer()
            self.start_mqtt_client()
            self.start_serial_communication()
            self.start_data_processing()
            self.start_web_server()
            
            logger.info("EEG Application started successfully")
            logger.info(f"Web interface available at: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            return False
    
    def stop(self):
        """停止應用程式"""
        try:
            logger.info("Stopping EEG Application...")
            
            self.running = False
            
            # 停止串列程序
            if self.serial_process and self.serial_process.is_alive():
                self.serial_process.terminate()
                self.serial_process.join(timeout=5)
            
            # 停止MQTT客戶端
            if self.mqtt_client:
                self.mqtt_client.stop()
                
            # 停止資料庫寫入器
            if self.db_writer:
                self.db_writer.running = False
                
            # 停止濾波器處理器
            if self.filter_processor:
                self.filter_processor.cleanup()
            
            logger.info("EEG Application stopped")
            
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
    
    def export_data(self, start_time: float, end_time: float, format: str = 'csv', table: str = 'unified_records'):
        """匯出指定時間範圍的資料"""
        try:
            if not self.data_exporter:
                logger.error("Data exporter not initialized")
                return None
            
            return self.data_exporter.export_data(
                start_time=start_time,
                end_time=end_time,
                format=format,
                db_path=DATABASE_PATH,
                table=table
            )
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None


def main():
    """主要函式"""
    app = EEGApplication()
    
    try:
        if app.start():
            logger.info("Application running. Press Ctrl+C to stop.")
            
            # 保持主線程運行
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        app.stop()


if __name__ == "__main__":
    main()