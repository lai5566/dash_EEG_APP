"""音頻錄製服務"""

import os
import time
import threading
import uuid
import logging
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from resources.config.audio_config import AUDIO_CONFIG

logger = logging.getLogger(__name__)


class AudioRecorder:
    """音頻錄製服務"""
    
    def __init__(self, device_index: Optional[int] = None, 
                 sample_rate: int = 44100, channels: int = 1):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording_data = []
        self.is_recording = False
        self.recording_thread = None
        self.current_group_id = None
        self.start_time = None
        
        # 錄製狀態
        self.recording_state = {
            'is_recording': False,
            'current_group_id': None,
            'recording_thread': None,
            'audio_data': [],
            'start_time': None
        }
        
    def list_audio_devices(self) -> Optional[int]:
        """列出可用的音頻裝置並自動檢測最佳裝置"""
        if not AUDIO_AVAILABLE:
            logger.warning("Audio modules not installed")
            logger.info("Please install audio modules: pip install sounddevice scipy")
            return None
            
        try:
            logger.info("Querying audio devices...")
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            logger.info("-" * 80)
            
            pd100x_device = None
            recommended_device = None
            
            for i, dev in enumerate(devices):
                status = "✅" if dev['max_input_channels'] > 0 else "❌"
                default_marker = " (default)" if i == sd.default.device[0] else ""
                
                # 檢查PD100X或USB音頻裝置
                name_upper = dev['name'].upper()
                is_pd100x = any([
                    "PD100X" in name_upper,
                    "PODCAST MICROPHONE" in name_upper,
                    ("USB AUDIO" in name_upper and dev['max_input_channels'] > 0),
                    ("MICROPHONE" in name_upper and "USB" in name_upper),
                    # 裝置配置的特殊檢查
                    (i == 1 and "USB AUDIO" in name_upper and
                     dev['max_input_channels'] == 1 and
                     dev['default_samplerate'] == 44100.0 and
                     dev['hostapi'] == 0)
                ])
                
                pd100x_marker = "PD100X!" if is_pd100x and dev['max_input_channels'] > 0 else ""
                
                logger.info(f"  {status} {i}: {dev['name']}{default_marker}{pd100x_marker}")
                logger.info(f"      Input channels: {dev['max_input_channels']}, "
                           f"Output channels: {dev['max_output_channels']}")
                logger.info(f"      Default sample rate: {dev['default_samplerate']}Hz")
                logger.info(f"      Host API: {dev['hostapi']}")
                
                if is_pd100x and dev['max_input_channels'] > 0:
                    pd100x_device = i
                    logger.info(f"PD100X device detected: Index {i} - {dev['name']}")
                    logger.info(f"   Sample rate: {dev['default_samplerate']}Hz, "
                               f"Input channels: {dev['max_input_channels']}")
                elif dev['max_input_channels'] > 0 and recommended_device is None:
                    recommended_device = i
                    logger.info(f"Alternative input device: Index {i} - {dev['name']}")
                    
            logger.info(f"Total devices found: {len(devices)}")
            
            # 自動選擇最佳裝置
            if pd100x_device is not None:
                self.device_index = pd100x_device
                # 使用裝置的預設採樣率
                device_sample_rate = int(devices[pd100x_device]['default_samplerate'])
                self.sample_rate = device_sample_rate
                self.channels = 1  # PD100X是單聲道
                logger.info(f"✅ Auto-selected PD100X device: Index {pd100x_device}")
                logger.info(f"Updated settings: Sample rate={self.sample_rate}Hz, "
                           f"Channels={self.channels}")
                return pd100x_device
            elif recommended_device is not None:
                logger.info(f"Recommended device index: {recommended_device}")
                self.device_index = recommended_device
                return recommended_device
            else:
                logger.warning("No suitable input device found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            return None
            
    def _test_device(self) -> bool:
        """測試音頻裝置"""
        if not AUDIO_AVAILABLE:
            return False
            
        try:
            logger.info(f"Testing audio device {self.device_index}...")
            
            # 檢查裝置訊息
            devices = sd.query_devices()
            if self.device_index >= len(devices):
                logger.error(f"Device index {self.device_index} out of range")
                return False
                
            device_info = devices[self.device_index]
            logger.info(f"Using device: {device_info['name']}")
            
            if device_info['max_input_channels'] < self.channels:
                logger.error(f"Device doesn't support {self.channels} channels")
                return False
                
            # 測試錄製
            test_duration = 0.1  # 100ms測試
            test_recording = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                device=self.device_index
            )
            sd.wait()
            
            logger.info("✅ Device test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Device test failed: {e}")
            return False
            
    def start_recording(self, group_id: Optional[str] = None, db_writer=None) -> bool:
        """開始錄製音頻"""
        if not AUDIO_AVAILABLE:
            logger.error("Audio modules not installed, cannot record")
            return False
            
        if self.is_recording:
            logger.warning("Already recording")
            return False
            
        # 確保有活動會話以支援recording_group_id關聯
        if db_writer and db_writer.current_session_id is None:
            logger.info("No active session detected, creating default session for recording...")
            session_id = db_writer.create_default_session_if_needed()
            if session_id:
                logger.info(f"Auto-created session {session_id} for recording")
            else:
                logger.error("Failed to create session for recording")
                return False
            
        try:
            # 如果未提供則生成群組ID
            if not group_id:
                group_id = str(uuid.uuid4())[:8]
                
            # 測試裝置
            if not self._test_device():
                logger.error("Audio device test failed")
                return False
                
            # 初始化錄製
            self.recording_data = []
            self.is_recording = True
            self.current_group_id = group_id
            self.start_time = time.time()
            
            # 更新錄製狀態
            self.recording_state.update({
                'is_recording': True,
                'current_group_id': group_id,
                'start_time': self.start_time
            })
            
            logger.info(f"Starting recording with group ID: {group_id}")
            logger.info(f"Recording settings: {self.sample_rate}Hz, "
                       f"{self.channels} channels, device {self.device_index}")
            
            # 啟動錄製執行緒
            def record_thread():
                try:
                    logger.info("🎵 Recording thread started...")
                    
                    # 錄製最大持續時間或直到停止
                    max_duration = AUDIO_CONFIG.get('max_duration', 60)
                    recording = sd.rec(
                        int(max_duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='int16',
                        device=self.device_index
                    )
                    
                    logger.info(f"🎤 Recording... (max {max_duration} seconds)")
                    
                    # 等待錄製完成或被停止
                    start_time = time.time()
                    while self.is_recording and (time.time() - start_time) < max_duration:
                        time.sleep(0.1)
                        
                    if self.is_recording:
                        sd.stop()
                        logger.info("Recording stopped (max duration reached)")
                    else:
                        logger.info("Recording stopped manually")
                        
                    # 儲存錄製資料
                    actual_duration = time.time() - start_time
                    actual_frames = int(actual_duration * self.sample_rate)
                    frames_to_save = min(actual_frames, len(recording))
                    
                    self.recording_data = recording[:frames_to_save]
                    logger.info(f"Recording data saved ({frames_to_save} frames, "
                               f"{actual_duration:.1f}s)")
                    
                except Exception as e:
                    logger.error(f"Recording thread error: {e}")
                    
            self.recording_thread = threading.Thread(target=record_thread, daemon=True)
            self.recording_thread.start()
            
            self.recording_state['recording_thread'] = self.recording_thread
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False
            
    def stop_recording(self, db_writer=None) -> Optional[str]:
        """停止錄製並儲存檔案"""
        if not self.is_recording:
            logger.warning("Not currently recording")
            return None
            
        try:
            logger.info("Stopping recording...")
            
            # 停止錄製
            self.is_recording = False
            self.recording_state['is_recording'] = False
            
            try:
                sd.stop()
                logger.info("Audio stream stopped")
            except Exception as e:
                logger.warning(f"Error stopping audio stream: {e}")
                
            # 等待錄製執行緒完成
            if self.recording_thread:
                logger.info("Waiting for recording thread to finish...")
                self.recording_thread.join(timeout=5)
                if self.recording_thread.is_alive():
                    logger.warning("Recording thread did not finish in time")
                else:
                    logger.info("✅ Recording thread finished")
                    
            # 如果數據存在則儲存錄製
            if (self.current_group_id and 
                hasattr(self, 'recording_data') and 
                self.recording_data is not None and 
                len(self.recording_data) > 0):
                
                # 建立錄製目錄
                recordings_dir = AUDIO_CONFIG.get('output_directory', 'recordings')
                os.makedirs(recordings_dir, exist_ok=True)
                
                # 生成檔案名稱
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{recordings_dir}/recording_{self.current_group_id}_{timestamp}.wav"
                
                # 儲存WAV檔案
                wav.write(filename, self.sample_rate, self.recording_data)
                logger.info(f"WAV file saved: {filename}")
                
                # 驗證檔案
                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    logger.info(f"File verified: {filename} ({file_size} bytes)")
                    
                    # 如果提供則儲存到資料庫
                    if db_writer:
                        try:
                            end_time = time.time()
                            duration = end_time - self.start_time if self.start_time else 0
                            
                            db_writer.add_recording_file(
                                self.current_group_id,
                                filename,
                                self.start_time,
                                end_time,
                                self.sample_rate,
                                file_size
                            )
                            logger.info(f"Database record added (duration: {duration:.1f}s)")
                            
                        except Exception as e:
                            logger.error(f"Database recording failed: {e}")
                            
                    # 清理狀態
                    self._cleanup_recording_state()
                    
                    logger.info(f"Recording completed: {filename}")
                    return filename
                else:
                    logger.error(f"File not created: {filename}")
                    
            else:
                logger.error("No recording data to save")
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            
        # 即使失敗也要清理狀態
        self._cleanup_recording_state()
        return None
        
    def _cleanup_recording_state(self):
        """清理錄製狀態"""
        self.is_recording = False
        self.current_group_id = None
        self.start_time = None
        self.recording_thread = None
        
        self.recording_state.update({
            'is_recording': False,
            'current_group_id': None,
            'recording_thread': None,
            'start_time': None
        })
        
    def get_recording_status(self) -> Dict[str, Any]:
        """取得目前錄製狀態"""
        return {
            'is_recording': self.is_recording,
            'current_group_id': self.current_group_id,
            'start_time': self.start_time,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'device_index': self.device_index,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'audio_available': AUDIO_AVAILABLE,
            'recording_state': self.recording_state
        }
        
    def get_device_info(self) -> Dict[str, Any]:
        """取得目前裝置訊息"""
        if not AUDIO_AVAILABLE:
            return {'available': False, 'error': 'Audio modules not installed'}
            
        try:
            devices = sd.query_devices()
            if self.device_index is not None and self.device_index < len(devices):
                device = devices[self.device_index]
                return {
                    'available': True,
                    'index': self.device_index,
                    'name': device['name'],
                    'max_input_channels': device['max_input_channels'],
                    'max_output_channels': device['max_output_channels'],
                    'default_samplerate': device['default_samplerate'],
                    'hostapi': device['hostapi']
                }
            else:
                return {'available': True, 'error': 'No device selected'}
                
        except Exception as e:
            return {'available': False, 'error': str(e)}