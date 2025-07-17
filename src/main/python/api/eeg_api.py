"""EEG API 端點"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import threading
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from typing import Dict, List

from core.eeg_processor import RealTimeEEGProcessor
from models.data_buffer import EnhancedCircularBuffer
from services.database_service import EnhancedDatabaseWriter
from services.serial_service import mock_serial_worker
from utils.data_utils import DataValidator
from resources.config.app_config import API_CONFIG

app = Flask(__name__)
CORS(app)

# 全域變數
eeg_buffer = None
db_writer = None
processor = None
is_running = False


def initialize_api():
    """初始化API組件"""
    global eeg_buffer, db_writer, processor
    
    eeg_buffer = EnhancedCircularBuffer(API_CONFIG['buffer_size'])
    db_writer = EnhancedDatabaseWriter()
    processor = RealTimeEEGProcessor(
        sample_rate=API_CONFIG['sample_rate'],
        window_size=API_CONFIG['window_size']
    )
    
    # 啟動資料庫寫入器
    db_writer.start()


@app.route('/api/status', methods=['GET'])
def get_status():
    """取得系統狀態"""
    return jsonify({
        'status': 'running' if is_running else 'stopped',
        'timestamp': time.time(),
        'buffer_size': eeg_buffer.size if eeg_buffer else 0,
        'sample_rate': API_CONFIG['sample_rate']
    })


@app.route('/api/eeg/raw', methods=['GET'])
def get_raw_eeg():
    """取得原始EEG資料"""
    if not eeg_buffer:
        return jsonify({'error': 'Buffer not initialized'}), 500
    
    try:
        data, timestamps = eeg_buffer.get_data()
        
        # 限制API回應的資料點數量
        max_points = request.args.get('max_points', 1000, type=int)
        if len(data) > max_points:
            step = len(data) // max_points
            data = data[::step]
            timestamps = timestamps[::step]
        
        return jsonify({
            'data': data.tolist(),
            'timestamps': timestamps.tolist(),
            'length': len(data),
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eeg/cognitive', methods=['GET'])
def get_cognitive_data():
    """取得認知指標"""
    if not eeg_buffer:
        return jsonify({'error': 'Buffer not initialized'}), 500
    
    try:
        cognitive_data = eeg_buffer.get_cognitive_data()
        return jsonify(cognitive_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eeg/bands', methods=['GET'])
def get_band_data():
    """取得頻帶資料"""
    if not eeg_buffer:
        return jsonify({'error': 'Buffer not initialized'}), 500
    
    try:
        asic_data = eeg_buffer.get_asic_data()
        return jsonify(asic_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eeg/blinks', methods=['GET'])
def get_blink_data():
    """取得眨眼事件"""
    if not eeg_buffer:
        return jsonify({'error': 'Buffer not initialized'}), 500
    
    try:
        blink_data = eeg_buffer.get_blink_data()
        return jsonify(blink_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sensors', methods=['GET'])
def get_sensor_data():
    """取得感測器資料"""
    if not eeg_buffer:
        return jsonify({'error': 'Buffer not initialized'}), 500
    
    try:
        sensor_data = eeg_buffer.get_sensor_data()
        return jsonify(sensor_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eeg/processed', methods=['GET'])
def get_processed_data():
    """取得已處理的EEG資料及分析"""
    if not processor:
        return jsonify({'error': 'Processor not initialized'}), 500
    
    try:
        # 取得目前視窗並進行處理
        processed_result = processor.process_current_window()
        
        if not processed_result:
            return jsonify({'error': 'No data available'}), 404
        
        # 將numpy陣列轉換為列表以便JSON序列化
        if 'processed_data' in processed_result:
            processed_result['processed_data'] = processed_result['processed_data'].tolist()
        
        return jsonify(processed_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/eeg/analysis', methods=['GET'])
def get_analysis():
    """取得全面的EEG分析"""
    if not eeg_buffer or not processor:
        return jsonify({'error': 'Components not initialized'}), 500
    
    try:
        # 取得所有資料類型
        raw_data, timestamps = eeg_buffer.get_data()
        cognitive_data = eeg_buffer.get_cognitive_data()
        asic_data = eeg_buffer.get_asic_data()
        blink_data = eeg_buffer.get_blink_data()
        sensor_data = eeg_buffer.get_sensor_data()
        
        # 取得處理後的分析
        processed_result = processor.process_current_window()
        
        analysis = {
            'timestamp': time.time(),
            'raw_data_points': len(raw_data),
            'cognitive': cognitive_data,
            'frequency_bands': asic_data,
            'blinks': blink_data,
            'sensors': sensor_data,
            'processed': processed_result,
            'signal_quality': {
                'current': cognitive_data.get('signal_quality', 200),
                'status': 'good' if cognitive_data.get('signal_quality', 200) < 50 else 'poor'
            }
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    """啟動錄製工作階段"""
    global is_running
    
    try:
        data = request.get_json() or {}
        session_name = data.get('session_name', f'recording_{int(time.time())}')
        
        # 在此處啟動錄製邏輯
        is_running = True
        
        return jsonify({
            'status': 'recording_started',
            'session_name': session_name,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    """停止錄製工作階段"""
    global is_running
    
    try:
        is_running = False
        
        return jsonify({
            'status': 'recording_stopped',
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """取得API配置"""
    return jsonify(API_CONFIG)


@app.route('/api/config', methods=['POST'])
def update_config():
    """更新API配置"""
    try:
        data = request.get_json()
        
        # 驗證配置
        validator = DataValidator()
        if not validator.validate_config(data):
            return jsonify({'error': 'Invalid configuration'}), 400
        
        # 更新配置（在實際實作中）
        # API_CONFIG.update(data)
        
        return jsonify({
            'status': 'config_updated',
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })


@app.errorhandler(404)
def not_found(error):
    """處理404錯誤"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """處理500錯誤"""
    return jsonify({'error': 'Internal server error'}), 500


def create_app():
    """建立並配置Flask應用程式"""
    initialize_api()
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )