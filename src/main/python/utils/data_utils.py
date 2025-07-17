"""資料工具函式"""

import json
import csv
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import os
import time

logger = logging.getLogger(__name__)


class DataValidator:
    """資料驗證工具"""
    
    @staticmethod
    def validate_serial_data(data: Dict[str, Any]) -> bool:
        """驗證串列資料格式"""
        try:
            # 檢查資料是否為字典
            if not isinstance(data, dict):
                return False
            
            # 檢查必要的時間戳記
            if 'timestamp' not in data:
                return False
            
            # 驗證時間戳記
            timestamp = data['timestamp']
            if not isinstance(timestamp, (int, float)) or timestamp <= 0:
                return False
            
            # 驗證原始值（若存在）
            if 'raw_value' in data:
                raw_value = data['raw_value']
                if not isinstance(raw_value, (int, float)):
                    return False
                # 檢查合理的腦電圖電壓範圍（-0.01到0.01V）
                if abs(raw_value) > 0.01:
                    return False
            
            # 驗證認知資料（若存在）
            if 'attention' in data:
                attention = data['attention']
                if not isinstance(attention, int) or not (0 <= attention <= 100):
                    return False
            
            if 'meditation' in data:
                meditation = data['meditation']
                if not isinstance(meditation, int) or not (0 <= meditation <= 100):
                    return False
            
            if 'signal_quality' in data:
                signal_quality = data['signal_quality']
                if not isinstance(signal_quality, int) or not (0 <= signal_quality <= 200):
                    return False
            
            # 驗證ASIC頻帶（若存在）
            if 'asic_bands' in data:
                asic_bands = data['asic_bands']
                if not isinstance(asic_bands, list) or len(asic_bands) != 8:
                    return False
                for band in asic_bands:
                    if not isinstance(band, int) or band < 0:
                        return False
            
            # 驗證眨眼資料（若存在）
            if 'blink' in data:
                blink = data['blink']
                if not isinstance(blink, int) or not (0 <= blink <= 255):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating serial data: {e}")
            return False
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """驗證配置資料"""
        try:
            # 檢查基本結構
            if not isinstance(config, dict):
                return False
            
            # 驗證取樣率
            if 'sample_rate' in config:
                sample_rate = config['sample_rate']
                if not isinstance(sample_rate, int) or sample_rate <= 0:
                    return False
            
            # 驗證緩衝區大小
            if 'buffer_size' in config:
                buffer_size = config['buffer_size']
                if not isinstance(buffer_size, int) or buffer_size <= 0:
                    return False
            
            # 驗證主機和埠號
            if 'host' in config:
                host = config['host']
                if not isinstance(host, str):
                    return False
            
            if 'port' in config:
                port = config['port']
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False
    
    @staticmethod
    def validate_eeg_data(data: np.ndarray) -> bool:
        """驗證腦電圖資料陣列"""
        try:
            # 檢查是否為numpy陣列
            if not isinstance(data, np.ndarray):
                return False
            
            # 檢查是否為一維陣列
            if data.ndim != 1:
                return False
            
            # 檢查是否有資料
            if len(data) == 0:
                return False
            
            # 檢查是否有NaN或無限值
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                return False
            
            # 檢查合理的電壓範圍
            if np.max(np.abs(data)) > 1.0:  # 1V對於腦電圖來說非常高
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating EEG data: {e}")
            return False


class DataExporter:
    """資料匯出工具"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> Optional[str]:
        """匯出資料到CSV檔案"""
        try:
            if not data:
                return None
            
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            
            # 取得所有唯一鍵值
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"Data exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return None
    
    def export_to_json(self, data: Any, filename: str) -> Optional[str]:
        """匯出資料到JSON檔案"""
        try:
            filepath = os.path.join(self.output_dir, f"{filename}.json")
            
            with open(filepath, 'w') as jsonfile:
                json.dump(data, jsonfile, indent=2, default=str)
            
            logger.info(f"Data exported to JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return None
    
    def export_eeg_to_csv(self, data: np.ndarray, timestamps: np.ndarray, filename: str) -> Optional[str]:
        """匯出腦電圖資料到CSV檔案"""
        try:
            if len(data) != len(timestamps):
                logger.error("資料和時間戳記長度不匹配")
                return None
            
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'eeg_value': data
            })
            
            df.to_csv(filepath, index=False)
            logger.info(f"EEG data exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting EEG to CSV: {e}")
            return None
    
    def export_database_table(self, db_path: str, table_name: str, filename: str) -> Optional[str]:
        """匯出資料庫表格到CSV"""
        try:
            conn = sqlite3.connect(db_path)
            
            # 檢查表格是否存在
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            
            if not cursor.fetchone():
                logger.error(f"資料庫中找不到表格 {table_name}")
                return None
            
            # 匯出到CSV
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            filepath = os.path.join(self.output_dir, f"{filename}.csv")
            df.to_csv(filepath, index=False)
            
            conn.close()
            logger.info(f"Database table exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting database table: {e}")
            return None
    
    def export_data(self, start_time: float, end_time: float, format: str = 'csv', db_path: str = None):
        """匯出指定時間範圍的資料"""
        try:
            if not db_path or not os.path.exists(db_path):
                logger.error("找不到資料庫")
                return None
            
            conn = sqlite3.connect(db_path)
            
            # 查詢指定時間範圍的資料
            query = """
                SELECT timestamp, voltage as eeg_value 
                FROM raw_data 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(start_time, end_time))
            
            if df.empty:
                logger.warning("指定時間範圍內未找到資料")
                return None
            
            # 產生檔案名稱
            start_date = datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H%M%S')
            end_date = datetime.fromtimestamp(end_time).strftime('%Y%m%d_%H%M%S')
            filename = f"eeg_data_{start_date}_to_{end_date}"
            
            if format.lower() == 'csv':
                filepath = os.path.join(self.output_dir, f"{filename}.csv")
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                filepath = os.path.join(self.output_dir, f"{filename}.json")
                df.to_json(filepath, orient='records', indent=2)
            else:
                logger.error(f"不支援的格式：{format}")
                return None
            
            conn.close()
            logger.info(f"Data exported: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None


class DataProcessor:
    """資料處理工具"""
    
    @staticmethod
    def resample_data(data: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
        """重新取樣資料到目標頻率"""
        try:
            from scipy import signal
            
            # 計算重新取樣比例
            ratio = target_fs / original_fs
            
            # 重新取樣
            resampled = signal.resample(data, int(len(data) * ratio))
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return data
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'z-score') -> np.ndarray:
        """使用指定方法正規化資料"""
        try:
            if method == 'z-score':
                return (data - np.mean(data)) / np.std(data)
            elif method == 'min-max':
                return (data - np.min(data)) / (np.max(data) - np.min(data))
            elif method == 'robust':
                median = np.median(data)
                mad = np.median(np.abs(data - median))
                return (data - median) / mad
            else:
                logger.warning(f"未知的正規化方法：{method}")
                return data
                
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data
    
    @staticmethod
    def remove_artifacts(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """移除資料中的偽訊"""
        try:
            # 計算z分數
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            
            # 找出偽訊
            artifact_indices = np.where(z_scores > threshold)[0]
            
            # 用插值替換偽訊
            cleaned_data = data.copy()
            for idx in artifact_indices:
                # 簡單線性插值
                if idx > 0 and idx < len(data) - 1:
                    cleaned_data[idx] = (data[idx-1] + data[idx+1]) / 2
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error removing artifacts: {e}")
            return data
    
    @staticmethod
    def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
        """計算基本統計資料"""
        try:
            return {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'rms': float(np.sqrt(np.mean(data**2))),
                'variance': float(np.var(data)),
                'range': float(np.max(data) - np.min(data))
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}


class DataFormatter:
    """資料格式化工具"""
    
    @staticmethod
    def format_timestamp(timestamp: float, format: str = '%Y-%m-%d %H:%M:%S') -> str:
        """格式化時間戳記為字串"""
        try:
            return datetime.fromtimestamp(timestamp).strftime(format)
        except Exception as e:
            logger.error(f"Error formatting timestamp: {e}")
            return str(timestamp)
    
    @staticmethod
    def format_eeg_value(value: float, precision: int = 6) -> str:
        """格式化腦電圖值為字串"""
        try:
            return f"{value:.{precision}f}"
        except Exception as e:
            logger.error(f"Error formatting EEG value: {e}")
            return str(value)
    
    @staticmethod
    def format_data_size(size_bytes: int) -> str:
        """格式化資料大小為人類可讀字串"""
        try:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} TB"
        except Exception as e:
            logger.error(f"Error formatting data size: {e}")
            return f"{size_bytes} bytes"
    
    @staticmethod
    def format_duration(duration_seconds: float) -> str:
        """格式化持續時間為人類可讀字串"""
        try:
            hours, remainder = divmod(int(duration_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except Exception as e:
            logger.error(f"Error formatting duration: {e}")
            return f"{duration_seconds} seconds"