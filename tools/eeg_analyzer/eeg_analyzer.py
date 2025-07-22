#!/usr/bin/env python3
"""
EEG資料傳輸分析和模擬工具

用途:
1. 分析EEG設備資料傳輸速率、頻率和資料型態
2. 統計資料範圍和傳輸特性
3. 生成模擬EEG資料用於測試
4. 提供設備使用模式分析報告

Author: EEG Dashboard Project
Version: 1.0.0
"""

import os
import sys
import time
import sqlite3
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import json
import random
import math
import multiprocessing
from dataclasses import dataclass

# 添加專案路徑
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'src', 'main', 'python'))
sys.path.insert(0, os.path.join(project_root, 'src', 'main', 'resources'))

try:
    from config.app_config import APP_CONFIG, DATABASE_PATH, FFT_TEST_DATA_CONFIG
except ImportError:
    # 如果無法導入配置，使用默認值
    APP_CONFIG = {
        'sample_rate': 512,
        'buffer_size': 1024,
        'window_size': 512
    }
    DATABASE_PATH = os.path.join(project_root, 'enhanced_eeg.db')
    FFT_TEST_DATA_CONFIG = {
        'amplitudes': {
            'delta': 0.15, 'theta': 0.12, 'alpha': 0.10,
            'beta': 0.08, 'gamma': 0.05, 'noise': 0.02
        },
        'frequencies': {
            'delta': 2.0, 'theta': 6.0, 'alpha': 10.0,
            'beta': 20.0, 'gamma': 40.0
        }
    }


@dataclass
class DataStats:
    """資料統計資訊"""
    count: int = 0
    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    frequency_per_second: float = 0.0
    data_type: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'count': self.count,
            'min_value': self.min_val,
            'max_value': self.max_val,
            'mean': self.mean,
            'std_deviation': self.std_dev,
            'frequency_per_second': self.frequency_per_second,
            'data_type': self.data_type
        }


class EEGDataAnalyzer:
    """EEG資料傳輸分析器"""
    
    def __init__(self, db_path: str = None):
        """初始化分析器"""
        self.db_path = db_path or DATABASE_PATH
        self.analysis_results = {}
        self.data_stats = defaultdict(lambda: DataStats())
        
    def connect_database(self) -> Optional[sqlite3.Connection]:
        """連接資料庫"""
        try:
            if not os.path.exists(self.db_path):
                print(f"⚠️  資料庫文件不存在: {self.db_path}")
                return None
                
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            print(f"❌ 資料庫連接失敗: {e}")
            return None
    
    def get_table_info(self, conn: sqlite3.Connection) -> Dict[str, List[str]]:
        """獲取資料庫表結構資訊"""
        tables = {}
        try:
            cursor = conn.cursor()
            
            # 獲取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
            
            for table_name in table_names:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                tables[table_name] = columns
                
        except Exception as e:
            print(f"❌ 獲取表結構失敗: {e}")
            
        return tables
    
    def analyze_raw_data_transmission(self, conn: sqlite3.Connection, hours: int = 1) -> Dict:
        """分析原始資料傳輸特性"""
        print(f"🔍 分析最近 {hours} 小時的原始資料傳輸...")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.timestamp()
        
        try:
            cursor = conn.cursor()
            
            # 分析 raw_data_batched 表
            cursor.execute("""
                SELECT COUNT(*) as total_records,
                       MIN(start_timestamp) as start_time,
                       MAX(end_timestamp) as end_time,
                       COUNT(DISTINCT session_id) as unique_sessions
                FROM raw_data_batched 
                WHERE start_timestamp >= ?
            """, (cutoff_timestamp,))
            
            batch_info = cursor.fetchone()
            
            if batch_info['total_records'] == 0:
                print("⚠️  指定時間範圍內沒有原始資料")
                return {}
            
            # 分析電壓資料分布
            cursor.execute("""
                SELECT voltage_data 
                FROM raw_data_batched 
                WHERE start_timestamp >= ? 
                LIMIT 1000
            """, (cutoff_timestamp,))
            
            voltage_values = []
            for row in cursor.fetchall():
                try:
                    voltage_data = json.loads(row['voltage_data'])
                    if isinstance(voltage_data, list):
                        voltage_values.extend(voltage_data)
                except:
                    continue
            
            # 計算統計資訊
            if voltage_values:
                stats = DataStats(
                    count=len(voltage_values),
                    min_val=min(voltage_values),
                    max_val=max(voltage_values),
                    mean=statistics.mean(voltage_values),
                    std_dev=statistics.stdev(voltage_values) if len(voltage_values) > 1 else 0.0,
                    data_type="voltage_raw"
                )
                
                duration = batch_info['end_time'] - batch_info['start_time']
                if duration > 0:
                    stats.frequency_per_second = len(voltage_values) / duration
                
                return {
                    'batch_records': batch_info['total_records'],
                    'unique_sessions': batch_info['unique_sessions'],
                    'duration_seconds': duration,
                    'voltage_stats': stats.to_dict(),
                    'estimated_sample_rate': stats.frequency_per_second
                }
                
        except Exception as e:
            print(f"❌ 原始資料分析失敗: {e}")
            
        return {}
    
    def analyze_unified_records(self, conn: sqlite3.Connection, hours: int = 1) -> Dict:
        """分析統一記錄傳輸特性"""
        print(f"🔍 分析最近 {hours} 小時的統一記錄...")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.timestamp()
        
        try:
            cursor = conn.cursor()
            
            # 獲取統一記錄統計
            cursor.execute("""
                SELECT COUNT(*) as total_records,
                       MIN(timestamp) as earliest_start,
                       MAX(timestamp) as latest_end
                FROM unified_records 
                WHERE timestamp >= ?
            """, (cutoff_timestamp,))
            
            unified_info = cursor.fetchone()
            
            if unified_info['total_records'] == 0:
                print("⚠️  指定時間範圍內沒有統一記錄")
                return {}
            
            # 分析電壓陣列資料 (unified_records 沒有 voltage_data 欄位，改為分析 raw_voltage_avg)
            cursor.execute("""
                SELECT raw_voltage_avg
                FROM unified_records 
                WHERE timestamp >= ? 
                AND raw_voltage_avg IS NOT NULL
                LIMIT 100
            """, (cutoff_timestamp,))
            
            voltage_values = []
            
            for row in cursor.fetchall():
                if row['raw_voltage_avg'] is not None:
                    voltage_values.append(row['raw_voltage_avg'])
            
            # 分析認知資料
            cursor.execute("""
                SELECT attention, meditation, signal_quality
                FROM unified_records 
                WHERE timestamp >= ?
                AND (attention IS NOT NULL OR meditation IS NOT NULL)
            """, (cutoff_timestamp,))
            
            cognitive_data = cursor.fetchall()
            
            result = {
                'total_unified_records': unified_info['total_records'],
                'duration_seconds': unified_info['latest_end'] - unified_info['earliest_start'] if unified_info['latest_end'] and unified_info['earliest_start'] else 0,
                'records_per_second': 0
            }
            
            if result['duration_seconds'] > 0:
                result['records_per_second'] = unified_info['total_records'] / result['duration_seconds']
            
            # 電壓資料統計
            if voltage_values:
                voltage_stats = DataStats(
                    count=len(voltage_values),
                    min_val=min(voltage_values),
                    max_val=max(voltage_values),
                    mean=statistics.mean(voltage_values),
                    std_dev=statistics.stdev(voltage_values) if len(voltage_values) > 1 else 0.0,
                    data_type="voltage_unified"
                )
                result['voltage_stats'] = voltage_stats.to_dict()
                
            # 認知資料統計  
            if cognitive_data:
                attention_vals = [r['attention'] for r in cognitive_data if r['attention'] is not None]
                meditation_vals = [r['meditation'] for r in cognitive_data if r['meditation'] is not None]
                
                if attention_vals:
                    result['attention_stats'] = {
                        'count': len(attention_vals),
                        'min': min(attention_vals),
                        'max': max(attention_vals),
                        'mean': statistics.mean(attention_vals)
                    }
                    
                if meditation_vals:
                    result['meditation_stats'] = {
                        'count': len(meditation_vals),
                        'min': min(meditation_vals),
                        'max': max(meditation_vals),
                        'mean': statistics.mean(meditation_vals)
                    }
            
            return result
            
        except Exception as e:
            print(f"❌ 統一記錄分析失敗: {e}")
            
        return {}
    
    def analyze_asic_data(self, conn: sqlite3.Connection, hours: int = 1) -> Dict:
        """分析ASIC頻帶資料"""
        print(f"🔍 分析最近 {hours} 小時的ASIC頻帶資料...")
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_timestamp = cutoff_time.timestamp()
        
        try:
            cursor = conn.cursor()
            
            # 檢查ASIC資料 (表名是 asic_bands_data)
            cursor.execute("""
                SELECT COUNT(*) as total_records,
                       delta, theta, low_alpha, high_alpha,
                       low_beta, high_beta, low_gamma, mid_gamma
                FROM asic_bands_data 
                WHERE timestamp >= ?
            """, (cutoff_timestamp,))
            
            asic_records = cursor.fetchall()
            
            if not asic_records or asic_records[0]['total_records'] == 0:
                print("⚠️  指定時間範圍內沒有ASIC資料")
                return {}
            
            bands = ['delta', 'theta', 'low_alpha', 'high_alpha', 
                    'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']
            
            band_stats = {}
            for band in bands:
                values = [r[band] for r in asic_records if r[band] is not None]
                if values:
                    band_stats[band] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': statistics.mean(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
                    }
            
            return {
                'total_asic_records': asic_records[0]['total_records'],
                'band_statistics': band_stats,
                'bands_available': list(band_stats.keys())
            }
            
        except Exception as e:
            print(f"❌ ASIC資料分析失敗: {e}")
            
        return {}
    
    def generate_transmission_report(self) -> Dict:
        """生成完整的傳輸特性報告"""
        print("📊 生成EEG資料傳輸分析報告...")
        
        conn = self.connect_database()
        if not conn:
            return {}
            
        try:
            # 獲取資料庫結構
            tables_info = self.get_table_info(conn)
            
            # 分析不同資料類型
            raw_analysis = self.analyze_raw_data_transmission(conn)
            unified_analysis = self.analyze_unified_records(conn)
            asic_analysis = self.analyze_asic_data(conn)
            
            report = {
                'analysis_time': datetime.now().isoformat(),
                'database_path': self.db_path,
                'tables_structure': tables_info,
                'raw_data_analysis': raw_analysis,
                'unified_records_analysis': unified_analysis,
                'asic_data_analysis': asic_analysis,
                'configuration': {
                    'expected_sample_rate': APP_CONFIG['sample_rate'],
                    'buffer_size': APP_CONFIG['buffer_size'],
                    'window_size': APP_CONFIG['window_size']
                }
            }
            
            return report
            
        except Exception as e:
            print(f"❌ 報告生成失敗: {e}")
            return {}
        finally:
            conn.close()


class EEGDataSimulator:
    """EEG資料模擬器"""
    
    def __init__(self):
        """初始化模擬器"""
        self.config = APP_CONFIG
        self.fft_config = FFT_TEST_DATA_CONFIG
        self.sample_rate = self.config['sample_rate']
        
    def generate_realistic_eeg_signal(self, duration_seconds: int = 10) -> List[float]:
        """生成真實的EEG信號"""
        total_samples = int(duration_seconds * self.sample_rate)
        signal = []
        
        # 基於FFT測試配置生成各頻帶
        for i in range(total_samples):
            t = i / self.sample_rate
            sample = 0.0
            
            # 添加各個腦波頻帶
            for wave_name, amplitude in self.fft_config['amplitudes'].items():
                if wave_name in self.fft_config['frequencies']:
                    freq = self.fft_config['frequencies'][wave_name]
                    # 添加隨機相位和振幅變化
                    phase = random.random() * 2 * math.pi
                    amp_variation = 1.0 + 0.2 * (random.random() - 0.5)
                    sample += amplitude * amp_variation * math.sin(2 * math.pi * freq * t + phase)
            
            # 添加隨機雜訊
            noise = self.fft_config['amplitudes']['noise'] * (random.random() - 0.5)
            sample += noise
            
            signal.append(sample)
            
        return signal
    
    def generate_cognitive_data(self, duration_seconds: int = 60) -> List[Dict]:
        """生成認知資料 (attention, meditation)"""
        # 認知資料更新頻率較低，約每秒幾次
        updates_per_second = 5
        total_updates = duration_seconds * updates_per_second
        
        cognitive_data = []
        for i in range(total_updates):
            timestamp = time.time() + i / updates_per_second
            
            # 生成具有趨勢的認知資料
            base_attention = 50 + 30 * math.sin(i * 0.01)  # 緩慢變化
            base_meditation = 40 + 25 * math.cos(i * 0.005)
            
            # 添加隨機變化
            attention = max(0, min(100, int(base_attention + random.gauss(0, 10))))
            meditation = max(0, min(100, int(base_meditation + random.gauss(0, 8))))
            signal_quality = random.randint(0, 200)
            
            cognitive_data.append({
                'timestamp': timestamp,
                'attention': attention,
                'meditation': meditation,
                'signal_quality': signal_quality
            })
            
        return cognitive_data
    
    def generate_asic_bands_data(self, duration_seconds: int = 60) -> List[Dict]:
        """生成ASIC頻帶資料"""
        # ASIC資料更新頻率更低
        updates_per_second = 1
        total_updates = duration_seconds * updates_per_second
        
        band_data = []
        for i in range(total_updates):
            timestamp = time.time() + i / updates_per_second
            
            # 生成8個頻帶的資料，模擬真實腦波特徵
            bands = {
                'delta': random.randint(50000, 200000),      # 低頻高幅度
                'theta': random.randint(30000, 150000),
                'low_alpha': random.randint(20000, 100000),
                'high_alpha': random.randint(15000, 80000),
                'low_beta': random.randint(10000, 60000),
                'high_beta': random.randint(8000, 50000),
                'low_gamma': random.randint(5000, 30000),    # 高頻低幅度
                'mid_gamma': random.randint(3000, 20000)
            }
            
            bands['timestamp'] = timestamp
            band_data.append(bands)
            
        return band_data
    
    def simulate_real_time_transmission(self, duration_seconds: int = 30):
        """模擬即時資料傳輸"""
        print(f"🎭 開始模擬 {duration_seconds} 秒的即時EEG資料傳輸...")
        
        start_time = time.time()
        sample_count = 0
        
        # 生成基礎信號
        eeg_signal = self.generate_realistic_eeg_signal(duration_seconds)
        cognitive_data = self.generate_cognitive_data(duration_seconds)
        asic_data = self.generate_asic_bands_data(duration_seconds)
        
        cognitive_idx = 0
        asic_idx = 0
        
        print("📡 開始傳輸模擬...")
        
        for i, voltage in enumerate(eeg_signal):
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 模擬512Hz的電壓資料
            print(f"📊 Raw voltage: {voltage:.6f}V (樣本 {sample_count+1})")
            sample_count += 1
            
            # 模擬認知資料 (較低頻率)
            if cognitive_idx < len(cognitive_data) and elapsed >= cognitive_idx / 5:
                cog_data = cognitive_data[cognitive_idx]
                print(f"🧠 Cognitive - Attention: {cog_data['attention']}, Meditation: {cog_data['meditation']}")
                cognitive_idx += 1
            
            # 模擬ASIC資料 (最低頻率)
            if asic_idx < len(asic_data) and elapsed >= asic_idx:
                asic = asic_data[asic_idx]
                print(f"🌈 ASIC Bands - Delta: {asic['delta']}, Alpha: {asic['low_alpha']}")
                asic_idx += 1
            
            # 控制傳輸速率
            target_time = start_time + (i + 1) / self.sample_rate
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        print(f"✅ 模擬完成! 總共傳輸 {sample_count} 個電壓樣本")


def main():
    """主函數"""
    print("🎯 EEG資料傳輸分析和模擬工具")
    print("=" * 50)
    
    while True:
        print("\n選擇操作:")
        print("1. 分析現有EEG資料傳輸特性")
        print("2. 生成資料傳輸報告")
        print("3. 模擬即時EEG資料傳輸")
        print("4. 生成模擬資料文件")
        print("5. 退出")
        
        choice = input("\n請輸入選擇 (1-5): ").strip()
        
        if choice == '1':
            analyzer = EEGDataAnalyzer()
            conn = analyzer.connect_database()
            if conn:
                raw_analysis = analyzer.analyze_raw_data_transmission(conn)
                unified_analysis = analyzer.analyze_unified_records(conn)
                print(f"\n📊 原始資料分析: {json.dumps(raw_analysis, indent=2, ensure_ascii=False)}")
                print(f"\n📊 統一記錄分析: {json.dumps(unified_analysis, indent=2, ensure_ascii=False)}")
                conn.close()
        
        elif choice == '2':
            analyzer = EEGDataAnalyzer()
            report = analyzer.generate_transmission_report()
            if report:
                # 保存報告
                report_file = f"eeg_transmission_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                report_path = os.path.join(os.path.dirname(__file__), report_file)
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"✅ 報告已保存至: {report_path}")
        
        elif choice == '3':
            duration = input("輸入模擬持續時間(秒, 默認30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            
            simulator = EEGDataSimulator()
            simulator.simulate_real_time_transmission(duration)
        
        elif choice == '4':
            duration = input("輸入生成資料持續時間(秒, 默認60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            
            simulator = EEGDataSimulator()
            
            # 生成各類模擬資料
            eeg_signal = simulator.generate_realistic_eeg_signal(duration)
            cognitive_data = simulator.generate_cognitive_data(duration)
            asic_data = simulator.generate_asic_bands_data(duration)
            
            # 保存到文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            data_file = f"simulated_eeg_data_{timestamp}.json"
            data_path = os.path.join(os.path.dirname(__file__), data_file)
            
            simulation_data = {
                'metadata': {
                    'generation_time': datetime.now().isoformat(),
                    'duration_seconds': duration,
                    'sample_rate': simulator.sample_rate,
                    'total_voltage_samples': len(eeg_signal)
                },
                'eeg_voltage_signal': eeg_signal[:1000],  # 只保存前1000個樣本避免文件過大
                'cognitive_data': cognitive_data,
                'asic_bands_data': asic_data
            }
            
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(simulation_data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ 模擬資料已保存至: {data_path}")
        
        elif choice == '5':
            print("👋 再見!")
            break
        
        else:
            print("❌ 無效選擇，請重新輸入")


if __name__ == "__main__":
    main()