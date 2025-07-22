#!/usr/bin/env python3
"""
EEG設備直接監控和分析工具
Direct EEG Device Monitor and Analysis Tool

直接與EEG設備對接，即時接收並分析資料傳輸特性
用途：
- 了解EEG設備的資料傳輸頻率和模式
- 即時統計各類資料的範圍和特性
- 分析設備的工作模式和性能表現
- 生成設備特性報告

單檔案獨立運行，無需資料庫
"""

import serial
import time
import json
import statistics
import threading
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import os
import sys

# EEG配置常數
SERIAL_PORT = "/dev/tty.usbserial-1410"
BAUD_RATE = 57600
SYNC = 0xaa
POOR_SIGNAL = 0x02
ATTENTION = 0x04
MEDITATION = 0x05
BLINK = 0x16
RAW_VALUE = 0x80
ASIC_EEG_POWER = 0x83

# ASIC頻帶名稱
ASIC_BANDS = ["Delta", "Theta", "Low-Alpha", "High-Alpha",
              "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma"]


class EEGDataStats:
    """EEG資料統計類別"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.values = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)
        self.count = 0
        self.start_time = None
        
    def add_value(self, value: float, timestamp: float = None):
        """添加新數值"""
        if timestamp is None:
            timestamp = time.time()
            
        if self.start_time is None:
            self.start_time = timestamp
            
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.count += 1
        
    def get_stats(self) -> Dict:
        """獲取統計資訊"""
        if not self.values:
            return {}
            
        values_list = list(self.values)
        timestamps_list = list(self.timestamps)
        
        # 基本統計
        stats = {
            'count': len(values_list),
            'min': min(values_list),
            'max': max(values_list),
            'mean': statistics.mean(values_list),
            'median': statistics.median(values_list)
        }
        
        if len(values_list) > 1:
            stats['std_dev'] = statistics.stdev(values_list)
        else:
            stats['std_dev'] = 0.0
            
        # 頻率統計
        if len(timestamps_list) > 1:
            duration = timestamps_list[-1] - timestamps_list[0]
            if duration > 0:
                stats['frequency_hz'] = (len(timestamps_list) - 1) / duration
                stats['duration_seconds'] = duration
            else:
                stats['frequency_hz'] = 0
                stats['duration_seconds'] = 0
        else:
            stats['frequency_hz'] = 0
            stats['duration_seconds'] = 0
            
        return stats


class EEGDirectMonitor:
    """EEG設備直接監控器"""
    
    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self.port = port
        self.baud = baud
        self.serial_conn = None
        self.is_running = False
        self.buffer = bytearray()
        
        # 資料統計容器
        self.raw_voltage_stats = EEGDataStats()
        self.attention_stats = EEGDataStats()
        self.meditation_stats = EEGDataStats()
        self.signal_quality_stats = EEGDataStats()
        self.blink_stats = EEGDataStats()
        self.asic_bands_stats = {band: EEGDataStats() for band in ASIC_BANDS}
        
        # 計數器
        self.packet_count = 0
        self.error_count = 0
        self.start_time = None
        
        # 即時顯示緩衝
        self.display_buffer = []
        self.max_display_lines = 20
        
    def connect_device(self) -> bool:
        """連接EEG設備"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.1)
            self.serial_conn.reset_input_buffer()
            print(f"✅ 已連接到EEG設備: {self.port} @ {self.baud}")
            return True
        except serial.SerialException as e:
            print(f"❌ 無法連接設備: {e}")
            return False
        except Exception as e:
            print(f"❌ 連接錯誤: {e}")
            return False
    
    def disconnect_device(self):
        """斷開設備連接"""
        if self.serial_conn:
            try:
                self.serial_conn.close()
                print("📱 設備已斷開連接")
            except:
                pass
            self.serial_conn = None
    
    def parse_thinkgear_packet(self, payload: bytearray) -> Dict:
        """解析ThinkGear協議封包"""
        data = {}
        i = 0
        
        while i < len(payload):
            code = payload[i]
            
            try:
                if code == POOR_SIGNAL and i + 1 < len(payload):
                    data['signal_quality'] = payload[i + 1]
                    i += 2
                    
                elif code == ATTENTION and i + 1 < len(payload):
                    data['attention'] = payload[i + 1]
                    i += 2
                    
                elif code == MEDITATION and i + 1 < len(payload):
                    data['meditation'] = payload[i + 1]
                    i += 2
                    
                elif code == BLINK and i + 1 < len(payload):
                    data['blink'] = payload[i + 1]
                    i += 2
                    
                elif code == RAW_VALUE and i + 2 < len(payload):
                    # 16位原始電壓值
                    raw_val = (payload[i + 1] << 8) | payload[i + 2]
                    if raw_val >= 32768:
                        raw_val -= 65536
                    # 轉換為電壓 (基於ThinkGear協議)
                    voltage = raw_val * (1.8 / 4096) / 2000
                    data['raw_voltage'] = voltage
                    i += 3
                    
                elif code == ASIC_EEG_POWER and i + 24 < len(payload):
                    # 解析8個頻帶資料 (每個3位元組)
                    bands_data = []
                    for j in range(8):
                        band_value = (payload[i + 1 + j * 3] << 16) | \
                                   (payload[i + 2 + j * 3] << 8) | \
                                   payload[i + 3 + j * 3]
                        bands_data.append(band_value)
                    data['asic_bands'] = bands_data
                    i += 25
                else:
                    i += 1
                    
            except IndexError:
                # 封包不完整，跳過
                break
                
        return data
    
    def read_and_parse_data(self) -> Optional[Dict]:
        """讀取並解析EEG資料"""
        if not self.serial_conn:
            return None
            
        try:
            # 讀取可用資料
            available = self.serial_conn.in_waiting
            if available > 0:
                chunk = self.serial_conn.read(available)
                self.buffer.extend(chunk)
            
            # 解析完整的封包
            while len(self.buffer) >= 4:
                # 尋找同步位元組
                sync_pos = -1
                for i in range(len(self.buffer) - 1):
                    if self.buffer[i] == SYNC and self.buffer[i + 1] == SYNC:
                        sync_pos = i
                        break
                
                if sync_pos == -1:
                    self.buffer.clear()
                    break
                
                # 移除同步之前的資料
                if sync_pos > 0:
                    self.buffer = self.buffer[sync_pos:]
                
                if len(self.buffer) < 4:
                    break
                
                length = self.buffer[2]
                if len(self.buffer) < 4 + length:
                    break
                
                payload = self.buffer[3:3 + length]
                checksum = self.buffer[3 + length]
                
                # 驗證校驗和
                calc_checksum = (~sum(payload)) & 0xFF
                if calc_checksum == checksum:
                    parsed_data = self.parse_thinkgear_packet(payload)
                    if parsed_data:
                        parsed_data['timestamp'] = time.time()
                        parsed_data['packet_size'] = length
                        self.packet_count += 1
                    self.buffer = self.buffer[4 + length:]
                    return parsed_data
                else:
                    self.error_count += 1
                    self.buffer = self.buffer[2:]
            
        except Exception as e:
            self.error_count += 1
            print(f"❌ 資料讀取錯誤: {e}")
            
        return None
    
    def update_statistics(self, data: Dict):
        """更新統計資料"""
        timestamp = data.get('timestamp', time.time())
        
        # 更新各類資料統計
        if 'raw_voltage' in data:
            self.raw_voltage_stats.add_value(data['raw_voltage'], timestamp)
            
        if 'attention' in data:
            self.attention_stats.add_value(data['attention'], timestamp)
            
        if 'meditation' in data:
            self.meditation_stats.add_value(data['meditation'], timestamp)
            
        if 'signal_quality' in data:
            self.signal_quality_stats.add_value(data['signal_quality'], timestamp)
            
        if 'blink' in data:
            self.blink_stats.add_value(data['blink'], timestamp)
            
        if 'asic_bands' in data:
            bands_data = data['asic_bands']
            for i, band_value in enumerate(bands_data):
                if i < len(ASIC_BANDS):
                    band_name = ASIC_BANDS[i]
                    self.asic_bands_stats[band_name].add_value(band_value, timestamp)
    
    def format_display_line(self, data: Dict) -> str:
        """格式化顯示行"""
        timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()))
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
        
        parts = [f"[{time_str}]"]
        
        if 'raw_voltage' in data:
            parts.append(f"電壓:{data['raw_voltage']:+.6f}V")
            
        if 'attention' in data and 'meditation' in data:
            parts.append(f"專注:{data['attention']} 冥想:{data['meditation']}")
            
        if 'signal_quality' in data:
            parts.append(f"信號:{data['signal_quality']}")
            
        if 'blink' in data:
            parts.append(f"眨眼:{data['blink']}")
            
        if 'asic_bands' in data:
            bands = data['asic_bands'][:4]  # 只顯示前4個頻帶
            band_str = " ".join([f"{ASIC_BANDS[i]}:{v}" for i, v in enumerate(bands)])
            parts.append(f"頻帶:[{band_str}]")
            
        return " | ".join(parts)
    
    def display_realtime_data(self, data: Dict):
        """即時顯示資料"""
        display_line = self.format_display_line(data)
        
        # 添加到顯示緩衝
        self.display_buffer.append(display_line)
        if len(self.display_buffer) > self.max_display_lines:
            self.display_buffer.pop(0)
        
        # 清空螢幕並重新顯示
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # 顯示標題
        print("🧠 EEG設備即時監控 - 按 Ctrl+C 停止")
        print("=" * 80)
        
        # 顯示統計摘要
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"📊 運行時間: {elapsed:.1f}s | 封包數: {self.packet_count} | 錯誤: {self.error_count}")
            
            # 顯示各類資料頻率
            stats_summary = []
            if self.raw_voltage_stats.count > 0:
                freq = self.raw_voltage_stats.get_stats().get('frequency_hz', 0)
                stats_summary.append(f"電壓:{freq:.1f}Hz")
            
            if self.attention_stats.count > 0:
                freq = self.attention_stats.get_stats().get('frequency_hz', 0)
                stats_summary.append(f"認知:{freq:.1f}Hz")
                
            if any(stats.count > 0 for stats in self.asic_bands_stats.values()):
                total_asic = sum(stats.count for stats in self.asic_bands_stats.values())
                freq = total_asic / elapsed if elapsed > 0 else 0
                stats_summary.append(f"ASIC:{freq:.1f}Hz")
                
            if stats_summary:
                print(f"📈 資料頻率: {' | '.join(stats_summary)}")
        
        print("-" * 80)
        
        # 顯示即時資料
        for line in self.display_buffer:
            print(line)
    
    def generate_analysis_report(self) -> Dict:
        """生成分析報告"""
        if not self.start_time:
            return {}
            
        elapsed_time = time.time() - self.start_time
        
        report = {
            'device_info': {
                'port': self.port,
                'baud_rate': self.baud,
                'connection_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'monitoring_duration_seconds': elapsed_time,
                'total_packets_received': self.packet_count,
                'total_errors': self.error_count,
                'packet_success_rate': (self.packet_count / (self.packet_count + self.error_count)) * 100 if (self.packet_count + self.error_count) > 0 else 0
            },
            'data_characteristics': {}
        }
        
        # 電壓資料特性
        if self.raw_voltage_stats.count > 0:
            report['data_characteristics']['raw_voltage'] = self.raw_voltage_stats.get_stats()
            
        # 認知資料特性
        if self.attention_stats.count > 0:
            report['data_characteristics']['attention'] = self.attention_stats.get_stats()
            
        if self.meditation_stats.count > 0:
            report['data_characteristics']['meditation'] = self.meditation_stats.get_stats()
            
        if self.signal_quality_stats.count > 0:
            report['data_characteristics']['signal_quality'] = self.signal_quality_stats.get_stats()
        
        # 眨眼事件
        if self.blink_stats.count > 0:
            report['data_characteristics']['blink_events'] = self.blink_stats.get_stats()
        
        # ASIC頻帶資料
        asic_stats = {}
        for band_name, stats in self.asic_bands_stats.items():
            if stats.count > 0:
                asic_stats[band_name] = stats.get_stats()
                
        if asic_stats:
            report['data_characteristics']['asic_bands'] = asic_stats
            
        return report
    
    def save_report(self, report: Dict, filename: str = None):
        """保存報告到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eeg_device_report_{timestamp}.json"
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 分析報告已保存至: {filename}")
            return filename
        except Exception as e:
            print(f"❌ 報告保存失敗: {e}")
            return None
    
    def start_monitoring(self, duration: int = None):
        """開始監控"""
        if not self.connect_device():
            return False
            
        self.is_running = True
        self.start_time = time.time()
        
        print(f"🚀 開始EEG設備監控...")
        if duration:
            print(f"⏰ 監控時長: {duration} 秒")
        else:
            print("⏰ 持續監控，按 Ctrl+C 停止")
        
        try:
            while self.is_running:
                data = self.read_and_parse_data()
                
                if data:
                    self.update_statistics(data)
                    self.display_realtime_data(data)
                    
                # 檢查是否達到指定時長
                if duration and (time.time() - self.start_time) >= duration:
                    break
                    
                time.sleep(0.01)  # 小幅延遲避免CPU過載
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  監控已停止")
            
        finally:
            self.is_running = False
            self.disconnect_device()
            
        # 生成並保存報告
        print(f"\n📊 生成分析報告...")
        report = self.generate_analysis_report()
        
        if report:
            self.save_report(report)
            self.print_summary_report(report)
        
        return True
    
    def print_summary_report(self, report: Dict):
        """打印摘要報告"""
        print(f"\n" + "="*60)
        print(f"📋 EEG設備特性分析摘要")
        print(f"="*60)
        
        device_info = report.get('device_info', {})
        print(f"🔌 設備: {device_info.get('port')} @ {device_info.get('baud_rate')}")
        print(f"⏱️  監控時長: {device_info.get('monitoring_duration_seconds', 0):.1f} 秒")
        print(f"📦 成功封包: {device_info.get('total_packets_received', 0)}")
        print(f"❌ 錯誤封包: {device_info.get('total_errors', 0)}")
        print(f"✅ 成功率: {device_info.get('packet_success_rate', 0):.1f}%")
        
        data_chars = report.get('data_characteristics', {})
        
        if 'raw_voltage' in data_chars:
            rv = data_chars['raw_voltage']
            print(f"\n⚡ 原始電壓:")
            print(f"   頻率: {rv.get('frequency_hz', 0):.1f} Hz")
            print(f"   範圍: {rv.get('min', 0):.6f}V ~ {rv.get('max', 0):.6f}V")
            print(f"   平均: {rv.get('mean', 0):.6f}V")
            print(f"   樣本數: {rv.get('count', 0)}")
            
        if 'attention' in data_chars:
            att = data_chars['attention']
            print(f"\n🧠 注意力:")
            print(f"   頻率: {att.get('frequency_hz', 0):.1f} Hz")
            print(f"   範圍: {att.get('min', 0)} ~ {att.get('max', 0)}")
            print(f"   平均: {att.get('mean', 0):.1f}")
            
        if 'meditation' in data_chars:
            med = data_chars['meditation']
            print(f"\n🧘 冥想度:")
            print(f"   頻率: {med.get('frequency_hz', 0):.1f} Hz")
            print(f"   範圍: {med.get('min', 0)} ~ {med.get('max', 0)}")
            print(f"   平均: {med.get('mean', 0):.1f}")
            
        if 'asic_bands' in data_chars:
            print(f"\n🌈 ASIC頻帶:")
            asic = data_chars['asic_bands']
            for band_name, stats in asic.items():
                print(f"   {band_name}: {stats.get('frequency_hz', 0):.1f}Hz, "
                      f"範圍:{stats.get('min', 0)}-{stats.get('max', 0)}, "
                      f"平均:{stats.get('mean', 0):.0f}")


def main():
    """主函數"""
    print("🧠 EEG設備直接監控分析工具")
    print("=" * 50)
    
    # 檢查參數
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("用法:")
            print("  python eeg_direct_monitor.py [時長(秒)]")
            print("  python eeg_direct_monitor.py --help")
            print("\n範例:")
            print("  python eeg_direct_monitor.py        # 持續監控")
            print("  python eeg_direct_monitor.py 60     # 監控60秒")
            return
    
    # 解析監控時長
    duration = None
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
            print(f"⏰ 設定監控時長: {duration} 秒")
        except ValueError:
            print("❌ 無效的時長參數")
            return
    
    # 創建監控器
    monitor = EEGDirectMonitor()
    
    # 開始監控
    success = monitor.start_monitoring(duration)
    
    if success:
        print("\n🎉 EEG設備分析完成!")
    else:
        print("\n❌ EEG設備分析失敗!")


if __name__ == "__main__":
    main()