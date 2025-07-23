#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG設備直接監控與原始封包記錄工具
Direct EEG Device Monitor, Raw Packet Logger & Analyzer

功能總覽：
1. 直接連接 ThinkGear 協議 EEG 設備（如 MindWave Mobile 2），解析封包。
2. 針對每一類資料（raw_voltage、attention、meditation、blink、ASIC 8 頻帶）建立滑動視窗統計（min/max/mean/std/頻率）。
3. **新增**：完整記錄實際接收到的封包長相（SYNC、length、payload、checksum），可選擇：
   - 只在記憶體保存最近 N 筆（deque）
   - 持續寫入 NDJSON 檔 (每行一個 JSON)
4. **新增**：封包大小分佈、各資料欄位出現次數統計。
5. **新增**：一次解析 buffer 內所有完整封包（避免阻塞導致堆積）。
6. **新增**：離線重播模式（--replay FILE），可從 NDJSON/raw 二進位檔重播測試。
7. CLI 參數化：可設定監控時長、raw_log 上限、輸出檔名等。

注意：本程式以 Python 3.8+ 撰寫，需安裝 pyserial。
"""

import os
import sys
import time
import json
import argparse
import statistics
from datetime import datetime
from collections import defaultdict, deque, Counter
from typing import Dict, List, Optional, Any, Iterable

try:
    import serial
except ImportError:
    serial = None  # 允許離線重播模式無需安裝 pyserial

# ===============================
# ThinkGear Protocol 常數定義
# ===============================
SYNC = 0xAA
POOR_SIGNAL = 0x02
ATTENTION = 0x04
MEDITATION = 0x05
BLINK = 0x16
RAW_VALUE = 0x80
ASIC_EEG_POWER = 0x83

# ASIC頻帶名稱
ASIC_BANDS = [
    "Delta", "Theta", "Low-Alpha", "High-Alpha",
    "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma"
]


# ===============================
# 資料統計類別
# ===============================
class EEGDataStats:
    """單一數值序列的滑動視窗統計"""
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.values: deque = deque(maxlen=max_samples)
        self.timestamps: deque = deque(maxlen=max_samples)
        self.count_total = 0  # 累計總數（不會被 maxlen 重置）
        self.start_time: Optional[float] = None

    def add_value(self, value: float, timestamp: float):
        if self.start_time is None:
            self.start_time = timestamp
        self.values.append(value)
        self.timestamps.append(timestamp)
        self.count_total += 1

    def get_stats(self) -> Dict[str, Any]:
        if not self.values:
            return {}
        v = list(self.values)
        t = list(self.timestamps)
        stats = {
            "count": len(v),
            "min": min(v),
            "max": max(v),
            "mean": statistics.mean(v),
            "median": statistics.median(v),
            "std_dev": statistics.stdev(v) if len(v) > 1 else 0.0,
            "count_total": self.count_total,
        }
        if len(t) > 1:
            duration = t[-1] - t[0]
            stats["duration_seconds"] = duration
            stats["frequency_hz"] = (len(t) - 1) / duration if duration > 0 else 0.0
        else:
            stats["duration_seconds"] = 0.0
            stats["frequency_hz"] = 0.0
        return stats


# ===============================
# 主監控器類別
# ===============================
class EEGDirectMonitor:
    def __init__(
        self,
        port: str,
        baud: int = 57600,
        max_samples: int = 1000,
        raw_log_max: int = 5000,
        dump_raw_file: Optional[str] = None,
        replay_file: Optional[str] = None,
    ):
        self.port = port
        self.baud = baud
        self.serial_conn: Optional[serial.Serial] = None
        self.is_running = False
        self.buffer = bytearray()

        # 統計容器
        self.raw_voltage_stats = EEGDataStats(max_samples)
        self.attention_stats = EEGDataStats(max_samples)
        self.meditation_stats = EEGDataStats(max_samples)
        self.signal_quality_stats = EEGDataStats(max_samples)
        self.blink_stats = EEGDataStats(max_samples)
        self.asic_bands_stats: Dict[str, EEGDataStats] = {
            band: EEGDataStats(max_samples) for band in ASIC_BANDS
        }

        # 封包、錯誤統計
        self.packet_count = 0
        self.error_count = 0
        self.start_time: Optional[float] = None

        # 即時顯示
        self.display_buffer: List[str] = []
        self.max_display_lines = 20

        # 原始封包記錄
        self.raw_packet_log = deque(maxlen=raw_log_max)
        self.dump_raw_file = dump_raw_file
        self.packet_size_counter = Counter()
        self.code_block_counter = Counter()

        # 離線重播
        self.replay_file = replay_file
        self._replay_iter: Optional[Iterable[bytes]] = None

    # ---------- 連線 / 重播 ----------
    def connect_device(self) -> bool:
        """連接真實設備或初始化重播來源"""
        if self.replay_file:
            # 離線模式，不需 serial
            if not os.path.exists(self.replay_file):
                print(f"重播檔不存在: {self.replay_file}")
                return False
            self._replay_iter = self._replay_generator(self.replay_file)
            print(f"使用離線重播檔: {self.replay_file}")
            return True

        if serial is None:
            print("未安裝 pyserial，且沒有指定 --replay 檔案，無法連接設備。")
            return False

        try:
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.1)
            self.serial_conn.reset_input_buffer()
            print(f"已連接到EEG設備: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"無法連接設備: {e}")
            return False

    def disconnect_device(self):
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    def _replay_generator(self, filename: str):
        """從 NDJSON 檔（或 raw hex 檔）讀取封包並轉成 bytes 流"""
        # 假設 NDJSON 格式，每行有 raw_bytes_hex 或 payload_hex 等欄位
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    raw_hex = obj.get("raw_bytes_hex") or obj.get("payload_hex")
                    if raw_hex:
                        # 去除空白再轉 bytes
                        bs = bytes.fromhex(raw_hex.replace(" ", ""))
                        yield bs
                except json.JSONDecodeError:
                    # 可能是純 hex 行
                    try:
                        bs = bytes.fromhex(line.replace(" ", ""))
                        yield bs
                    except Exception:
                        pass

    # ---------- 封包解析 ----------
    @staticmethod
    def _calc_checksum(payload: bytes) -> int:
        # ThinkGear: checksum = 0xFF - (sum(payload) & 0xFF)
        return (0xFF - (sum(payload) & 0xFF)) & 0xFF

    def parse_thinkgear_packet(self, payload: bytes) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        i = 0
        L = len(payload)
        while i < L:
            code = payload[i]
            try:
                if code == POOR_SIGNAL and i + 1 < L:
                    data['signal_quality'] = payload[i + 1]
                    i += 2
                elif code == ATTENTION and i + 1 < L:
                    data['attention'] = payload[i + 1]
                    i += 2
                elif code == MEDITATION and i + 1 < L:
                    data['meditation'] = payload[i + 1]
                    i += 2
                elif code == BLINK and i + 1 < L:
                    data['blink'] = payload[i + 1]
                    i += 2
                elif code == RAW_VALUE and i + 2 < L:
                    raw_val = (payload[i + 1] << 8) | payload[i + 2]
                    if raw_val >= 32768:
                        raw_val -= 65536
                    voltage = raw_val * (1.8 / 4096) / 2000  # 依 ThinkGear 標準轉換
                    data['raw_voltage'] = voltage
                    data['raw_value_int'] = raw_val  # 保留未轉換值
                    i += 3
                elif code == ASIC_EEG_POWER and i + 24 < L:
                    bands = []
                    for j in range(8):
                        idx = i + 1 + j * 3
                        band_value = (payload[idx] << 16) | (payload[idx + 1] << 8) | payload[idx + 2]
                        bands.append(band_value)
                    data['asic_bands'] = bands
                    i += 25
                else:
                    # 未知或沒有對應解析的 code，跳過 1byte
                    i += 1
            except IndexError:
                break
        return data

    def _record_raw_packet(self, payload: bytes, full_packet: bytes, length: int,
                           checksum_ok: bool, parsed: Dict[str, Any], ts: float):
        rec = {
            "ts": ts,
            "length": length,
            "checksum_ok": checksum_ok,
            "raw_bytes_hex": full_packet.hex(" "),
            "payload_hex": payload.hex(" "),
            "parsed": parsed,
        }
        self.raw_packet_log.append(rec)
        if self.dump_raw_file:
            try:
                with open(self.dump_raw_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[RAW-LOG] 寫檔失敗: {e}")

    def _process_one_packet(self, payload: bytes, checksum: int) -> Optional[Dict[str, Any]]:
        ts_now = time.time()
        calc_checksum = self._calc_checksum(payload)
        checksum_ok = (calc_checksum == checksum)
        parsed_data: Dict[str, Any] = {}
        if checksum_ok:
            parsed_data = self.parse_thinkgear_packet(payload)
            if parsed_data:
                parsed_data['timestamp'] = ts_now
                self.packet_count += 1
                # 統計欄位次數
                for k in parsed_data.keys():
                    if k not in ("timestamp",):
                        self.code_block_counter[k] += 1
        else:
            self.error_count += 1

        # 記錄原始封包
        full_packet = bytes([SYNC, SYNC, len(payload)]) + payload + bytes([checksum])
        self.packet_size_counter[len(payload)] += 1
        self._record_raw_packet(payload, full_packet, len(payload), checksum_ok, parsed_data, ts_now)
        return parsed_data if checksum_ok else None

    def _read_from_serial(self) -> bytes:
        if self.serial_conn is None:
            return b""
        available = self.serial_conn.in_waiting
        if available > 0:
            return self.serial_conn.read(available)
        return b""

    def _read_from_replay(self) -> bytes:
        if self._replay_iter is None:
            return b""
        try:
            # 模擬串口: 每次吐一些 bytes
            return next(self._replay_iter)
        except StopIteration:
            time.sleep(0.1)
            return b""

    def read_and_parse_all(self) -> List[Dict[str, Any]]:
        """一次讀取並解析 buffer 中所有完整封包，回傳解析後 dict 的 list"""
        parsed_list: List[Dict[str, Any]] = []

        # 讀取來源
        chunk = self._read_from_replay() if self.replay_file else self._read_from_serial()
        if chunk:
            self.buffer.extend(chunk)

        while len(self.buffer) >= 4:
            # 找 SYNC SYNC
            sync_pos = -1
            for i in range(len(self.buffer) - 1):
                if self.buffer[i] == SYNC and self.buffer[i + 1] == SYNC:
                    sync_pos = i
                    break
            if sync_pos == -1:
                # 沒找到同步，直接清掉以免無限膨脹
                self.buffer.clear()
                break

            if sync_pos > 0:
                del self.buffer[:sync_pos]
                if len(self.buffer) < 4:
                    break

            length = self.buffer[2]
            if len(self.buffer) < 4 + length:
                # 等待更多資料
                break

            payload = bytes(self.buffer[3:3 + length])
            checksum = self.buffer[3 + length]

            # 移除已處理部分
            del self.buffer[:4 + length]

            parsed = self._process_one_packet(payload, checksum)
            if parsed:
                parsed_list.append(parsed)

        return parsed_list

    # ---------- 統計更新 / 顯示 ----------
    def update_statistics(self, data: Dict[str, Any]):
        ts = data.get('timestamp', time.time())
        if 'raw_voltage' in data:
            self.raw_voltage_stats.add_value(data['raw_voltage'], ts)
        if 'attention' in data:
            self.attention_stats.add_value(data['attention'], ts)
        if 'meditation' in data:
            self.meditation_stats.add_value(data['meditation'], ts)
        if 'signal_quality' in data:
            self.signal_quality_stats.add_value(data['signal_quality'], ts)
        if 'blink' in data:
            self.blink_stats.add_value(data['blink'], ts)
        if 'asic_bands' in data:
            bands = data['asic_bands']
            for i, v in enumerate(bands):
                if i < len(ASIC_BANDS):
                    self.asic_bands_stats[ASIC_BANDS[i]].add_value(v, ts)

    def format_display_line(self, data: Dict[str, Any]) -> str:
        ts = datetime.fromtimestamp(data.get('timestamp', time.time()))
        tstr = ts.strftime("%H:%M:%S.%f")[:-3]
        parts = [f"[{tstr}]"]
        if 'raw_voltage' in data:
            parts.append(f"電壓:{data['raw_voltage']:+.6f}V")
        if 'attention' in data and 'meditation' in data:
            parts.append(f"專注:{data['attention']} 冥想:{data['meditation']}")
        if 'signal_quality' in data:
            parts.append(f"信號:{data['signal_quality']}")
        if 'blink' in data:
            parts.append(f"眨眼:{data['blink']}")
        if 'asic_bands' in data:
            bands = data['asic_bands'][:4]  # 只顯示前四個
            band_str = " ".join([f"{ASIC_BANDS[i]}:{v}" for i, v in enumerate(bands)])
            parts.append(f"頻帶:[{band_str}]")
        return " | ".join(parts)

    def display_realtime_data(self, data_list: List[Dict[str, Any]]):
        for data in data_list:
            line = self.format_display_line(data)
            self.display_buffer.append(line)
        if len(self.display_buffer) > self.max_display_lines:
            self.display_buffer = self.display_buffer[-self.max_display_lines:]

        # 清螢幕並顯示
        os.system('clear' if os.name == 'posix' else 'cls')
        print("EEG設備即時監控 - 按 Ctrl+C 停止")
        print("=" * 80)
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"運行時間: {elapsed:.1f}s | 封包數: {self.packet_count} | 錯誤: {self.error_count}")
            stats_summary = []
            if self.raw_voltage_stats.count_total > 0:
                freq = self.raw_voltage_stats.get_stats().get('frequency_hz', 0)
                stats_summary.append(f"電壓:{freq:.1f}Hz")
            if self.attention_stats.count_total > 0:
                freq = self.attention_stats.get_stats().get('frequency_hz', 0)
                stats_summary.append(f"認知:{freq:.1f}Hz")
            # ASIC封包頻率估計：用任一 band 的 stats
            first_band = next(iter(self.asic_bands_stats.values()))
            if first_band.count_total > 0:
                freq = first_band.get_stats().get('frequency_hz', 0)
                stats_summary.append(f"ASIC:{freq:.1f}Hz")
            if stats_summary:
                print(f"資料頻率: {' | '.join(stats_summary)}")
        print("-" * 80)
        for line in self.display_buffer:
            print(line)

    # ---------- 報告 ----------
    def generate_analysis_report(self) -> Dict[str, Any]:
        if not self.start_time:
            return {}
        elapsed = time.time() - self.start_time
        report: Dict[str, Any] = {
            "device_info": {
                "port": self.port,
                "baud_rate": self.baud,
                "connection_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "monitoring_duration_seconds": elapsed,
                "total_packets_received": self.packet_count,
                "total_errors": self.error_count,
                "packet_success_rate": (self.packet_count / (self.packet_count + self.error_count) * 100)
                if (self.packet_count + self.error_count) > 0 else 0.0,
            },
            "data_characteristics": {}
        }

        def add_if(stats_obj: EEGDataStats, name: str):
            if stats_obj.count_total > 0:
                report['data_characteristics'][name] = stats_obj.get_stats()

        add_if(self.raw_voltage_stats, 'raw_voltage')
        add_if(self.attention_stats, 'attention')
        add_if(self.meditation_stats, 'meditation')
        add_if(self.signal_quality_stats, 'signal_quality')
        add_if(self.blink_stats, 'blink_events')

        asic_stats = {}
        for band, s in self.asic_bands_stats.items():
            if s.count_total > 0:
                asic_stats[band] = s.get_stats()
        if asic_stats:
            report['data_characteristics']['asic_bands'] = asic_stats

        report['raw_packet_stats'] = {
            "packet_size_hist": dict(self.packet_size_counter),
            "code_block_hist": dict(self.code_block_counter),
            "logged_packets": len(self.raw_packet_log)
        }
        return report

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> Optional[str]:
        if not filename:
            filename = f"eeg_device_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            print(f"報告保存失敗: {e}")
            return None

    def print_summary_report(self, report: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("EEG設備特性分析摘要")
        print("=" * 60)
        di = report.get('device_info', {})
        print(f"設備: {di.get('port')} @ {di.get('baud_rate')}")
        print(f"監控時長: {di.get('monitoring_duration_seconds', 0):.1f} 秒")
        print(f"成功封包: {di.get('total_packets_received', 0)}")
        print(f"錯誤封包: {di.get('total_errors', 0)}")
        print(f"成功率: {di.get('packet_success_rate', 0):.1f}%")

        dc = report.get('data_characteristics', {})
        if 'raw_voltage' in dc:
            rv = dc['raw_voltage']
            print("\n原始電壓:")
            print(f"   頻率: {rv.get('frequency_hz', 0):.1f} Hz")
            print(f"   範圍: {rv.get('min', 0):.6f}V ~ {rv.get('max', 0):.6f}V")
            print(f"   平均: {rv.get('mean', 0):.6f}V")
            print(f"   樣本數(視窗內): {rv.get('count', 0)} / 累計: {rv.get('count_total', 0)}")

        if 'attention' in dc:
            att = dc['attention']
            print("\n注意力:")
            print(f"   頻率: {att.get('frequency_hz', 0):.1f} Hz")
            print(f"   範圍: {att.get('min', 0)} ~ {att.get('max', 0)}")
            print(f"   平均: {att.get('mean', 0):.1f}")

        if 'meditation' in dc:
            med = dc['meditation']
            print("\n冥想度:")
            print(f"   頻率: {med.get('frequency_hz', 0):.1f} Hz")
            print(f"   範圍: {med.get('min', 0)} ~ {med.get('max', 0)}")
            print(f"   平均: {med.get('mean', 0):.1f}")

        if 'asic_bands' in dc:
            print("\nASIC頻帶:")
            asic = dc['asic_bands']
            for band_name, stats in asic.items():
                print(f"   {band_name}: {stats.get('frequency_hz', 0):.1f}Hz, 範圍:{stats.get('min', 0)}-{stats.get('max', 0)}, 平均:{stats.get('mean', 0):.0f}")

        raw_stats = report.get('raw_packet_stats', {})
        print("\n原始封包統計：")
        print(f"   記錄筆數: {raw_stats.get('logged_packets', 0)}")
        print(f"   封包大小分佈: {raw_stats.get('packet_size_hist', {})}")
        print(f"   資料欄位出現次數: {raw_stats.get('code_block_hist', {})}")

    # ---------- 主流程 ----------
    def start_monitoring(self, duration: Optional[int] = None, no_display: bool = False) -> bool:
        if not self.connect_device():
            return False
        self.is_running = True
        self.start_time = time.time()
        if duration:
            print(f"監控時長: {duration} 秒")
        else:
            print("持續監控中，按 Ctrl+C 停止")

        try:
            while self.is_running:
                parsed_packets = self.read_and_parse_all()
                for pkt in parsed_packets:
                    self.update_statistics(pkt)
                if not no_display and parsed_packets:
                    self.display_realtime_data(parsed_packets)

                if duration and (time.time() - self.start_time) >= duration:
                    break

                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n使用者中止監控")
        finally:
            self.is_running = False
            if not self.replay_file:
                self.disconnect_device()

        # 生成報告
        print("\n生成分析報告...")
        report = self.generate_analysis_report()
        if report:
            saved = self.save_report(report)
            if saved:
                print(f"分析報告已保存至: {saved}")
            self.print_summary_report(report)
        return True


# ===============================
# CLI 入口
# ===============================

def parse_args():
    p = argparse.ArgumentParser(description="EEG Device Monitor & Raw Logger")
    p.add_argument("duration", nargs="?", type=int, default=None, help="監控秒數 (預設持續至 Ctrl+C)")
    p.add_argument("--port", default="/dev/tty.usbserial-11410", help="串口路徑")
    p.add_argument("--baud", type=int, default=57600, help="鮑率")
    p.add_argument("--max-samples", type=int, default=1000, help="每個統計序列最大樣本數")
    p.add_argument("--raw-max", type=int, default=5000, help="原始封包記錄的滑動視窗大小")
    p.add_argument("--dump-raw", default=None, help="NDJSON 檔案路徑，啟用後會持續寫入原始封包")
    p.add_argument("--replay", default=None, help="離線重播 NDJSON/HEX 檔案")
    p.add_argument("--no-display", action="store_true", help="不顯示即時畫面（只記錄）")
    return p.parse_args()


def main():
    args = parse_args()
    monitor = EEGDirectMonitor(
        port=args.port,
        baud=args.baud,
        max_samples=args.max_samples,
        raw_log_max=args.raw_max,
        dump_raw_file=args.dump_raw,
        replay_file=args.replay,
    )
    ok = monitor.start_monitoring(duration=args.duration, no_display=args.no_display)
    if ok:
        print("\nEEG設備分析完成!")
    else:
        print("\nEEG設備分析失敗!")


if __name__ == "__main__":
    main()
