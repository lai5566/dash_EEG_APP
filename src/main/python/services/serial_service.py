"""增強型EEG資料序列埠服務"""

import serial
import time
import multiprocessing
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from typing import Dict
from resources.config.eeg_config import (
    SERIAL_PORT, BAUD_RATE, SYNC, POOR_SIGNAL, ATTENTION, MEDITATION, 
    BLINK, RAW_VALUE, ASIC_EEG_POWER
)


class EnhancedSerialReader:
    """增強型序列埠讀取器，支援ThinkGear協定"""

    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.buffer = bytearray()

    def parse_payload(self, payload: bytearray) -> Dict:
        """解析ThinkGear協定資料載荷"""
        data = {}
        i = 0

        while i < len(payload):
            code = payload[i]

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
                raw_val = (payload[i + 1] << 8) | payload[i + 2]
                if raw_val >= 32768:
                    raw_val -= 65536
                data['raw_value'] = raw_val * (1.8 / 4096) / 2000
                i += 3

            elif code == ASIC_EEG_POWER and i + 24 < len(payload):
                # 解析8個頻帶（每僋3位元組）
                bands_data = []
                for j in range(8):
                    band_value = (payload[i + 1 + j * 3] << 16) | (payload[i + 2 + j * 3] << 8) | payload[i + 3 + j * 3]
                    bands_data.append(band_value)
                data['asic_bands'] = bands_data
                print(f"[ASIC DEBUG] SerialService: Parsed ASIC bands: {bands_data}")
                i += 25

            else:
                i += 1

        return data

    def read_data(self, ser: serial.Serial) -> Dict:
        """讀取及解析EEG資料"""
        try:
            available = ser.in_waiting
            if available > 0:
                chunk = ser.read(available)
                self.buffer.extend(chunk)

            # 解析完整的資料包
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

                # 驗證檢查碼
                calc_checksum = (~sum(payload)) & 0xFF
                if calc_checksum == checksum:
                    parsed_data = self.parse_payload(payload)
                    self.buffer = self.buffer[4 + length:]
                    return parsed_data
                else:
                    self.buffer = self.buffer[2:]

        except Exception as e:
            print(f"[EnhancedSerialReader] Error: {e}")

        return {}


def enhanced_serial_worker(out_queue: multiprocessing.Queue):
    """增強型序列埠工作程序"""
    reader = EnhancedSerialReader(SERIAL_PORT, BAUD_RATE)
    ser = None

    while True:
        try:
            if ser is None:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
                ser.reset_input_buffer()
                print(f"[enhanced_serial_worker] Connected to {SERIAL_PORT}")

            parsed_data = reader.read_data(ser)

            if parsed_data:
                timestamp = time.time()
                parsed_data['timestamp'] = timestamp
                out_queue.put(parsed_data)
            else:
                time.sleep(0.01)

        except serial.SerialException as e:
            print(f"[enhanced_serial_worker] SerialException: {e}")
            try:
                ser.close()
            except:
                pass
            ser = None
            time.sleep(2)
        except Exception as e:
            print(f"[enhanced_serial_worker] Unexpected exception: {e}")
            time.sleep(1)


def mock_serial_worker(out_queue: multiprocessing.Queue):
    """測試用模擬序列埠工作程序"""
    while True:
        try:
            # 生成模擬資料
            timestamp = time.time()

            # 模擬原始資料
            raw_value = random.uniform(-0.001, 0.001)
            out_queue.put({'raw_value': raw_value, 'timestamp': timestamp})

            # 隨機認知資料
            if random.random() < 0.1:  # 10% chance
                attention = random.randint(30, 90)
                meditation = random.randint(20, 80)
                signal_quality = random.randint(0, 150)
                out_queue.put({
                    'attention': attention,
                    'meditation': meditation,
                    'signal_quality': signal_quality,
                    'timestamp': timestamp
                })

            # 隨機ASIC資料
            if random.random() < 0.30:  # 30% chance (increased from 5%)
                asic_bands = [random.randint(10, 100) for _ in range(8)]
                print(f"[ASIC DEBUG] MockWorker: Generated ASIC bands: {asic_bands}")
                out_queue.put({'asic_bands': asic_bands, 'timestamp': timestamp})

            # 隨機眨眼事件
            if random.random() < 0.02:  # 2% chance
                blink_intensity = random.randint(50, 200)
                out_queue.put({'blink': blink_intensity, 'timestamp': timestamp})

            time.sleep(0.02)  # 50Hz更新頁率

        except Exception as e:
            print(f"[mock_serial_worker] Error: {e}")
            time.sleep(1)