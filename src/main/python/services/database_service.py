"""增強型EEG資料庫服務"""

import sqlite3
import threading
import time
from typing import List, Dict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from resources.config.app_config import DATABASE_PATH, DATABASE_WRITE_INTERVAL


class EnhancedDatabaseWriter:
    """增強型資料庫寫入器，支援所有EEG資料類型"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.raw_buffer = []
        self.cognitive_buffer = []
        self.asic_buffer = []
        self.blink_buffer = []
        self.sensor_buffer = []
        self.unified_buffer = []
        self.running = True
        self.lock = threading.Lock()

    def setup_database(self):
        """建立資料庫表格"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA cache_size=10000;")

        # 原始ADC資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                voltage REAL NOT NULL
            )
        """)

        # 認知指標資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                attention INTEGER,
                meditation INTEGER,
                signal_quality INTEGER
            )
        """)

        # ASIC頻帶資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS asic_bands_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                delta INTEGER, theta INTEGER, low_alpha INTEGER, high_alpha INTEGER,
                low_beta INTEGER, high_beta INTEGER, low_gamma INTEGER, mid_gamma INTEGER
            )
        """)

        # 眨眼事件表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS blink_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                intensity INTEGER
            )
        """)

        # 感測器資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                temperature REAL,
                humidity REAL,
                light INTEGER
            )
        """)

        # 統一記錄表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS unified_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                recording_group_id TEXT,
                attention INTEGER,
                meditation INTEGER,
                signal_quality INTEGER,
                temperature REAL,
                humidity REAL,
                light INTEGER,
                blink_intensity INTEGER,
                raw_voltage REAL,
                delta_power INTEGER,
                theta_power INTEGER,
                low_alpha_power INTEGER,
                high_alpha_power INTEGER,
                low_beta_power INTEGER,
                high_beta_power INTEGER,
                low_gamma_power INTEGER,
                mid_gamma_power INTEGER
            )
        """)

        # 錄製檔案表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recording_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_group_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration REAL,
                sample_rate INTEGER,
                file_size INTEGER
            )
        """)

        conn.commit()
        return conn

    def add_raw_data(self, timestamp: float, voltage: float):
        """新增原始資料"""
        with self.lock:
            self.raw_buffer.append((timestamp, voltage))

    def add_cognitive_data(self, timestamp: float, attention: int, meditation: int, signal_quality: int):
        """新增認知資料"""
        with self.lock:
            self.cognitive_buffer.append((timestamp, attention, meditation, signal_quality))

    def add_asic_data(self, timestamp: float, bands_data: List[int]):
        """新增ASIC資料"""
        with self.lock:
            self.asic_buffer.append((timestamp, *bands_data))

    def add_blink_data(self, timestamp: float, intensity: int):
        """新增眨眼資料"""
        with self.lock:
            self.blink_buffer.append((timestamp, intensity))

    def add_sensor_data(self, timestamp: float, temperature: float, humidity: float, light: int):
        """新增感測器資料"""
        with self.lock:
            self.sensor_buffer.append((timestamp, temperature, humidity, light))

    def add_unified_record(self, timestamp: float, recording_group_id: str = None, **kwargs):
        """新增統一記錄"""
        with self.lock:
            record = (
                timestamp,
                recording_group_id,
                kwargs.get('attention'),
                kwargs.get('meditation'),
                kwargs.get('signal_quality'),
                kwargs.get('temperature'),
                kwargs.get('humidity'),
                kwargs.get('light'),
                kwargs.get('blink_intensity'),
                kwargs.get('raw_voltage'),
                kwargs.get('delta_power'),
                kwargs.get('theta_power'),
                kwargs.get('low_alpha_power'),
                kwargs.get('high_alpha_power'),
                kwargs.get('low_beta_power'),
                kwargs.get('high_beta_power'),
                kwargs.get('low_gamma_power'),
                kwargs.get('mid_gamma_power')
            )
            self.unified_buffer.append(record)

    def add_recording_file(self, recording_group_id: str, filename: str, start_time: float,
                           end_time: float = None, sample_rate: int = None, file_size: int = None):
        """新增錄製檔案記錄"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        duration = (end_time - start_time) if end_time else None
        cur.execute("""
            INSERT OR REPLACE INTO recording_files 
            (recording_group_id, filename, start_time, end_time, duration, sample_rate, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (recording_group_id, filename, start_time, end_time, duration, sample_rate, file_size))
        conn.commit()
        conn.close()

    def writer_thread(self):
        """資料庫寫入執行緒"""
        conn = self.setup_database()
        cur = conn.cursor()

        while self.running:
            try:
                time.sleep(DATABASE_WRITE_INTERVAL)

                with self.lock:
                    # 寫入原始資料
                    if self.raw_buffer:
                        cur.executemany(
                            "INSERT INTO raw_data (timestamp, voltage) VALUES (?, ?)",
                            self.raw_buffer
                        )
                        self.raw_buffer.clear()

                    # 寫入認知資料
                    if self.cognitive_buffer:
                        cur.executemany(
                            "INSERT INTO cognitive_data (timestamp, attention, meditation, signal_quality) VALUES (?, ?, ?, ?)",
                            self.cognitive_buffer
                        )
                        self.cognitive_buffer.clear()

                    # 寫入ASIC資料
                    if self.asic_buffer:
                        cur.executemany(
                            "INSERT INTO asic_bands_data (timestamp, delta, theta, low_alpha, high_alpha, low_beta, high_beta, low_gamma, mid_gamma) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            self.asic_buffer
                        )
                        self.asic_buffer.clear()

                    # 寫入眨眼資料
                    if self.blink_buffer:
                        cur.executemany(
                            "INSERT INTO blink_events (timestamp, intensity) VALUES (?, ?)",
                            self.blink_buffer
                        )
                        self.blink_buffer.clear()

                    # 寫入感測器資料
                    if self.sensor_buffer:
                        cur.executemany(
                            "INSERT INTO sensor_data (timestamp, temperature, humidity, light) VALUES (?, ?, ?, ?)",
                            self.sensor_buffer
                        )
                        self.sensor_buffer.clear()

                    # 寫入統一記錄
                    if self.unified_buffer:
                        cur.executemany("""
                            INSERT INTO unified_records 
                            (timestamp, recording_group_id, attention, meditation, signal_quality, 
                             temperature, humidity, light, blink_intensity, raw_voltage,
                             delta_power, theta_power, low_alpha_power, high_alpha_power,
                             low_beta_power, high_beta_power, low_gamma_power, mid_gamma_power)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, self.unified_buffer)
                        self.unified_buffer.clear()

                conn.commit()

            except Exception as e:
                print(f"[EnhancedDatabaseWriter] Error: {e}")

        conn.close()

    def start(self):
        """啟動寫入執行緒"""
        thread = threading.Thread(target=self.writer_thread, daemon=True)
        thread.start()
        return thread