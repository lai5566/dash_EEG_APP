"""增強型EEG資料庫服務"""


import sqlite3
import threading
import time
import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from resources.config.app_config import DATABASE_PATH, DATABASE_WRITE_INTERVAL


class TimeUtils:
    """時間處理工具類別"""
    
    GMT8 = timezone(timedelta(hours=8))
    
    @classmethod
    def unix_to_local_time(cls, timestamp: float) -> str:
        """轉換Unix時間戳為GMT+8時間字串"""
        dt = datetime.fromtimestamp(timestamp, tz=cls.GMT8)
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    @classmethod
    def current_time_pair(cls) -> tuple:
        """返回當前時間的Unix時間戳和GMT+8字串"""
        now = time.time()
        return now, cls.unix_to_local_time(now)


class BatchedRawDataProcessor:
    """批次原始資料處理器"""
    
    def __init__(self, batch_duration=1.0, sample_rate=512):
        self.batch_duration = batch_duration
        self.sample_rate = sample_rate
        self.current_batch = []
        self.batch_start_time = None
        self.current_session_id = None
    
    def start_new_batch(self, session_id: str):
        """開始新的批次"""
        self.current_session_id = session_id
        self.current_batch = []
        self.batch_start_time = time.time()
    
    def add_sample(self, voltage: float, timestamp: float):
        """新增單一樣本到當前批次"""
        if self.batch_start_time is None:
            self.batch_start_time = timestamp
        
        self.current_batch.append(voltage)
        
        # 檢查是否需要flush批次
        if timestamp - self.batch_start_time >= self.batch_duration:
            return self.flush_batch()
        return None
    
    def flush_batch(self):
        """flush當前批次並返回批次資料"""
        if not self.current_batch:
            return None
        
        end_timestamp = self.batch_start_time + self.batch_duration
        voltage_array = np.array(self.current_batch)
        
        batch_data = {
            'session_id': self.current_session_id,
            'start_timestamp': self.batch_start_time,
            'start_time_local': TimeUtils.unix_to_local_time(self.batch_start_time),
            'end_timestamp': end_timestamp,
            'end_time_local': TimeUtils.unix_to_local_time(end_timestamp),
            'sample_rate': self.sample_rate,
            'sample_count': len(self.current_batch),
            'voltage_data': json.dumps(self.current_batch),
            'voltage_min': float(voltage_array.min()),
            'voltage_max': float(voltage_array.max()),
            'voltage_avg': float(voltage_array.mean())
        }
        
        self.current_batch = []
        self.batch_start_time = time.time()
        return batch_data


class EnhancedDatabaseWriter:
    """增強型資料庫寫入器，支援受試者管理和批次儲存"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.raw_batch_processor = BatchedRawDataProcessor()
        self.raw_batched_buffer = []
        self.cognitive_buffer = []
        self.asic_buffer = []
        self.blink_buffer = []
        self.sensor_buffer = []
        self.unified_buffer = []
        self.running = True
        self.lock = threading.Lock()
        self.current_session_id = None

    def set_current_session(self, session_id: str):
        """設定當前實驗會話ID"""
        self.current_session_id = session_id
        self.raw_batch_processor.start_new_batch(session_id)

    def setup_database(self):
        """建立增強版資料庫表格"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA cache_size=10000;")
        cur.execute("PRAGMA foreign_keys=ON;")

        # 受試者資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT UNIQUE NOT NULL,
                gender TEXT CHECK(gender IN ('M', 'F', 'Other')),
                age INTEGER CHECK(age >= 0 AND age <= 150),
                created_at REAL NOT NULL,
                created_at_local TEXT NOT NULL,
                notes TEXT,
                researcher_name TEXT,
                created_by TEXT
            )
        """)

        # 環境音效風格表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ambient_sounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sound_name TEXT UNIQUE NOT NULL,
                style_category TEXT NOT NULL,
                filename TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                file_size_bytes INTEGER,
                sample_rate INTEGER,
                bit_depth INTEGER,
                channels INTEGER,
                created_at REAL NOT NULL,
                created_at_local TEXT NOT NULL,
                description TEXT,
                tags TEXT
            )
        """)

        # 實驗會話表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS experiment_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                subject_id TEXT NOT NULL,
                eye_state TEXT CHECK(eye_state IN ('open', 'closed', 'mixed')),
                start_time REAL NOT NULL,
                start_time_local TEXT NOT NULL,
                end_time REAL,
                end_time_local TEXT,
                duration REAL,
                recording_group_id TEXT,
                ambient_sound_id INTEGER,
                researcher_name TEXT,
                notes TEXT,
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
                FOREIGN KEY (ambient_sound_id) REFERENCES ambient_sounds(id)
            )
        """)

        # 批次原始資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_data_batched (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                start_time_local TEXT NOT NULL,
                end_timestamp REAL NOT NULL,
                end_time_local TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                voltage_data TEXT NOT NULL,
                voltage_min REAL,
                voltage_max REAL,
                voltage_avg REAL,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # 認知指標資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                timestamp_local TEXT NOT NULL,
                attention INTEGER,
                meditation INTEGER,
                signal_quality INTEGER,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # ASIC頻帶資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS asic_bands_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                timestamp_local TEXT NOT NULL,
                delta INTEGER, theta INTEGER, low_alpha INTEGER, high_alpha INTEGER,
                low_beta INTEGER, high_beta INTEGER, low_gamma INTEGER, mid_gamma INTEGER,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # 眨眼事件表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS blink_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                timestamp_local TEXT NOT NULL,
                intensity INTEGER,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # 感測器資料表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                timestamp_local TEXT NOT NULL,
                temperature REAL,
                humidity REAL,
                light INTEGER,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # 統一記錄表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS unified_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                timestamp_local TEXT NOT NULL,
                recording_group_id TEXT,
                attention INTEGER,
                meditation INTEGER,
                signal_quality INTEGER,
                temperature REAL,
                humidity REAL,
                light INTEGER,
                blink_intensity INTEGER,
                raw_voltage_avg REAL,
                delta_power INTEGER,
                theta_power INTEGER,
                low_alpha_power INTEGER,
                high_alpha_power INTEGER,
                low_beta_power INTEGER,
                high_beta_power INTEGER,
                low_gamma_power INTEGER,
                mid_gamma_power INTEGER,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # 錄製檔案表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recording_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                recording_group_id TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_time REAL NOT NULL,
                start_time_local TEXT NOT NULL,
                end_time REAL,
                end_time_local TEXT,
                duration REAL,
                sample_rate INTEGER,
                file_size INTEGER,
                FOREIGN KEY (session_id) REFERENCES experiment_sessions(session_id)
            )
        """)

        # 建立索引
        self._create_indexes(cur)

        conn.commit()
        return conn

    def _create_indexes(self, cur):
        """建立數據庫索引來提升查詢效能"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_subjects_subject_id ON subjects(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_subject_id ON experiment_sessions(subject_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON experiment_sessions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_time ON experiment_sessions(start_time, end_time)",
            "CREATE INDEX IF NOT EXISTS idx_raw_session_time ON raw_data_batched(session_id, start_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_cognitive_session_time ON cognitive_data(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_asic_session_time ON asic_bands_data(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_blink_session_time ON blink_events(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_session_time ON sensor_data(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_unified_session_time ON unified_records(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_ambient_category ON ambient_sounds(style_category)",
            "CREATE INDEX IF NOT EXISTS idx_ambient_duration ON ambient_sounds(duration_seconds)"
        ]
        
        for index_sql in indexes:
            cur.execute(index_sql)

    def add_raw_data(self, timestamp: float, voltage: float):
        """新增原始資料（批次處理）"""
        if self.current_session_id is None:
            print("Warning: No current session set for raw data")
            return
            
        batch_data = self.raw_batch_processor.add_sample(voltage, timestamp)
        if batch_data:
            with self.lock:
                self.raw_batched_buffer.append(batch_data)

    def add_cognitive_data(self, timestamp: float, attention: int, meditation: int, signal_quality: int):
        """新增認知資料"""
        if self.current_session_id is None:
            print("Warning: No current session set for cognitive data")
            return
            
        timestamp_local = TimeUtils.unix_to_local_time(timestamp)
        with self.lock:
            self.cognitive_buffer.append((self.current_session_id, timestamp, timestamp_local, attention, meditation, signal_quality))

    def add_asic_data(self, timestamp: float, bands_data: List[int]):
        """新增ASIC資料"""
        if self.current_session_id is None:
            print("Warning: No current session set for ASIC data")
            return
            
        timestamp_local = TimeUtils.unix_to_local_time(timestamp)
        with self.lock:
            self.asic_buffer.append((self.current_session_id, timestamp, timestamp_local, *bands_data))

    def add_blink_data(self, timestamp: float, intensity: int):
        """新增眨眼資料"""
        if self.current_session_id is None:
            print("Warning: No current session set for blink data")
            return
            
        timestamp_local = TimeUtils.unix_to_local_time(timestamp)
        with self.lock:
            self.blink_buffer.append((self.current_session_id, timestamp, timestamp_local, intensity))

    def add_sensor_data(self, timestamp: float, temperature: float, humidity: float, light: int):
        """新增感測器資料"""
        if self.current_session_id is None:
            print("Warning: No current session set for sensor data")
            return
            
        timestamp_local = TimeUtils.unix_to_local_time(timestamp)
        with self.lock:
            self.sensor_buffer.append((self.current_session_id, timestamp, timestamp_local, temperature, humidity, light))

    def add_unified_record(self, timestamp: float, recording_group_id: str = None, **kwargs):
        """新增統一記錄"""
        if self.current_session_id is None:
            print("Warning: No current session set for unified record")
            return
            
        timestamp_local = TimeUtils.unix_to_local_time(timestamp)
        with self.lock:
            record = (
                self.current_session_id,
                timestamp,
                timestamp_local,
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

    # ========================
    # 受試者和會話管理方法
    # ========================
    
    def add_subject(self, subject_id: str, gender: str, age: int, 
                   researcher_name: str = None, notes: str = None) -> bool:
        """新增受試者"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            
            timestamp, timestamp_local = TimeUtils.current_time_pair()
            
            cur.execute("""
                INSERT INTO subjects 
                (subject_id, gender, age, created_at, created_at_local, notes, researcher_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (subject_id, gender, age, timestamp, timestamp_local, notes, researcher_name))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError as e:
            print(f"Subject already exists: {e}")
            return False
        except Exception as e:
            print(f"Error adding subject: {e}")
            return False
    
    def add_ambient_sound(self, sound_name: str, style_category: str, filename: str,
                         file_path: str, duration_seconds: float, description: str = None,
                         tags: list = None, **kwargs) -> Optional[int]:
        """新增環境音效"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            
            timestamp, timestamp_local = TimeUtils.current_time_pair()
            tags_json = json.dumps(tags) if tags else None
            
            cur.execute("""
                INSERT INTO ambient_sounds 
                (sound_name, style_category, filename, file_path, duration_seconds,
                 file_size_bytes, sample_rate, bit_depth, channels, created_at, 
                 created_at_local, description, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (sound_name, style_category, filename, file_path, duration_seconds,
                  kwargs.get('file_size_bytes'), kwargs.get('sample_rate'),
                  kwargs.get('bit_depth'), kwargs.get('channels'),
                  timestamp, timestamp_local, description, tags_json))
            
            sound_id = cur.lastrowid
            conn.commit()
            conn.close()
            return sound_id
            
        except Exception as e:
            print(f"Error adding ambient sound: {e}")
            return None
    
    def start_experiment_session(self, subject_id: str, eye_state: str,
                               ambient_sound_id: int = None, researcher_name: str = None,
                               notes: str = None) -> str:
        """開始實驗會話"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            
            # 生成會話ID
            timestamp, timestamp_local = TimeUtils.current_time_pair()
            session_id = f"{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cur.execute("""
                INSERT INTO experiment_sessions 
                (session_id, subject_id, eye_state, start_time, start_time_local,
                 ambient_sound_id, researcher_name, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, subject_id, eye_state, timestamp, timestamp_local,
                  ambient_sound_id, researcher_name, notes))
            
            conn.commit()
            conn.close()
            
            # 設定為當前會話
            self.set_current_session(session_id)
            
            return session_id
            
        except Exception as e:
            print(f"Error starting experiment session: {e}")
            return None
    
    def end_experiment_session(self, session_id: str = None) -> bool:
        """結束實驗會話"""
        if session_id is None:
            session_id = self.current_session_id
            
        if session_id is None:
            print("No session to end")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            
            timestamp, timestamp_local = TimeUtils.current_time_pair()
            
            # 獲取開始時間來計算持續時間
            cur.execute("SELECT start_time FROM experiment_sessions WHERE session_id = ?", (session_id,))
            result = cur.fetchone()
            
            if result:
                start_time = result[0]
                duration = timestamp - start_time
                
                cur.execute("""
                    UPDATE experiment_sessions 
                    SET end_time = ?, end_time_local = ?, duration = ?
                    WHERE session_id = ?
                """, (timestamp, timestamp_local, duration, session_id))
                
                conn.commit()
                conn.close()
                
                # 清除當前會話
                self.current_session_id = None
                
                return True
            else:
                print(f"Session {session_id} not found")
                return False
                
        except Exception as e:
            print(f"Error ending experiment session: {e}")
            return False
    
    def get_subjects(self) -> List[Dict]:
        """獲取所有受試者"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT subject_id, gender, age, created_at_local, researcher_name, notes
                FROM subjects ORDER BY created_at DESC
            """)
            
            subjects = []
            for row in cur.fetchall():
                subjects.append({
                    'subject_id': row[0],
                    'gender': row[1],
                    'age': row[2],
                    'created_at': row[3],
                    'researcher_name': row[4],
                    'notes': row[5]
                })
            
            conn.close()
            return subjects
            
        except Exception as e:
            print(f"Error getting subjects: {e}")
            return []
    
    def get_ambient_sounds(self) -> List[Dict]:
        """獲取所有環境音效"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT id, sound_name, style_category, filename, duration_seconds, description
                FROM ambient_sounds ORDER BY style_category, sound_name
            """)
            
            sounds = []
            for row in cur.fetchall():
                sounds.append({
                    'id': row[0],
                    'sound_name': row[1],
                    'style_category': row[2],
                    'filename': row[3],
                    'duration_seconds': row[4],
                    'description': row[5]
                })
            
            conn.close()
            return sounds
            
        except Exception as e:
            print(f"Error getting ambient sounds: {e}")
            return []

    def add_recording_file(self, recording_group_id: str, filename: str, 
                          start_time: float, end_time: float = None, 
                          sample_rate: int = None, file_size: int = None):
        """新增錄製檔案記錄"""
        if self.current_session_id is None:
            print("Warning: No current session set for recording file")
            return
            
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        
        duration = (end_time - start_time) if end_time else None
        start_time_local = TimeUtils.unix_to_local_time(start_time)
        end_time_local = TimeUtils.unix_to_local_time(end_time) if end_time else None
        
        cur.execute("""
            INSERT OR REPLACE INTO recording_files 
            (session_id, recording_group_id, filename, file_path, start_time, start_time_local,
             end_time, end_time_local, duration, sample_rate, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.current_session_id, recording_group_id, filename, filename, 
              start_time, start_time_local, end_time, end_time_local, 
              duration, sample_rate, file_size))
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
                    # 寫入批次原始資料
                    if self.raw_batched_buffer:
                        for batch_data in self.raw_batched_buffer:
                            cur.execute("""
                                INSERT INTO raw_data_batched 
                                (session_id, start_timestamp, start_time_local, end_timestamp, 
                                 end_time_local, sample_rate, sample_count, voltage_data,
                                 voltage_min, voltage_max, voltage_avg)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                batch_data['session_id'], batch_data['start_timestamp'],
                                batch_data['start_time_local'], batch_data['end_timestamp'],
                                batch_data['end_time_local'], batch_data['sample_rate'],
                                batch_data['sample_count'], batch_data['voltage_data'],
                                batch_data['voltage_min'], batch_data['voltage_max'],
                                batch_data['voltage_avg']
                            ))
                        self.raw_batched_buffer.clear()

                    # 寫入認知資料
                    if self.cognitive_buffer:
                        cur.executemany(
                            "INSERT INTO cognitive_data (session_id, timestamp, timestamp_local, attention, meditation, signal_quality) VALUES (?, ?, ?, ?, ?, ?)",
                            self.cognitive_buffer
                        )
                        self.cognitive_buffer.clear()

                    # 寫入ASIC資料
                    if self.asic_buffer:
                        cur.executemany(
                            "INSERT INTO asic_bands_data (session_id, timestamp, timestamp_local, delta, theta, low_alpha, high_alpha, low_beta, high_beta, low_gamma, mid_gamma) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            self.asic_buffer
                        )
                        self.asic_buffer.clear()

                    # 寫入眨眼資料
                    if self.blink_buffer:
                        cur.executemany(
                            "INSERT INTO blink_events (session_id, timestamp, timestamp_local, intensity) VALUES (?, ?, ?, ?)",
                            self.blink_buffer
                        )
                        self.blink_buffer.clear()

                    # 寫入感測器資料
                    if self.sensor_buffer:
                        cur.executemany(
                            "INSERT INTO sensor_data (session_id, timestamp, timestamp_local, temperature, humidity, light) VALUES (?, ?, ?, ?, ?, ?)",
                            self.sensor_buffer
                        )
                        self.sensor_buffer.clear()

                    # 寫入統一記錄
                    if self.unified_buffer:
                        cur.executemany("""
                            INSERT INTO unified_records 
                            (session_id, timestamp, timestamp_local, recording_group_id, attention, meditation, signal_quality, 
                             temperature, humidity, light, blink_intensity, raw_voltage_avg,
                             delta_power, theta_power, low_alpha_power, high_alpha_power,
                             low_beta_power, high_beta_power, low_gamma_power, mid_gamma_power)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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