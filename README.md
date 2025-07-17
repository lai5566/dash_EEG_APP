# 進階EEG腦波監控系統

## 專案概述

本專案是一個基於Python的即時腦電圖（EEG）監控系統，採用模組化架構設計，支援ThinkGear協定的腦波裝置。系統提供完整的資料採集、處理、分析、儲存和視覺化功能，並整合了音訊錄製、環境感測器監控和MQTT通訊等進階功能。

## 主要功能

### 核心功能
- **即時腦波資料採集** - 支援ThinkGear協定設備
- **多頻帶信號處理** - Delta、Theta、Alpha、Beta、Gamma頻帶分析
- **認知指標監控** - 注意力、冥想狀態即時追蹤
- **響應式網頁介面** - 基於Dash的互動式儀表板
- **資料庫完整記錄** - SQLite多表結構化儲存
- **音訊同步錄製** - 支援USB麥克風錄音功能

### 進階功能
- **MQTT感測器整合** - 環境資料（溫度、濕度、光線）同步收集
- **自適應性能優化** - 動態調整更新頻率和處理參數
- **並行濾波處理** - 多執行緒頻帶分析提升效能
- **資料匯出功能** - 支援CSV、JSON格式匯出
- **設備自動檢測** - 智慧型音訊設備識別與配置

## 資料來源與計算公式

### 腦波資料處理

#### 原始資料來源
- **ADC數值轉換**: `voltage = raw_value × (1.8V / 4096) / 2000`
- **取樣頻率**: 512 Hz
- **資料範圍**: ±0.001V (合理EEG電壓範圍)

#### 頻帶功率計算
```python
# 使用FFT進行頻譜分析
def compute_power_spectrum(data):
    windowed = data * np.hanning(len(data))
    fft_data = fft(windowed)
    psd = np.abs(fft_data) ** 2
    return freqs, psd

# 各頻帶功率計算
band_power = np.mean(psd[freq_indices])
```

#### 認知指標來源
- **注意力指數**: ThinkGear ASIC晶片直接輸出 (0-100)
- **冥想指數**: ThinkGear ASIC晶片直接輸出 (0-100)
- **信號品質**: 電極接觸品質指標 (0-200, 0為最佳)

#### 眨眼偵測
- **強度計算**: ThinkGear協定眨眼事件強度 (0-255)
- **頻率統計**: 累積眨眼次數與時間軸分析

### 環境感測器資料
- **溫度**: 透過MQTT接收 (°C)
- **濕度**: 透過MQTT接收 (%)
- **光線強度**: 透過MQTT接收 (Lux)

## 資料庫結構

### 資料庫表格詳細說明

#### 1. `raw_data` - 原始腦波資料
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| timestamp | REAL | Unix時間戳記 |
| voltage | REAL | 處理後的電壓值 (V) |

**記錄頻率**: 每秒512筆 (512 Hz取樣)

#### 2. `cognitive_data` - 認知指標資料
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| timestamp | REAL | Unix時間戳記 |
| attention | INTEGER | 注意力指數 (0-100) |
| meditation | INTEGER | 冥想指數 (0-100) |
| signal_quality | INTEGER | 信號品質 (0-200) |

**記錄頻率**: 約每秒1-2筆 (依ThinkGear輸出頻率)

#### 3. `asic_bands_data` - ASIC頻帶功率資料
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| timestamp | REAL | Unix時間戳記 |
| delta | INTEGER | Delta頻帶功率 (0.5-4Hz) |
| theta | INTEGER | Theta頻帶功率 (4-8Hz) |
| low_alpha | INTEGER | 低Alpha頻帶功率 (8-10Hz) |
| high_alpha | INTEGER | 高Alpha頻帶功率 (10-12Hz) |
| low_beta | INTEGER | 低Beta頻帶功率 (12-18Hz) |
| high_beta | INTEGER | 高Beta頻帶功率 (18-30Hz) |
| low_gamma | INTEGER | 低Gamma頻帶功率 (30-40Hz) |
| mid_gamma | INTEGER | 中Gamma頻帶功率 (40-50Hz) |

**記錄頻率**: 約每2-3秒1筆

#### 4. `blink_events` - 眨眼事件記錄
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| timestamp | REAL | Unix時間戳記 |
| intensity | INTEGER | 眨眼強度 (0-255) |

**記錄頻率**: 事件驅動 (偵測到眨眼時記錄)

#### 5. `sensor_data` - 環境感測器資料
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| timestamp | REAL | Unix時間戳記 |
| temperature | REAL | 溫度 (°C) |
| humidity | REAL | 濕度 (%) |
| light | INTEGER | 光線強度 (Lux) |

**記錄頻率**: 每秒1筆 (透過MQTT接收)

#### 6. `unified_records` - 整合記錄表
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| timestamp | REAL | Unix時間戳記 |
| recording_group_id | TEXT | 錄製群組ID |
| attention | INTEGER | 注意力指數 |
| meditation | INTEGER | 冥想指數 |
| signal_quality | INTEGER | 信號品質 |
| temperature | REAL | 溫度 |
| humidity | REAL | 濕度 |
| light | INTEGER | 光線強度 |
| blink_intensity | INTEGER | 眨眼強度 |
| raw_voltage | REAL | 原始電壓值 |
| delta_power | INTEGER | Delta頻帶功率 |
| theta_power | INTEGER | Theta頻帶功率 |
| low_alpha_power | INTEGER | 低Alpha頻帶功率 |
| high_alpha_power | INTEGER | 高Alpha頻帶功率 |
| low_beta_power | INTEGER | 低Beta頻帶功率 |
| high_beta_power | INTEGER | 高Beta頻帶功率 |
| low_gamma_power | INTEGER | 低Gamma頻帶功率 |
| mid_gamma_power | INTEGER | 中Gamma頻帶功率 |

**用途**: 跨資料類型整合分析，特別用於錄音期間的同步資料分析

#### 7. `recording_files` - 錄音檔案記錄
| 欄位名稱 | 資料類型 | 說明 |
|---------|----------|------|
| id | INTEGER PRIMARY KEY | 自動遞增主鍵 |
| recording_group_id | TEXT UNIQUE | 錄製群組ID |
| filename | TEXT | 錄音檔案路徑 |
| start_time | REAL | 錄製開始時間 |
| end_time | REAL | 錄製結束時間 |
| duration | REAL | 錄製持續時間 (秒) |
| sample_rate | INTEGER | 音訊取樣率 (Hz) |
| file_size | INTEGER | 檔案大小 (bytes) |

## 音訊錄製功能詳細說明

### 技術實現

#### 硬體支援
- **相容設備**: USB麥克風、PD100X專業麥克風
- **自動偵測**: 系統啟動時自動掃描並識別最佳音訊設備
- **設備配置**: 自動調整取樣率、聲道數等參數

#### 錄製流程
```python
# 1. 設備偵測與測試
device_index = audio_recorder.list_audio_devices()
test_result = audio_recorder._test_device()

# 2. 錄製啟動
group_id = str(uuid.uuid4())[:8]  # 產生唯一群組ID
audio_recorder.start_recording(group_id)

# 3. 背景錄製執行緒
def record_thread():
    recording = sd.rec(duration * sample_rate, 
                      samplerate=sample_rate, 
                      channels=channels, 
                      dtype='int16')
    
# 4. 檔案儲存
filename = f"recordings/recording_{group_id}_{timestamp}.wav"
wav.write(filename, sample_rate, recording_data)
```

### 資料關聯機制

#### 群組ID關聯系統
錄音功能透過 `recording_group_id` 與其他資料類型建立關聯：

1. **錄音啟動時**: 產生唯一的 `group_id`
2. **資料同步**: 所有在錄音期間產生的資料都標記相同的 `group_id`
3. **整合分析**: 可透過 `group_id` 查詢特定錄音期間的所有相關資料

#### 同步資料類型
- **腦波資料**: `unified_records` 表中的所有EEG相關欄位
- **環境資料**: 溫度、濕度、光線強度
- **認知狀態**: 注意力、冥想指數變化
- **生理反應**: 眨眼頻率與強度
- **音訊檔案**: WAV格式音訊檔案

### 應用場景

#### 1. 認知負荷分析
- **語音任務**: 記錄語音表達時的腦波變化
- **學習評估**: 分析學習過程中的注意力集中程度
- **壓力監控**: 透過語音語調與腦波關聯分析壓力狀態

#### 2. 多模態資料研究
- **腦語介面**: 結合腦波與語音的雙向互動研究
- **情緒識別**: 透過語音情緒與腦波狀態的關聯分析
- **神經反饋**: 即時語音回饋對腦波狀態的影響

#### 3. 長期追蹤研究
- **訓練效果**: 記錄冥想、專注力訓練過程
- **康復追蹤**: 神經康復過程的多維度監控
- **個人化分析**: 建立個人腦波-語音關聯模型

## 快速開始

### 環境需求
```bash
Python 3.8+
pip install -r requirements.txt
```

### 安裝依賴
```bash
# 核心依賴
pip install dash plotly numpy scipy pandas
pip install sqlite3 pyserial psutil sounddevice

# 可選依賴
pip install paho-mqtt  # MQTT支援
pip install librosa    # 進階音訊分析
```

### 啟動系統
```bash
cd d0717
python src/main/python/main.py
```

### 存取介面
- **Web介面**: http://localhost:8052
- **API端點**: http://localhost:8052/api/

## 專案結構

```
EEG_dash_app/
├── src/
│   ├── main/
│   │   ├── python/
│   │   │   ├── core/          # 核心信號處理
│   │   │   ├── services/      # 資料服務
│   │   │   ├── models/        # 資料模型
│   │   │   ├── utils/         # 工具函式
│   │   │   ├── api/           # Web API端點
│   │   │   ├── ui/            # 使用者介面
│   │   │   └── main.py        # 主程式入口
│   │   └── resources/
│   │       ├── config/        # 配置檔案
│   │       └── assets/        # 靜態資源
│   └── test/
│       ├── unit/              # 單元測試
│       └── integration/       # 整合測試
├── docs/                      # 文件
├── output/                    # 輸出檔案
└── recordings/               # 錄音檔案
```

## 資料分析範例

### 認知狀態分析
```sql
-- 查詢錄音期間的注意力變化
SELECT timestamp, attention, meditation 
FROM unified_records 
WHERE recording_group_id = 'abc12345' 
ORDER BY timestamp;
```

### 環境因素關聯
```sql
-- 分析環境因素對注意力的影響
SELECT temperature, humidity, light, attention 
FROM unified_records 
WHERE attention IS NOT NULL 
AND recording_group_id = 'abc12345';
```

### 頻帶功率分析
```sql
-- 獲取特定時間段的頻帶功率分布
SELECT delta_power, theta_power, low_alpha_power, high_alpha_power
FROM unified_records 
WHERE recording_group_id = 'abc12345' 
AND timestamp BETWEEN 1640995200 AND 1640995800;
```

### 資料匯出
```python
# Python資料匯出範例
app = EEGApplication()
app.export_data(start_time=1640995200, 
                end_time=1640995800, 
                format='csv')
```

## 技術架構

### 核心技術
- **信號處理**: SciPy、NumPy
- **Web框架**: Dash、Flask
- **資料庫**: SQLite with WAL模式
- **音訊處理**: SoundDevice、SciPy
- **通訊協定**: ThinkGear、MQTT
- **並行處理**: ThreadPoolExecutor、Multiprocessing

### 關鍵演算法
- **數位濾波**: Butterworth帶通濾波器
- **頻譜分析**: 快速傅立葉變換 (FFT)
- **訊號檢測**: 統計閾值偽訊偵測
- **資料同步**: 時間戳記對齊機制

## 🔧 系統配置

### 主要配置檔案
- `app_config.py`: 應用程式主配置
- `eeg_config.py`: EEG處理參數
- `mqtt_config.py`: MQTT通訊設定
- `audio_config.py`: 音訊錄製配置

### 效能調優
- **自適應更新**: 根據系統負載動態調整更新頻率
- **並行處理**: 多執行緒濾波器提升處理效率
- **記憶體管理**: 循環緩衝區避免記憶體洩漏
- **資料庫優化**: WAL模式提升寫入效能
