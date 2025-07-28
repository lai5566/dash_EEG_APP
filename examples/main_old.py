#!/usr/bin/env python3
import multiprocessing
import threading
import time
import serial
from collections import deque
import queue
import json
import sqlite3

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ----- 全局設定 -----
SERIAL_PORT     = "/dev/tty.usbserial-11410"
BAUD_RATE       = 57600
WINDOW_SIZE     = 512       # 緩衝區長度
UPDATE_INTERVAL = 500      # ms, 更新頻率
FS              = 256       # 取樣率 (Hz)

# ----- 頻帶定義 & SOS 濾波器預計算 -----
bands = {
    "Delta (0.5-4Hz)": (0.5, 4),
    "Theta (4-8Hz)":   (4,   8),
    "Alpha (8-12Hz)":  (8,  12),
    "Beta (12-35Hz)":  (12, 35),
    "Gamma (35-50Hz)": (35, 50),
}
sos_filters = {
    name: butter(4, [low/(0.5*FS), high/(0.5*FS)], btype='band', output='sos')
    for name, (low, high) in bands.items()
}

def serial_worker(out_queue: multiprocessing.Queue):
    """
    串口讀取子進程，利用 in_waiting 避免空讀錯誤，
    斷線自動重試，將 (voltage, timestamp) 丟到 out_queue。
    """
    ser = None
    while True:
        try:
            if ser is None:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
                ser.reset_input_buffer()
                ser.write(b'{"enableRawOutput": true, "format": "Raw"}\r')
                time.sleep(1)
                print("[serial_worker] Connected to", SERIAL_PORT)

            if ser.in_waiting < 2:
                time.sleep(0.005)
                continue
            hdr = ser.read(2)
            if hdr != b'\xAA\xAA':
                continue

            # 讀 length
            while ser.in_waiting < 1:
                time.sleep(0.001)
            length = ser.read(1)[0]

            # 讀 payload + checksum
            while ser.in_waiting < length + 1:
                time.sleep(0.001)
            payload = ser.read(length)
            chksum  = ser.read(1)[0]

            if ((~sum(payload)) & 0xFF) != chksum:
                continue

            i = 0
            while i < length:
                code = payload[i]
                if code == 0x80 and i+2 < length:
                    raw_val = (payload[i+1] << 8) | payload[i+2]
                    if raw_val >= 32768:
                        raw_val -= 65536
                    voltage = raw_val * (1.8 / 4096)/2000
                    ts_now  = time.time()
                    out_queue.put((voltage, ts_now))
                    i += 3
                else:
                    i += 3 if code >= 0x80 else 2

        except serial.SerialException as e:
            print("[serial_worker] SerialException:", e)
            try: ser.close()
            except: pass
            ser = None
            time.sleep(2)
        except Exception as e:
            print("[serial_worker] Unexpected exception:", e)
            time.sleep(1)

def main():
    # --- 啟動串口讀取子進程 ---
    serial_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=serial_worker,
        args=(serial_queue,),
        daemon=True
    )
    p.start()

    # --- 本地緩衝 & 原始資料批次存儲 ---
    data_buffer = deque(maxlen=WINDOW_SIZE)
    time_buffer = deque(maxlen=WINDOW_SIZE)
    raw_storage = []

    # --- 資料庫寫入佇列 & 執行緒 ---
    db_queue = queue.Queue()

    def db_writer():
        conn = sqlite3.connect("eeg_raw_batches.db", check_same_thread=False)
        cur  = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_batches (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts       TEXT    NOT NULL,
                raw_data TEXT    NOT NULL
            )
        """)
        conn.commit()
        while True:
            ts_str, batch = db_queue.get()
            try:
                cur.execute(
                    "INSERT INTO raw_batches (ts, raw_data) VALUES (?, ?);",
                    (ts_str, json.dumps(batch))
                )
                conn.commit()
            except Exception as e:
                print("[db_writer] Error:", e)
            db_queue.task_done()

    threading.Thread(target=db_writer, daemon=True).start()

    # --- 串口資料監聽執行緒 ---
    def serial_listener():
        while True:
            voltage, ts_now = serial_queue.get()
            data_buffer.append(voltage)
            time_buffer.append(ts_now)
            raw_storage.append(voltage)
            if len(raw_storage) >= WINDOW_SIZE:
                batch = raw_storage[:WINDOW_SIZE]
                del raw_storage[:WINDOW_SIZE]
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                db_queue.put((ts_str, batch))

    threading.Thread(target=serial_listener, daemon=True).start()

    # --- Dash App ---
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("即時 EEG 波形"),
        dcc.RadioItems(
            id="mode-selector",
            options=[
                {"label": "帶通濾波",    "value": "bands"},
                {"label": "FFT 頻譜",   "value": "fft"},
                {"label": "FFT 分頻帶", "value": "fft_bands"},
            ],
            value="bands",
            inline=True,
            style={"margin-bottom": "10px"}
        ),
        dcc.Interval(id="interval", interval=UPDATE_INTERVAL, n_intervals=0),
        dcc.Graph(id="eeg-graph"),
        html.Div(id="latency-info", style={"margin-top": "10px", "fontSize": "16px"}),
    ])

    @app.callback(
        [Output("eeg-graph", "figure"),
         Output("latency-info", "children")],
        [Input("interval", "n_intervals"),
         Input("mode-selector", "value")]
    )
    def update_graph(n, mode):
        # 只消費 serial_queue 在 listener 中，這裡不再取 queue
        if not data_buffer:
            return go.Figure(), "Latency: N/A"

        # 準備資料 & 計算延遲
        data    = np.array(data_buffer)
        t       = np.arange(len(data)) / FS
        latency = (time.time() - time_buffer[-1]) * 1000

        # 繪製對應模式的圖表
        if mode == "bands":
            fig = make_subplots(rows=len(bands), cols=1,
                                shared_xaxes=True,
                                subplot_titles=list(bands.keys()))
            for i, name in enumerate(bands.keys(), start=1):
                sos = sos_filters[name]
                y   = sosfiltfilt(sos, data)
                fig.add_trace(
                    go.Scatter(x=t, y=y, mode="lines", showlegend=False),
                    row=i, col=1
                )
            fig.update_layout(height=150*len(bands), showlegend=False)
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Voltage (V)")

        elif mode == "fft":
            fft_vals = np.fft.rfft(data)
            freq     = np.fft.rfftfreq(len(data), 1/FS)
            fig      = go.Figure(
                go.Scatter(x=freq, y=np.abs(fft_vals), mode="lines")
            )
            fig.update_layout(
                title="FFT 頻譜",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude"
            )

        else:  # fft_bands
            fft_vals = np.fft.rfft(data)
            freq     = np.fft.rfftfreq(len(data), 1/FS)
            fig      = make_subplots(rows=len(bands), cols=1,
                                     shared_xaxes=True,
                                     subplot_titles=list(bands.keys()))
            for i, (name, (low, high)) in enumerate(bands.items(), start=1):
                mask = (freq >= low) & (freq <= high)
                y    = np.fft.irfft(fft_vals * mask, n=len(data))
                fig.add_trace(
                    go.Scatter(x=t, y=y, mode="lines", showlegend=False),
                    row=i, col=1
                )
            fig.update_layout(height=150*len(bands), showlegend=False)
            fig.update_xaxes(title_text="Time (s)")
            fig.update_yaxes(title_text="Voltage (V)")

        return fig, f"Latency: {latency:.1f} ms"

    app.run(debug=True, use_reloader=False)

if __name__ == "__main__":
    main()
