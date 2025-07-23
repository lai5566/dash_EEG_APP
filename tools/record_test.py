#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PD100X Podcast Microphone 錄音 + 測試工具
功能：
- 自動搜尋 PD100X 裝置（或指定名稱/ID）
- 錄音到 WAV
- 即時 VU meter
- (可選) 終端 ASCII 頻譜
- 自動重連（裝置拔插/中斷）
"""

import argparse
import sys
import time
import queue
import traceback
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.fft import rfft, rfftfreq

AUDIO_Q = queue.Queue()

def int_or_str(x):
    try:
        return int(x)
    except ValueError:
        return x

def list_devices():
    print(sd.query_devices())

def find_device(keyword="PD100X"):
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if keyword.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    return None

def vu_meter(samples, width=40):
    rms = np.sqrt(np.mean(samples**2))
    db = 20 * np.log10(rms + 1e-12)
    db_norm = np.clip((db + 60) / 60, 0, 1)  # -60dB~0dB → 0~1
    bar_len = int(db_norm * width)
    return f"[{'█'*bar_len}{'-'*(width-bar_len)}] {db:6.1f} dB"

def ascii_spectrum(samples, fs, width=50, max_freq=8000):
    # 單聲道 samples
    spec = np.abs(rfft(samples))  # magnitude
    freqs = rfftfreq(len(samples), 1/fs)
    # 限制頻率
    mask = freqs <= max_freq
    spec = spec[mask]; freqs = freqs[mask]
    # 分桶
    buckets = np.array_split(spec, width)
    bars = []
    for b in buckets:
        v = np.mean(b)
        bars.append(v)
    bars = np.array(bars)
    bars = bars / (bars.max() + 1e-9)
    return ''.join('█' if v > 0.75 else
                   '▓' if v > 0.5  else
                   '▒' if v > 0.25 else
                   '░' if v > 0.1  else
                   ' ' for v in bars)

def audio_callback(indata, frames, time_info, status):
    if status:
        # 將 status 印到 stderr
        print(status, file=sys.stderr)
    AUDIO_Q.put(indata.copy())

def record_loop(args):
    # 準備輸出檔
    with sf.SoundFile(args.output, mode='w',
                      samplerate=args.samplerate,
                      channels=args.channels, subtype='PCM_16') as wav:
        start = time.time()
        last_print = 0
        try:
            while True:
                block = AUDIO_Q.get()
                wav.write(block)
                now = time.time()
                if not args.no_vu or args.spectrum:
                    # 顯示第一聲道
                    mono = block[:, 0] if block.ndim > 1 else block
                    parts = []
                    if not args.no_vu:
                        parts.append(vu_meter(mono))
                    if args.spectrum and (now - last_print) >= args.spec_interval:
                        parts.append("\n" + ascii_spectrum(mono, args.samplerate))
                        last_print = now
                    if parts:
                        print("\r" + " ".join(parts), end="", flush=True)
                if args.duration and (now - start) >= args.duration:
                    break
        except KeyboardInterrupt:
            print("\n收到中斷，停止錄音。")
        print("\n錄音結束，寫入檔案中…")

def main():
    p = argparse.ArgumentParser(description="PD100X USB 麥克風錄音測試")
    p.add_argument("-l", "--list", action="store_true", help="列出裝置後退出")
    p.add_argument("-d", "--device", type=int_or_str, default=None,
                   help="裝置 ID 或名稱片段（預設自動抓 PD100X）")
    p.add_argument("-r", "--samplerate", type=int, default=48000)
    p.add_argument("-c", "--channels", type=int, default=1)
    p.add_argument("-t", "--duration", type=float, default=5.0,
                   help="錄音秒數；0/None 代表無限直到 Ctrl+C")
    p.add_argument("-o", "--output", default="pd100x.wav")
    p.add_argument("--no-vu", action="store_true", help="關閉即時音量條")
    p.add_argument("--spectrum", action="store_true", help="顯示 ASCII 頻譜")
    p.add_argument("--spec-interval", type=float, default=0.5,
                   help="頻譜刷新秒數")
    p.add_argument("--retry", type=int, default=5,
                   help="裝置中斷後重試次數（-1=無限）")
    p.add_argument("--retry-wait", type=float, default=1.0,
                   help="重試前等待秒數")
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    # 找裝置
    target = args.device
    if target is None or isinstance(target, str):
        if target is None:
            target = "PD100X"
        dev_id = find_device(target)
    else:
        dev_id = target

    if dev_id is None:
        print(f"找不到輸入裝置：{target}")
        print("用 -l 列出看看。")
        sys.exit(1)

    # 驗證裝置設定
    try:
        sd.check_input_settings(device=dev_id,
                                samplerate=args.samplerate,
                                channels=args.channels)
    except Exception as e:
        print("裝置設定檢查失敗：", e)
        sys.exit(1)

    print(f"使用裝置 ID {dev_id} - {sd.query_devices(dev_id)['name']}")
    print(f"開始錄音 {args.duration if args.duration else '∞'} 秒…(Ctrl+C 結束)")

    retries = 0
    while True:
        try:
            with sd.InputStream(device=dev_id,
                                channels=args.channels,
                                samplerate=args.samplerate,
                                callback=audio_callback):
                record_loop(args)
                break  # 正常結束
        except KeyboardInterrupt:
            print("\n手動中斷，離開。")
            break
        except Exception as e:
            print("\n裝置中斷或錯誤：", e)
            traceback.print_exc(limit=1)
            retries += 1
            if args.retry >= 0 and retries > args.retry:
                print("重試次數用盡，退出。")
                break
            print(f"{args.retry_wait} 秒後重試…(第 {retries} 次)")
            time.sleep(args.retry_wait)

    print("完成。輸出檔案：", args.output)

if __name__ == "__main__":
    main()
