"""EEG儀表板的Dash網頁介面"""

import time
import uuid
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from datetime import datetime
from typing import Dict, Any, Optional

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import psutil

from core.eeg_processor import RealTimeEEGProcessor
from models.data_buffer import EnhancedCircularBuffer
from services.database_service import EnhancedDatabaseWriter
from services.mqtt_client import MQTTSensorClient
from services.audio_recorder import AudioRecorder
from utils.data_utils import DataValidator, DataProcessor
from resources.config.app_config import UI_CONFIG, PROCESSING_CONFIG, API_CONFIG

logger = logging.getLogger(__name__)


class EEGDashboardApp:
    """EEG監控的主要Dash應用程式"""
    
    def __init__(self, data_buffer: EnhancedCircularBuffer, 
                 db_writer: EnhancedDatabaseWriter,
                 processor: RealTimeEEGProcessor,
                 mqtt_client: Optional[MQTTSensorClient] = None,
                 audio_recorder: Optional[AudioRecorder] = None):
        
        self.data_buffer = data_buffer
        self.db_writer = db_writer
        self.processor = processor
        self.mqtt_client = mqtt_client
        self.audio_recorder = audio_recorder
        
        # 初始化Dash應用程式
        self.app = dash.Dash(__name__)
        
        # 效能監控
        self.performance_monitor = {
            'last_update_time': time.time(),
            'update_count': 0,
            'avg_render_time': 0,
            'adaptive_interval': UI_CONFIG['update_interval']
        }
        
        # EEG頻帶視覺化設定
        self.bands = {
            "Delta (0.5-4Hz)": (0.5, 4),
            "Theta (4-8Hz)": (4, 8),
            "Alpha (8-12Hz)": (8, 12),
            "Beta (12-35Hz)": (12, 35),
            "Gamma (35-50Hz)": (35, 50),
        }
        
        # 頻帶顏色
        self.band_colors = {
            "Delta (0.5-4Hz)": "#FF6B6B",
            "Theta (4-8Hz)": "#4ECDC4",
            "Alpha (8-12Hz)": "#45B7D1",
            "Beta (12-35Hz)": "#96CEB4",
            "Gamma (35-50Hz)": "#FFEAA7",
        }
        
        # ASIC頻帶名稱
        self.asic_bands = ["Delta", "Theta", "Low-Alpha", "High-Alpha",
                          "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma"]
        
        # 設定版面配置和回呼函式
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """設定主要版面配置"""
        self.app.layout = html.Div([
            html.Div([
                # 標題
                html.H1(UI_CONFIG['title'],
                        style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333'}),
                
                # 第一行：FFT頻帶分析
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("FFT頻帶分析",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            dcc.Graph(id="fft-bands-main", 
                                     style={'height': f'{UI_CONFIG["chart_height"]}px'},
                                     config={'displayModeBar': False}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
                
                # 第二行：認知指標
                html.Div([
                    # 左側：趨勢圖表
                    html.Div([
                        html.Div([
                            html.H3("認知指標趨勢",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            dcc.Graph(id="cognitive-trends", style={'height': '250px'},
                                     config={'displayModeBar': False}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
                    
                    # 右側：儀表
                    html.Div([
                        html.Div([
                            html.H3("即時數值",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            html.Div([
                                dcc.Graph(id="attention-gauge", style={'height': '120px'},
                                         config={'displayModeBar': False}),
                                dcc.Graph(id="meditation-gauge", style={'height': '120px'},
                                         config={'displayModeBar': False}),
                            ]),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
                
                # 第三行：眨眼檢測
                html.Div([
                    # 左側：事件時間軸
                    html.Div([
                        html.Div([
                            html.H3("眨眼事件時間軸",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            dcc.Graph(id="blink-timeline", style={'height': '200px'},
                                     config={'displayModeBar': False}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
                    
                    # 右側：眨眼計數
                    html.Div([
                        html.Div([
                            html.H3("眨眼計數",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            dcc.Graph(id="blink-count-chart", style={'height': '200px'},
                                     config={'displayModeBar': False}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
                
                # 第四行：ASIC頻帶
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("ASIC頻帶分析",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            dcc.Graph(id="asic-bands-chart", style={'height': '300px'},
                                     config={'displayModeBar': False}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
                
                # 第五行：感測器和錄音
                html.Div([
                    # 左側：感測器數據
                    html.Div([
                        html.Div([
                            html.H3("環境感測器",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            html.Div(id="sensor-display",
                                     style={'fontSize': '12px', 'lineHeight': '1.5', 
                                            'fontFamily': 'monospace'}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
                    
                    # 右側：錄音控制
                    html.Div([
                        html.Div([
                            html.H3("錄音控制",
                                    style={'fontSize': '18px', 'fontWeight': 'bold', 
                                           'marginBottom': '10px', 'color': '#555'}),
                            html.Div([
                                html.Button("🎙️ 開始錄音", id="start-recording-btn",
                                           style={'marginRight': '10px', 'padding': '10px 20px', 
                                                  'fontSize': '14px'}),
                                html.Button("⏹️ 停止錄音", id="stop-recording-btn",
                                           style={'padding': '10px 20px', 'fontSize': '14px'}),
                            ], style={'marginBottom': '10px'}),
                            html.Div(id="recording-status",
                                     style={'fontSize': '12px', 'color': '#666'}),
                        ], style={'background': 'white', 'borderRadius': '8px', 
                                 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                 'padding': '15px', 'marginBottom': '15px'}),
                    ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
                
                # 狀態列
                html.Div([
                    html.Div(id="performance-status",
                             style={'fontSize': '12px', 'color': '#666', 
                                    'textAlign': 'center', 'padding': '10px',
                                    'borderTop': '1px solid #eee'}),
                ]),
                
                # 間隔組件
                dcc.Interval(id="interval", 
                           interval=UI_CONFIG['update_interval'], 
                           n_intervals=0),
                dcc.Store(id="performance-store", data={}),
                
            ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '10px'}),
        ])
        
    def _setup_callbacks(self):
        """設定所有儀表板回呼函式"""
        
        @self.app.callback(
            Output("fft-bands-main", "figure"),
            Input("interval", "n_intervals")
        )
        def update_fft_bands_main(n):
            """更新FFT頻帶視覺化 (折線圖)"""
            start_time = time.time()
            
            try:
                # 取得目前視窗並進行處理
                processed_result = self.processor.process_current_window()
                
                if not processed_result:
                    return go.Figure().add_annotation(
                        text="EEG處理器錯誤<br>正在初始化...",
                        showarrow=False, x=0.5, y=0.5, 
                        xref="paper", yref="paper",
                        font=dict(size=16, color="red")
                    )
                
                if 'fft_bands' not in processed_result:
                    return go.Figure().add_annotation(
                        text="FFT頻段數據缺失<br>正在生成測試數據...",
                        showarrow=False, x=0.5, y=0.5, 
                        xref="paper", yref="paper",
                        font=dict(size=16, color="orange")
                    )
                
                fft_bands = processed_result['fft_bands']
                
                # 驗證 FFT 頻段數據
                if not fft_bands or all(len(band_data) == 0 for band_data in fft_bands.values()):
                    return go.Figure().add_annotation(
                        text="FFT頻段為空<br>正在重新生成數據...",
                        showarrow=False, x=0.5, y=0.5, 
                        xref="paper", yref="paper",
                        font=dict(size=16, color="orange")
                    )
                
                # 建立多個子圖的折線圖
                band_names = list(self.bands.keys())
                fig = make_subplots(
                    rows=len(band_names), 
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=band_names,
                    vertical_spacing=0.05
                )
                
                # 計算時間軸
                if len(fft_bands) > 0:
                    # 找到第一個非空的頻段來計算時間軸
                    sample_length = 0
                    for band_data in fft_bands.values():
                        if len(band_data) > 0:
                            sample_length = len(band_data)
                            break
                    
                    if sample_length > 0:
                        t = np.arange(sample_length) / self.processor.sample_rate
                        
                        for i, (band_name, band_color) in enumerate(zip(band_names, self.band_colors.values()), start=1):
                            band_key = band_name.split(' ')[0].lower()
                            band_signal = fft_bands.get(band_key, np.array([]))
                            
                            if len(band_signal) > 0:
                                fig.add_trace(
                                    go.Scatter(
                                        x=t, 
                                        y=band_signal, 
                                        mode="lines",
                                        line=dict(color=band_color, width=1.5),
                                        showlegend=False
                                    ),
                                    row=i, col=1
                                )
                            else:
                                # 如果頻段數據為空，顯示零線
                                fig.add_trace(
                                    go.Scatter(
                                        x=t, 
                                        y=np.zeros(len(t)), 
                                        mode="lines",
                                        line=dict(color="gray", width=1, dash="dash"),
                                        showlegend=False
                                    ),
                                    row=i, col=1
                                )
                
                fig.update_layout(
                    title="FFT頻帶分析 (時域波形)",
                    height=UI_CONFIG['chart_height'],
                    margin=dict(l=40, r=15, t=40, b=60),
                    plot_bgcolor='white',
                    showlegend=False
                )
                
                # 更新x軸標籤
                fig.update_xaxes(title_text="時間 (秒)", row=len(band_names), col=1)
                fig.update_yaxes(title_text="振幅")
                
                # 更新效能監控器
                render_time = time.time() - start_time
                self.performance_monitor['avg_render_time'] = (
                    self.performance_monitor['avg_render_time'] * 0.9 + render_time * 0.1
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error in update_fft_bands_main: {e}")
                return go.Figure().add_annotation(
                    text=f"頻帶分析錯誤: {str(e)}",
                    showarrow=False, x=0.5, y=0.5, 
                    xref="paper", yref="paper"
                )
        
        @self.app.callback(
            [Output("attention-gauge", "figure"),
             Output("meditation-gauge", "figure")],
            Input("interval", "n_intervals")
        )
        def update_cognitive_gauges(n):
            """更新認知指標儀表"""
            try:
                cognitive_data = self.data_buffer.get_cognitive_data()
                attention = cognitive_data['attention']
                meditation = cognitive_data['meditation']
                
                def create_gauge(value, title, color):
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': title, 'font': {'size': 12}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 0},
                            'bar': {'color': color, 'thickness': 0.4},
                            'bgcolor': "white",
                            'borderwidth': 1,
                            'bordercolor': "lightgray"
                        }
                    ))
                    fig.update_layout(
                        height=120,
                        margin=dict(l=5, r=5, t=15, b=5),
                        font={'size': 9}
                    )
                    return fig
                
                attention_fig = create_gauge(attention, "注意力", "#1f77b4")
                meditation_fig = create_gauge(meditation, "冥想", "#2ca02c")
                
                return attention_fig, meditation_fig
                
            except Exception as e:
                logger.error(f"Error in update_cognitive_gauges: {e}")
                empty_fig = go.Figure()
                empty_fig.update_layout(height=120)
                return empty_fig, empty_fig
        
        @self.app.callback(
            Output("cognitive-trends", "figure"),
            Input("interval", "n_intervals")
        )
        def update_cognitive_trends(n):
            """更新認知趨勢圖表"""
            try:
                cognitive_data = self.data_buffer.get_cognitive_data()
                
                fig = go.Figure()
                
                max_points = UI_CONFIG['max_points']
                
                # 注意力趨勢
                if cognitive_data['attention_history']:
                    history = list(cognitive_data['attention_history'])[-max_points:]
                    if history:
                        times, values = zip(*history)
                        base_time = times[0] if times else 0
                        rel_times = [(t - base_time) for t in times]
                        fig.add_trace(go.Scatter(
                            x=rel_times, y=values,
                            mode='lines',
                            name='注意力',
                            line=dict(color='#1f77b4', width=2)
                        ))
                
                # 冥想趨勢
                if cognitive_data['meditation_history']:
                    history = list(cognitive_data['meditation_history'])[-max_points:]
                    if history:
                        times, values = zip(*history)
                        base_time = times[0] if times else 0
                        rel_times = [(t - base_time) for t in times]
                        fig.add_trace(go.Scatter(
                            x=rel_times, y=values,
                            mode='lines',
                            name='冥想',
                            line=dict(color='#2ca02c', width=2)
                        ))
                
                fig.update_layout(
                    xaxis_title="時間 (秒)",
                    yaxis_title="數值",
                    yaxis_range=[0, 100],
                    height=250,
                    margin=dict(l=30, r=15, t=15, b=30),
                    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
                    plot_bgcolor='white'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error in update_cognitive_trends: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("blink-timeline", "figure"),
            Input("interval", "n_intervals")
        )
        def update_blink_timeline(n):
            """更新眨眼時間軸"""
            try:
                blink_data = self.data_buffer.get_blink_data()
                events = list(blink_data['events'])[-10:]  # 最後10個事件
                
                fig = go.Figure()
                
                if events:
                    times, intensities = zip(*events)
                    base_time = times[0] if times else 0
                    rel_times = [(t - base_time) for t in times]
                    
                    fig.add_trace(go.Scatter(
                        x=rel_times, y=intensities,
                        mode='markers',
                        marker=dict(size=8, color='red', opacity=0.7),
                        name='眨眼事件'
                    ))
                
                fig.update_layout(
                    xaxis_title="時間 (秒)",
                    yaxis_title="強度",
                    height=200,
                    margin=dict(l=30, r=15, t=15, b=30),
                    plot_bgcolor='white'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error in update_blink_timeline: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("blink-count-chart", "figure"),
            Input("interval", "n_intervals")
        )
        def update_blink_count_chart(n):
            """更新眨眼計數圖表"""
            try:
                blink_data = self.data_buffer.get_blink_data()
                count_history = blink_data['count_history']
                
                fig = go.Figure()
                
                if count_history:
                    times, counts = zip(*count_history)
                    base_time = times[0] if times else 0
                    rel_times = [(t - base_time) for t in times]
                    
                    fig.add_trace(go.Scatter(
                        x=rel_times, y=counts,
                        mode='lines+markers',
                        name='累計次數',
                        line=dict(color='#9467bd', width=2),
                        marker=dict(size=4)
                    ))
                
                fig.update_layout(
                    xaxis_title="時間 (秒)",
                    yaxis_title="次數",
                    height=200,
                    margin=dict(l=40, r=20, t=20, b=40),
                    plot_bgcolor='white'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error in update_blink_count_chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("asic-bands-chart", "figure"),
            Input("interval", "n_intervals")
        )
        def update_asic_bands_chart(n):
            """更新ASIC頻帶圖表"""
            try:
                asic_data = self.data_buffer.get_asic_data()
                current_bands = asic_data['current_bands']
                print(f"[ASIC DEBUG] DashApp: Retrieved ASIC bands for display: {current_bands}")
                
                fig = go.Figure()
                
                if all(band == 0 for band in current_bands):
                    # 沒有ASIC數據
                    print(f"[ASIC DEBUG] DashApp: No ASIC data - all bands are zero")
                    fig.add_annotation(
                        text="沒收到ASIC數據<br><br>可能原因:<br>• ThinkGear設備未連接<br>• 串口設定錯誤<br>• 電極接觸不良",
                        showarrow=False, x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        font=dict(size=14, color="red"),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="red", borderwidth=2
                    )
                else:
                    # 顯示ASIC數據
                    print(f"[ASIC DEBUG] DashApp: Displaying ASIC chart with data: {current_bands}")
                    fig.add_trace(go.Bar(
                        x=self.asic_bands,
                        y=current_bands,
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                                     '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
                        text=[f'{v}' if v > 0 else '0' for v in current_bands],
                        textposition='auto',
                        name="ASIC頻帶功率"
                    ))
                
                fig.update_layout(
                    title="ASIC EEG 8頻帶功率分布",
                    xaxis_title="頻帶",
                    yaxis_title="功率值",
                    yaxis_range=[0, 100],
                    height=300,
                    margin=dict(l=30, r=15, t=30, b=30),
                    plot_bgcolor='white',
                    showlegend=False
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error in update_asic_bands_chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("sensor-display", "children"),
            Input("interval", "n_intervals")
        )
        def update_sensor_display(n):
            """更新感測器顯示"""
            try:
                sensor_data = self.data_buffer.get_sensor_data()
                
                display_text = f"""
溫度: {sensor_data['temperature']:.1f}°C
濕度: {sensor_data['humidity']:.1f}%
光線: {sensor_data['light']}
更新: {datetime.now().strftime('%H:%M:%S')}
                """.strip()
                
                return display_text
                
            except Exception as e:
                return f"感測器錯誤: {e}"
        
        @self.app.callback(
            Output("recording-status", "children"),
            [Input("start-recording-btn", "n_clicks"),
             Input("stop-recording-btn", "n_clicks"),
             Input("interval", "n_intervals")],
            prevent_initial_call=True
        )
        def handle_recording_control(start_clicks, stop_clicks, n):
            """處理錄音控制"""
            if not self.audio_recorder:
                return "❌ 音頻錄製器未初始化"
                
            try:
                ctx = callback_context
                if not ctx.triggered:
                    # 定期狀態更新
                    status = self.audio_recorder.get_recording_status()
                    if status['is_recording']:
                        elapsed = status['elapsed_time']
                        return f"🔴 錄音中... ({elapsed:.0f}秒) | 群組ID: {status['current_group_id']}"
                    else:
                        return "⚪ 待機中"
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == "start-recording-btn" and start_clicks:
                    status = self.audio_recorder.get_recording_status()
                    if not status['is_recording']:
                        group_id = str(uuid.uuid4())[:8]
                        success = self.audio_recorder.start_recording(group_id)
                        if success:
                            return f"🔴 錄音開始 | 群組ID: {group_id}"
                        else:
                            return "❌ 錄音啟動失敗"
                    else:
                        return "⚠️ 已在錄音中"
                
                elif button_id == "stop-recording-btn" and stop_clicks:
                    status = self.audio_recorder.get_recording_status()
                    if status['is_recording']:
                        filename = self.audio_recorder.stop_recording(self.db_writer)
                        if filename:
                            return f"✅ 錄音已停止並儲存: {filename}"
                        else:
                            return "⚠️ 錄音停止，但儲存失敗"
                    else:
                        return "⚠️ 目前沒有錄音"
                
                return "⚪ 待機中"
                
            except Exception as e:
                logger.error(f"Error in handle_recording_control: {e}")
                return f"錄音控制錯誤: {e}"
        
        @self.app.callback(
            [Output("performance-status", "children"),
             Output("interval", "interval")],
            Input("interval", "n_intervals")
        )
        def update_performance_status(n):
            """更新效能狀態"""
            try:
                current_time = time.time()
                
                # 系統效能
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_usage = psutil.virtual_memory().percent
                
                # 數據狀態
                data, timestamps = self.data_buffer.get_data()
                latency = (time.time() - timestamps[-1]) * 1000 if len(timestamps) > 0 else 0
                
                # 信號品質
                cognitive_data = self.data_buffer.get_cognitive_data()
                signal_quality = cognitive_data['signal_quality']
                
                # 效能統計
                avg_render = self.performance_monitor['avg_render_time'] * 1000
                
                status_text = (
                    f"CPU: {cpu_usage:.1f}% | "
                    f"Memory: {memory_usage:.1f}% | "
                    f"Latency: {latency:.1f}ms | "
                    f"Render: {avg_render:.1f}ms | "
                    f"Signal: {signal_quality} | "
                    f"Updates: {n}"
                )
                
                return status_text, UI_CONFIG['update_interval']
                
            except Exception as e:
                return f"Status Error: {e}", UI_CONFIG['update_interval']
    
    def run(self, host='0.0.0.0', port=8052, debug=False):
        """執行Dash應用程式"""
        logger.info(f"🚀 Starting EEG Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)