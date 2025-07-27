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
from ui.management_page import ManagementPage
from ui.sliding_panel import SlidingPanel

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

        # 初始化管理頁面
        self.management_page = ManagementPage(self.db_writer)

        # 初始化滑動面板
        self.sliding_panel = SlidingPanel(self.db_writer)

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
        # self.bands = {
        #     "Delta (0.5-2.75Hz)": (0.5, 2.75),
        #     "Theta (3.5-6.75Hz)": (3.5, 6.75),
        #     "Low-Alpha (7.5-9.25Hz)": (7.5, 9.25),
        #     "High-Alpha (10-11.75Hz)": (10, 11.75),
        #     "Low-Beta (13-16.75Hz)": (13, 16.75),
        #     "High-Beta (18-29.75Hz)": (18, 29.75),
        #     "Low-Gamma (31-39.75Hz)": (31, 39.75),
        #     "Mid-Gamma (41-49.75Hz)": (41, 49.75),
        # }

        # 頻帶顏色
        self.band_colors = {
            "Delta (0.5-4Hz)": "#FF6B6B",
            "Theta (4-8Hz)": "#4ECDC4",
            "Alpha (8-12Hz)": "#45B7D1",
            "Beta (12-35Hz)": "#96CEB4",
            "Gamma (35-50Hz)": "#FFEAA7",
        }
        # self.band_colors = {
        #     "Delta (0.5-2.75Hz)": "#FF6B6B",  # 櫻桃紅
        #     "Theta (3.5-6.75Hz)": "#4ECDC4",  # 青綠
        #     "Low-Alpha (7.5-9.25Hz)": "#45B7D1",  # 天空藍
        #     "High-Alpha (10-11.75Hz)": "#00A8E8",  # 寶石藍
        #     "Low-Beta (13-16.75Hz)": "#96CEB4",  # 薄荷綠
        #     "High-Beta (18-29.75Hz)": "#DDA0DD",  # 薰衣草紫
        #     "Low-Gamma (31-39.75Hz)": "#FFB347",  # 蜜橙
        #     "Mid-Gamma (41-49.75Hz)": "#F4D35E",  # 芥末黃
        # }



        # ASIC頻帶名稱
        self.asic_bands = ["Delta", "Theta", "Low-Alpha", "High-Alpha",
                           "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma"]

        # 實驗狀態管理
        self.experiment_state = {
            'current_session_id': None,
            'current_recording_group_id': None,
            'experiment_running': False,
            'recording_active': False,
            'selected_subject': None,
            'selected_sound': None,
            'selected_eye_state': 'open'
        }

        # 設定版面配置和回呼函式
        self._setup_layout()
        self._setup_callbacks()

        # 註冊管理頁面回調
        self.management_page.register_callbacks(self.app)

        # 註冊滑動面板回調
        self.sliding_panel.register_callbacks(self.app)

    def _setup_layout(self):
        """設定主要版面配置"""
        # 為應用程式添加外部樣式
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
                <style>
                    .nav-card:hover {
                        transform: scale(1.05) !important;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
                    }
                    .nav-card.active {
                        transform: scale(1.05) !important;
                        border: 2px solid #fff !important;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.3) !important;
                    }
                    .sensor-card {
                        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        border-left: 4px solid #007bff;
                        transition: all 0.3s ease;
                    }
                    .sensor-card:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

        self.app.layout = html.Div([
            # 滑動面板 (放在最前面以確保正確的z-index層級)
            self.sliding_panel.create_panel_layout(),

            # 頁面路由組件
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="page-store", data="dashboard"),

            # 全局數據存儲（用於頁面間共享數據）
            dcc.Store(id="global-subjects-store", data=[]),
            dcc.Store(id="global-sounds-store", data=[]),

            # 主容器
            html.Div([
                # 標題
                html.H1(UI_CONFIG['title'],
                        style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333'}),

                # 主要內容容器
                html.Div(id="page-content")
            ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '10px'}),
        ])

    def _create_dashboard_layout(self):
        """創建儀表板頁面佈局"""
        return html.Div([

            # 第一行：FFT頻帶分析
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("FFT Band Analysis",
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
                        html.H3("Cognitive Indicator Trends",
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
                        html.H3("Real-time data",
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
                        html.H3("Blink Event Timeline",
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
                        html.H3("Blink Count",
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
                        html.H3("ASIC Band Analysis",
                                style={'fontSize': '18px', 'fontWeight': 'bold',
                                       'marginBottom': '10px', 'color': '#555'}),
                        dcc.Graph(id="asic-bands-chart", style={'height': '300px'},
                                  config={'displayModeBar': False}),
                    ], style={'background': 'white', 'borderRadius': '8px',
                              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                              'padding': '15px', 'marginBottom': '15px'}),
                ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),

            # 第五行：實驗控制和感測器資料
            html.Div([
                # 左側：實驗控制面板
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Experimental Control",
                                    style={'fontSize': '18px', 'fontWeight': 'bold',
                                           'marginBottom': '10px', 'color': '#555'}),

                            # 快速測試會話按鈕
                            html.Div([
                                html.Button("⚡ 快速測試會話", id="quick-test-session-btn",
                                            style={'width': '100%', 'padding': '12px 20px',
                                                   'fontSize': '16px', 'fontWeight': 'bold',
                                                   'backgroundColor': '#17a2b8', 'color': 'white',
                                                   'border': 'none', 'borderRadius': '6px',
                                                   'cursor': 'pointer', 'marginBottom': '15px',
                                                   'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                                html.Hr(style={'margin': '15px 0', 'borderColor': '#dee2e6'}),
                                html.P("或設置完整實驗參數：",
                                       style={'fontSize': '14px', 'color': '#6c757d', 'marginBottom': '10px',
                                              'textAlign': 'center'})
                            ]),

                            # 受試者選擇
                            html.Div([
                                html.Label("Subject ID:",
                                           style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px',
                                                  'display': 'block'}),
                                dcc.Dropdown(
                                    id="subject-dropdown",
                                    placeholder="Select Subject ID",
                                    searchable=True,
                                    clearable=True,
                                    style={'marginBottom': '10px'}
                                ),
                            ]),

                            # 環境音效選擇
                            html.Div([
                                html.Label("Ambient Sound:",
                                           style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px',
                                                  'display': 'block'}),
                                dcc.Dropdown(
                                    id="ambient-sound-dropdown",
                                    placeholder="Ambient Sound",
                                    searchable=True,
                                    clearable=True,
                                    style={'marginBottom': '10px'}
                                ),
                            ]),

                            # 眼睛狀態選擇
                            html.Div([
                                html.Label("眼睛狀態:",
                                           style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px',
                                                  'display': 'block'}),
                                dcc.Dropdown(
                                    id="eye-state-dropdown",
                                    options=[
                                        {'label': 'Open', 'value': 'open'},
                                        {'label': 'Closed', 'value': 'closed'},
                                        {'label': 'Mixed', 'value': 'mixed'}
                                    ],
                                    value='open',
                                    clearable=False,
                                    style={'marginBottom': '15px'}
                                ),
                            ]),

                            # 控制按鈕
                            html.Div([
                                html.Button("Start Recording", id="start-experiment-btn",
                                            style={'marginRight': '10px', 'marginBottom': '10px',
                                                   'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#007bff',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%'}),
                                html.Button("Start Audio Recording", id="start-recording-btn",
                                            style={'marginBottom': '10px', 'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#28a745',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%', 'disabled': True}),
                                html.Button("Stop Recording", id="stop-recording-btn",
                                            style={'marginRight': '10px', 'marginBottom': '10px',
                                                   'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#dc3545',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%', 'disabled': True}),
                                html.Button("Stop Experiment", id="stop-experiment-btn",
                                            style={'marginBottom': '10px', 'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#6c757d',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%', 'disabled': True}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

                            # 狀態顯示 - 突出顯示會話狀態
                            html.Div(id="experiment-status",
                                     style={'fontSize': '16px', 'fontWeight': 'bold', 'marginTop': '15px',
                                            'padding': '15px', 'backgroundColor': '#e3f2fd', 'borderRadius': '8px',
                                            'border': '2px solid #2196f3', 'textAlign': 'center',
                                            'boxShadow': '0 2px 8px rgba(33, 150, 243, 0.2)'}),
                        ], style={'background': 'white', 'borderRadius': '8px',
                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                  'padding': '15px', 'marginBottom': '15px'}),
                    ], id="experiment-controls", style={'display': 'block'}),

                ], style={'flex': '1', 'padding': '5px', 'minWidth': '350px'}),

                # 右側：感測器數據
                html.Div([
                    html.Div([
                        html.H3([
                            html.I(className="fas fa-thermometer-half",
                                   style={'marginRight': '10px', 'color': '#007bff'}),
                            "Environmental Sensor"
                        ], style={'fontSize': '18px', 'fontWeight': 'bold',
                                  'marginBottom': '20px', 'color': '#2c3e50',
                                  'borderBottom': '2px solid #007bff', 'paddingBottom': '10px'}),
                        html.Div(id="sensor-display",
                                 style={'lineHeight': '1.6'}),
                    ], className='sensor-card',
                        style={'background': 'white', 'borderRadius': '12px',
                               'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                               'padding': '20px', 'marginBottom': '15px'}),
                ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
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

        ]),

    def _setup_callbacks(self):
        """設定所有儀表板回呼函式"""

        # 頁面路由回調（簡化版 - 只顯示儀表板）
        @self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")],
            prevent_initial_call=False
        )
        def display_page(pathname):
            """顯示主儀表板頁面"""
            # 現在只顯示儀表板，管理功能通過滑動面板提供
            return self._create_dashboard_layout()

        # 頁面狀態管理已移除 - 現在只使用滑動面板進行管理

        # 全局數據同步回調
        @self.app.callback(
            Output("global-subjects-store", "data"),
            Input("subjects-store-mgmt", "data"),
            prevent_initial_call=True
        )
        def sync_subjects_data(subjects_data):
            """同步受試者數據到全局存儲"""
            if subjects_data:
                return subjects_data
            # 如果沒有數據，從資料庫獲取
            return self.db_writer.get_subjects()

        @self.app.callback(
            Output("global-sounds-store", "data"),
            Input("sounds-store-mgmt", "data"),
            prevent_initial_call=True
        )
        def sync_sounds_data(sounds_data):
            """同步音效數據到全局存儲"""
            if sounds_data:
                return sounds_data
            # 如果沒有數據，從資料庫獲取
            return self.db_writer.get_ambient_sounds()

        # 初始化全局數據存儲
        @self.app.callback(
            [Output("global-subjects-store", "data", allow_duplicate=True),
             Output("global-sounds-store", "data", allow_duplicate=True)],
            Input("page-content", "children"),
            prevent_initial_call='initial_duplicate'
        )
        def initialize_global_data(page_content):
            """頁面載入時初始化全局數據"""
            try:
                subjects = self.db_writer.get_subjects()
                sounds = self.db_writer.get_ambient_sounds()
                return subjects, sounds
            except Exception as e:
                logger.error(f"Error initializing global data: {e}")
                return [], []

        @self.app.callback(
            Output("fft-bands-main", "figure"),
            Input("interval", "n_intervals")
        )
        def update_fft_bands_main(n):
            """更新FFT頻帶視覺化 (分屏波形圖，流動風景效果)"""
            start_time = time.time()

            try:
                # 從數據緩衝區獲取 FFT 頻帶歷史數據
                fft_data = self.data_buffer.get_fft_band_data()
                band_history = fft_data['band_history']

                # 建立多個子圖的折線圖
                band_names = list(self.bands.keys())
                fig = make_subplots(
                    rows=len(band_names),
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=band_names,
                    vertical_spacing=0.05
                )

                # 收集所有時間數據以計算動態範圍
                all_rel_times = []

                # 為每個頻帶繪製流動的功率值曲線
                for i, (band_name, band_color) in enumerate(zip(band_names, self.band_colors.values()), start=1):
                    band_key = band_name.split(' ')[0].lower()
                    
                    if band_key in band_history and len(band_history[band_key]) > 0:
                        # 取得該頻帶的時間序列數據
                        history = band_history[band_key]
                        if history:
                            times, powers = zip(*history)
                            
                            # 計算相對時間（最新的點為0，往前遞減）
                            current_time = times[-1] if times else 0
                            rel_times = [(t - current_time) for t in times]
                            
                            # 收集時間數據用於動態範圍計算
                            all_rel_times.extend(rel_times)
                            
                            # 將功率值轉換為 μV² 單位並進行縮放以便顯示
                            powers_scaled = [p * 1000000 for p in powers]
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=rel_times,
                                    y=powers_scaled,
                                    mode="lines",
                                    line=dict(color=band_color, width=2),
                                    showlegend=False,
                                    name=band_name,
                                    connectgaps=True  # 連接數據點避免斷裂
                                ),
                                row=i, col=1
                            )
                        else:
                            # 沒有數據時顯示零線
                            fig.add_trace(
                                go.Scatter(
                                    x=[-10, 0],
                                    y=[0, 0],
                                    mode="lines",
                                    line=dict(color="gray", width=1, dash="dash"),
                                    showlegend=False
                                ),
                                row=i, col=1
                            )
                    else:
                        # 沒有該頻帶數據時顯示零線
                        fig.add_trace(
                            go.Scatter(
                                x=[-10, 0],
                                y=[0, 0],
                                mode="lines",
                                line=dict(color="gray", width=1, dash="dash"),
                                showlegend=False
                            ),
                            row=i, col=1
                        )

                fig.update_layout(
                    title="FFT Band Power Flowing Waveforms (Moving Landscape)",
                    height=UI_CONFIG['chart_height'],
                    margin=dict(l=40, r=15, t=40, b=60),
                    plot_bgcolor='white',
                    showlegend=False
                )

                # 計算動態x軸範圍
                if all_rel_times:
                    min_time = min(all_rel_times)
                    max_time = max(all_rel_times)
                    # 添加一些邊距以確保數據點不會觸及邊界
                    time_range = max_time - min_time
                    padding = time_range * 0.05 if time_range > 0 else 0.5
                    x_range = [min_time - padding, max_time + padding]
                else:
                    # 默認範圍
                    x_range = [-10, 0]

                # 更新x軸設定 - 顯示相對時間，負值表示過去
                fig.update_xaxes(
                    title_text="Time (s, relative to current)", 
                    row=len(band_names), 
                    col=1,
                    range=x_range
                )
                
                # 設定y軸標籤
                for i in range(1, len(band_names) + 1):
                    fig.update_yaxes(title_text="Power (μV²)", row=i, col=1)

                # 更新效能監控器
                render_time = time.time() - start_time
                self.performance_monitor['avg_render_time'] = (
                        self.performance_monitor['avg_render_time'] * 0.9 + render_time * 0.1
                )

                return fig

            except Exception as e:
                logger.error(f"Error in update_fft_bands_main: {e}")
                return go.Figure().add_annotation(
                    text=f"FFT Band waveform error: {str(e)}",
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

                attention_fig = create_gauge(attention, "Attention", "#1f77b4")
                meditation_fig = create_gauge(meditation, "Relaxation", "#2ca02c")

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
                            name='Attention',
                            line=dict(color='#1f77b4', width=2)
                        ))

                # 放鬆趨勢
                if cognitive_data['meditation_history']:
                    history = list(cognitive_data['meditation_history'])[-max_points:]
                    if history:
                        times, values = zip(*history)
                        base_time = times[0] if times else 0
                        rel_times = [(t - base_time) for t in times]
                        fig.add_trace(go.Scatter(
                            x=rel_times, y=values,
                            mode='lines',
                            name='Relaxation',
                            line=dict(color='#2ca02c', width=2)
                        ))

                fig.update_layout(
                    xaxis_title="Time(s)",
                    yaxis_title="Value",
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
                        name='Blink Event'
                    ))

                fig.update_layout(
                    xaxis_title="Time(s)",
                    yaxis_title="Intensity",
                    height=200,
                    margin=dict(l=30, r=15, t=15, b=30),
                    plot_bgcolor='white'
                )

                return fig

            except Exception as e:
                logger.error(f"Error in update_blink_timeline: {e}")
                return go.Figure()

        # @self.app.callback(
        #     Output("blink-count-chart", "figure"),
        #     Input("interval", "n_intervals")
        # )
        # def update_blink_count_chart(n):
        #     """更新眨眼計數圖表"""
        #     try:
        #         blink_data = self.data_buffer.get_blink_data()
        #         count_history = blink_data['count_history']
        #
        #         fig = go.Figure()
        #
        #         if count_history:
        #             times, counts = zip(*count_history)
        #             base_time = times[0] if times else 0
        #             rel_times = [(t - base_time) for t in times]
        #
        #             fig.add_trace(go.Scatter(
        #                 x=rel_times, y=counts,
        #                 mode='lines+markers',
        #                 name='Cumulative Count',
        #                 line=dict(color='#9467bd', width=2),
        #                 marker=dict(size=4)
        #             ))
        #
        #         fig.update_layout(
        #             xaxis_title="Time(s)",
        #             yaxis_title="Count",
        #             height=200,
        #             margin=dict(l=40, r=20, t=20, b=40),
        #             plot_bgcolor='white'
        #         )
        #
        #         return fig
        #
        #     except Exception as e:
        #         logger.error(f"Error in update_blink_count_chart: {e}")
        #         return go.Figure()

        @self.app.callback(
            Output("blink-count-chart", "figure"),
            Input("interval", "n_intervals")
        )
        def update_blink_count_chart(n):
            """更新眨眼計數為數字顯示"""
            try:
                blink_data = self.data_buffer.get_blink_data()
                count_history = blink_data['count_history']

                # 取最新一筆計數
                latest_count = count_history[-1][1] if count_history else 0

                # 建立一個 number indicator
                fig = go.Figure(go.Indicator(
                    mode="number",
                    value=latest_count,
                    title={
                        "text": "Blink Count",
                        "font": {"size": 16}
                    },
                    domain={"x": [0, 1], "y": [0, 1]}
                ))

                # 調整版面
                fig.update_layout(
                    height=120,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white"
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
                        text="No ASIC data received<br><br>Possible Causes:<br>• ThinkGear device not connected<br>• Serial port setting error<br>• Poor electrode contact",
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
                        name="ASIC Band Power"
                    ))

                fig.update_layout(
                    title="ASIC EEG 8 Band Power Distribution",
                    xaxis_title="Band",
                    yaxis_title="Power Value",
                    yaxis_range=[0, max(current_bands) * 1.1],
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

                # 創建更豐富的顯示格式
                display_components = [
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-thermometer-half",
                                   style={'color': '#e74c3c', 'marginRight': '8px', 'fontSize': '16px'}),
                            html.Span("Temperature", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        ], style={'marginBottom': '5px'}),
                        html.Div(f"{sensor_data['temperature']:.1f}°C",
                                 style={'fontSize': '18px', 'color': '#e74c3c', 'marginLeft': '24px'})
                    ], style={'marginBottom': '15px'}),

                    html.Div([
                        html.Div([
                            html.I(className="fas fa-tint",
                                   style={'color': '#3498db', 'marginRight': '8px', 'fontSize': '16px'}),
                            html.Span("Humidity", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        ], style={'marginBottom': '5px'}),
                        html.Div(f"{sensor_data['humidity']:.1f}%",
                                 style={'fontSize': '18px', 'color': '#3498db', 'marginLeft': '24px'})
                    ], style={'marginBottom': '15px'}),

                    html.Div([
                        html.Div([
                            html.I(className="fas fa-sun",
                                   style={'color': '#f39c12', 'marginRight': '8px', 'fontSize': '16px'}),
                            html.Span("Light", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                        ], style={'marginBottom': '5px'}),
                        html.Div(f"{sensor_data['light']}",
                                 style={'fontSize': '18px', 'color': '#f39c12', 'marginLeft': '24px'})
                    ], style={'marginBottom': '15px'}),

                    html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),

                    html.Div([
                        html.I(className="fas fa-clock",
                               style={'color': '#95a5a6', 'marginRight': '8px', 'fontSize': '14px'}),
                        html.Span(f"Update: {datetime.now().strftime('%H:%M:%S')}",
                                  style={'fontSize': '18px', 'color': '#95a5a6'})
                    ])
                ]

                return display_components

            except Exception as e:
                return html.Div([
                    html.I(className="fas fa-exclamation-triangle",
                           style={'color': '#e74c3c', 'marginRight': '8px'}),
                    html.Span(f"感測器錯誤: {str(e)}", style={'color': '#e74c3c'})
                ])

        # 實驗控制回調函數
        @self.app.callback(
            [Output("subject-dropdown", "options"),
             Output("ambient-sound-dropdown", "options")],
            [Input("interval", "n_intervals"),
             Input("global-subjects-store", "data"),
             Input("global-sounds-store", "data")],
            prevent_initial_call=True
        )
        def update_dropdown_options(n, subjects_data, sounds_data):
            """更新下拉選單選項"""
            try:
                # 獲取受試者列表（優先使用管理頁面的數據，否則從資料庫重新獲取）
                if subjects_data and len(subjects_data) > 0:
                    subjects = subjects_data
                else:
                    subjects = self.db_writer.get_subjects()

                subject_options = [
                    {'label': f"{s['subject_id']} ({s['gender']}, {s['age']}years old)", 'value': s['subject_id']} for s
                    in subjects]

                # 獲取環境音效列表（同樣優先使用管理頁面的數據）
                if sounds_data and len(sounds_data) > 0:
                    sounds = sounds_data
                else:
                    sounds = self.db_writer.get_ambient_sounds()

                sound_options = [{'label': f"{s['sound_name']} ({s['style_category']})", 'value': s['id']} for s in
                                 sounds]

                return subject_options, sound_options
            except Exception as e:
                logger.error(f"Error updating dropdown options: {e}")
                return [], []

        @self.app.callback(
            Output("experiment-status", "children"),
            [Input("start-experiment-btn", "n_clicks"),
             Input("start-recording-btn", "n_clicks"),
             Input("stop-recording-btn", "n_clicks"),
             Input("stop-experiment-btn", "n_clicks"),
             Input("quick-test-session-btn", "n_clicks"),
             Input("interval", "n_intervals")],
            [State("subject-dropdown", "value"),
             State("ambient-sound-dropdown", "value"),
             State("eye-state-dropdown", "value")],
            prevent_initial_call=True
        )
        def handle_experiment_control(start_exp_clicks, start_rec_clicks, stop_rec_clicks, stop_exp_clicks,
                                      quick_test_clicks, n,
                                      subject_id, ambient_sound_id, eye_state):
            """處理實驗控制流程"""
            try:
                ctx = callback_context
                if not ctx.triggered:
                    # 定期狀態更新
                    if self.experiment_state['experiment_running']:
                        session_id = self.experiment_state['current_session_id']
                        recording_status = "🔴 Recording" if self.experiment_state['recording_active'] else "⚪ Standby"
                        return f"📊 Experiment in progress | Conversation: {session_id} | {recording_status}"
                    else:
                        return "⚪ Waiting to start expriment ..."

                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                if button_id == "quick-test-session-btn" and quick_test_clicks:
                    if not self.experiment_state['experiment_running']:
                        # 創建快速測試會話 - 使用默認參數
                        test_subject_id = f"test_user_{int(time.time())}"
                        session_id = self.db_writer.start_experiment_session(
                            subject_id=test_subject_id,
                            eye_state="open",
                            ambient_sound_id=None,
                            researcher_name="QuickTest",
                            notes="Quick test session - auto-generated"
                        )

                        if session_id:
                            self.experiment_state.update({
                                'current_session_id': session_id,
                                'experiment_running': True,
                                'selected_subject': test_subject_id,
                                'selected_sound': None,
                                'selected_eye_state': "open"
                            })
                            return f"🚀 快速測試會話已啟動 | 會話ID: {session_id} | 受試者: {test_subject_id}"
                        else:
                            return "❌ 快速測試會話啟動失敗"
                    else:
                        return "⚠️ 實驗已在進行中，請先停止當前實驗"

                elif button_id == "start-experiment-btn" and start_exp_clicks:
                    if not subject_id:
                        return "❌ Please select a Subject ID first"

                    if not self.experiment_state['experiment_running']:
                        # 開始新的實驗會話
                        session_id = self.db_writer.start_experiment_session(
                            subject_id=subject_id,
                            eye_state=eye_state,
                            ambient_sound_id=ambient_sound_id,
                            researcher_name="System",
                            notes="Automated experiment session"
                        )

                        if session_id:
                            self.experiment_state.update({
                                'current_session_id': session_id,
                                'experiment_running': True,
                                'selected_subject': subject_id,
                                'selected_sound': ambient_sound_id,
                                'selected_eye_state': eye_state
                            })
                            return f"✅ Experiment started | Conversation ID: {session_id}"
                        else:
                            return "❌ Experiment failed to start"
                    else:
                        return "⚠️ Experiment already in progress"

                elif button_id == "start-recording-btn" and start_rec_clicks:
                    if not self.experiment_state['experiment_running']:
                        return "❌ Please start the experiment first"

                    if not self.experiment_state['recording_active']:
                        # 生成錄音群組ID
                        session_id = self.experiment_state['current_session_id']
                        recording_group_id = f"{session_id}_rec_{int(time.time())}"

                        # 開始音頻錄音
                        if self.audio_recorder:
                            success = self.audio_recorder.start_recording(recording_group_id)
                            if success:
                                self.experiment_state.update({
                                    'current_recording_group_id': recording_group_id,
                                    'recording_active': True
                                })
                                return f"🔴 Recording started | Group ID: {recording_group_id}"
                            else:
                                return "❌ Recording failed to start"
                        else:
                            return "❌ Audio recorder not initialized"
                    else:
                        return "⚠️ Recording in progress"

                elif button_id == "stop-recording-btn" and stop_rec_clicks:
                    if self.experiment_state['recording_active']:
                        # 停止音頻錄音
                        if self.audio_recorder:
                            filename = self.audio_recorder.stop_recording(self.db_writer)
                            self.experiment_state.update({
                                'current_recording_group_id': None,
                                'recording_active': False
                            })
                            if filename:
                                return f"✅ Recording stopped | File: {os.path.basename(filename)}"
                            else:
                                return "⚠️ Recording stopped, but saving failed"
                        else:
                            return "❌ Audio recorder not initialized"
                    else:
                        return "⚠️ No recording currently"

                elif button_id == "stop-experiment-btn" and stop_exp_clicks:
                    if self.experiment_state['experiment_running']:
                        # 如果還在錄音，先停止錄音
                        if self.experiment_state['recording_active'] and self.audio_recorder:
                            self.audio_recorder.stop_recording(self.db_writer)

                        # 結束實驗會話
                        session_id = self.experiment_state['current_session_id']
                        success = self.db_writer.end_experiment_session(session_id)

                        if success:
                            self.experiment_state.update({
                                'current_session_id': None,
                                'current_recording_group_id': None,
                                'experiment_running': False,
                                'recording_active': False,
                                'selected_subject': None,
                                'selected_sound': None
                            })
                            return f"✅ Experiment ended | Conversation: {session_id}"
                        else:
                            return "❌ Experiment failed to end"
                    else:
                        return "⚠️ No experiment in progress"

                return "⚪ Waiting for operation..."

            except Exception as e:
                logger.error(f"Error in handle_experiment_control: {e}")
                return f"❌ Experiment control error: {str(e)}"

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
                # 檢查音頻模組是否可用
                status = self.audio_recorder.get_recording_status()
                if not status.get('audio_available', False):
                    return "❌ Audio module not installed (pip install sounddevice scipy)"

                ctx = callback_context
                if not ctx.triggered:
                    # 定期狀態更新
                    if status['is_recording']:
                        elapsed = status['elapsed_time']
                        group_id = status['current_group_id'] or "Unknown"
                        return f"🔴 Recording in progress... ({elapsed:.0f}秒) | Group ID: {group_id}"
                    else:
                        device_info = self.audio_recorder.get_device_info()
                        if device_info.get('available', False) and 'error' not in device_info:
                            device_name = device_info.get('name', 'Unknown Device')
                            return f"⚪ On standby | Device: {device_name}"
                        else:
                            error_msg = device_info.get('error', 'Unknown Error')
                            return f"⚠️ Devicec error: {error_msg}"

                button_id = ctx.triggered[0]['prop_id'].split('.')[0]

                if button_id == "start-recording-btn" and start_clicks:
                    if not status['is_recording']:
                        group_id = str(uuid.uuid4())[:8]
                        success = self.audio_recorder.start_recording(group_id)
                        if success:
                            return f"🔴 錄音開始 | 群組ID: {group_id}"
                        else:
                            return "❌ 錄音啟動失敗 - 請檢查音頻設備"
                    else:
                        return "⚠️ 已在錄音中"

                elif button_id == "stop-recording-btn" and stop_clicks:
                    if status['is_recording']:
                        filename = self.audio_recorder.stop_recording(self.db_writer)
                        if filename:
                            return f"✅ 錄音已停止並儲存: {os.path.basename(filename)}"
                        else:
                            return "⚠️ 錄音停止，但儲存失敗"
                    else:
                        return "⚠️ 目前沒有錄音"

                return "⚪ 待機中"

            except Exception as e:
                logger.error(f"Error in handle_recording_control: {e}")
                return f"❌ 錄音控制錯誤: {str(e)}"

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
        logger.info(f"Starting EEG Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)