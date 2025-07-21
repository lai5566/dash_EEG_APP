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
                
                # 導航卡片區域
                html.Div([
                    html.Div([
                        # 管理中心卡片
                        html.Div([
                            html.Div([
                                html.H4("📊 管理中心",
                                        style={'fontSize': '18px', 'fontWeight': 'bold',
                                               'marginBottom': '10px', 'color': '#fff',
                                               'textAlign': 'center'}),
                                html.P("受試者註冊\n音效上傳",
                                      style={'fontSize': '14px', 'color': '#fff',
                                            'textAlign': 'center', 'margin': '0',
                                            'whiteSpace': 'pre-line'})
                            ], style={'padding': '20px', 'cursor': 'pointer',
                                     'transition': 'all 0.3s ease'}),
                        ], id="management-card", className="nav-card",
                           style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                  'borderRadius': '12px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                                  'marginBottom': '15px', 'cursor': 'pointer',
                                  'transform': 'scale(1)', 'transition': 'all 0.3s ease',
                                  'flex': '1', 'marginRight': '10px'}),
                        
                        # EEG 實驗卡片
                        html.Div([
                            html.Div([
                                html.H4("📈 即時EEG",
                                        style={'fontSize': '18px', 'fontWeight': 'bold',
                                               'marginBottom': '10px', 'color': '#fff',
                                               'textAlign': 'center'}),
                                html.P("實驗控制\n數據監控",
                                      style={'fontSize': '14px', 'color': '#fff',
                                            'textAlign': 'center', 'margin': '0',
                                            'whiteSpace': 'pre-line'})
                            ], style={'padding': '20px', 'cursor': 'pointer',
                                     'transition': 'all 0.3s ease'}),
                        ], id="dashboard-card", className="nav-card active",
                           style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                  'borderRadius': '12px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                                  'marginBottom': '15px', 'cursor': 'pointer',
                                  'transform': 'scale(1.05)', 'transition': 'all 0.3s ease',
                                  'border': '2px solid #fff', 'flex': '1', 'marginLeft': '10px'}),
                        
                    ], style={'display': 'flex', 'marginBottom': '20px', 'padding': '0 20px'}),
                ]),
                
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

                # 第五行：頁面切換卡片和感測器資料
                html.Div([
                    # 左側：頁面切換卡片
                    html.Div([
                        # 管理中心卡片
                        html.Div([
                            html.Div([
                                html.H4("📊 管理中心",
                                        style={'fontSize': '18px', 'fontWeight': 'bold',
                                               'marginBottom': '10px', 'color': '#fff',
                                               'textAlign': 'center'}),
                                html.P("受試者註冊\n音效上傳",
                                      style={'fontSize': '14px', 'color': '#fff',
                                            'textAlign': 'center', 'margin': '0',
                                            'whiteSpace': 'pre-line'})
                            ], style={'padding': '20px', 'cursor': 'pointer',
                                     'transition': 'all 0.3s ease'}),
                        ], id="management-card", className="nav-card",
                           style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                  'borderRadius': '12px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                                  'marginBottom': '15px', 'cursor': 'pointer',
                                  'transform': 'scale(1)', 'transition': 'all 0.3s ease'}),
                        
                        # EEG 實驗卡片
                        html.Div([
                            html.Div([
                                html.H4("📈 即時EEG",
                                        style={'fontSize': '18px', 'fontWeight': 'bold',
                                               'marginBottom': '10px', 'color': '#fff',
                                               'textAlign': 'center'}),
                                html.P("實驗控制\n數據監控",
                                      style={'fontSize': '14px', 'color': '#fff',
                                            'textAlign': 'center', 'margin': '0',
                                            'whiteSpace': 'pre-line'})
                            ], style={'padding': '20px', 'cursor': 'pointer',
                                     'transition': 'all 0.3s ease'}),
                        ], id="dashboard-card", className="nav-card active",
                           style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                  'borderRadius': '12px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                                  'marginBottom': '15px', 'cursor': 'pointer',
                                  'transform': 'scale(1.05)', 'transition': 'all 0.3s ease',
                                  'border': '2px solid #fff'}),
                        
                        # 實驗控制面板（當在儀表板模式時顯示）
                        html.Div([
                            html.Div([
                                html.H3("實驗控制",
                                        style={'fontSize': '18px', 'fontWeight': 'bold',
                                               'marginBottom': '10px', 'color': '#555'}),
                            
                            # 受試者選擇
                            html.Div([
                                html.Label("受試者ID:", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id="subject-dropdown",
                                    placeholder="選擇或輸入受試者ID",
                                    searchable=True,
                                    clearable=True,
                                    style={'marginBottom': '10px'}
                                ),
                            ]),
                            
                            # 環境音效選擇
                            html.Div([
                                html.Label("環境音效:", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id="ambient-sound-dropdown",
                                    placeholder="選擇環境音效 (可選)",
                                    searchable=True,
                                    clearable=True,
                                    style={'marginBottom': '10px'}
                                ),
                            ]),
                            
                            # 眼睛狀態選擇
                            html.Div([
                                html.Label("眼睛狀態:", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                                dcc.Dropdown(
                                    id="eye-state-dropdown",
                                    options=[
                                        {'label': '睜眼', 'value': 'open'},
                                        {'label': '閉眼', 'value': 'closed'},
                                        {'label': '混合', 'value': 'mixed'}
                                    ],
                                    value='open',
                                    clearable=False,
                                    style={'marginBottom': '15px'}
                                ),
                            ]),
                            
                            # 控制按鈕
                            html.Div([
                                html.Button("📊 開始記錄", id="start-experiment-btn",
                                            style={'marginRight': '10px', 'marginBottom': '10px', 'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#007bff',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%'}),
                                html.Button("🎙️ 開始錄音", id="start-recording-btn",
                                            style={'marginBottom': '10px', 'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#28a745',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%', 'disabled': True}),
                                html.Button("⏹️ 停止錄音", id="stop-recording-btn",
                                            style={'marginRight': '10px', 'marginBottom': '10px', 'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#dc3545',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%', 'disabled': True}),
                                html.Button("🛑 停止實驗", id="stop-experiment-btn",
                                            style={'marginBottom': '10px', 'padding': '10px 20px',
                                                   'fontSize': '14px', 'backgroundColor': '#6c757d',
                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
                                                   'cursor': 'pointer', 'width': '48%', 'disabled': True}),
                            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),
                            
                            # 狀態顯示
                            html.Div(id="experiment-status",
                                     style={'fontSize': '12px', 'color': '#666', 'marginTop': '10px',
                                            'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'}),
                            ], style={'background': 'white', 'borderRadius': '8px',
                                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                      'padding': '15px', 'marginBottom': '15px'}),
                        ], id="experiment-controls", style={'display': 'block'}),
                        
                    ], style={'flex': '1', 'padding': '5px', 'minWidth': '350px'}),

                    # 右側：感測器數據
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


    def _setup_callbacks(self):
        """設定所有儀表板回呼函式"""
        
        # 頁面路由回調
        @self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname"),
             Input("management-card", "n_clicks"),
             Input("dashboard-card", "n_clicks"),
             Input("back-to-dashboard-btn", "n_clicks"),
             Input("page-store", "data")],
            prevent_initial_call=False
        )
        def display_page(pathname, management_clicks, dashboard_clicks, back_clicks, current_page):
            """根據URL或卡片點擊顯示對應頁面"""
            ctx = callback_context
            
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if trigger_id == "management-card" and management_clicks:
                    return self.management_page.create_layout()
                elif trigger_id == "dashboard-card" and dashboard_clicks:
                    return self._create_dashboard_layout()
                elif trigger_id == "back-to-dashboard-btn" and back_clicks:
                    return self._create_dashboard_layout()
            
            # 預設顯示儀表板
            return self._create_dashboard_layout()
        
        # 頁面狀態管理
        @self.app.callback(
            Output("page-store", "data"),
            [Input("management-card", "n_clicks"),
             Input("dashboard-card", "n_clicks"),
             Input("back-to-dashboard-btn", "n_clicks")],
            prevent_initial_call=True
        )
        def update_page_state(management_clicks, dashboard_clicks, back_clicks):
            """更新頁面狀態"""
            ctx = callback_context
            
            if ctx.triggered:
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if trigger_id == "management-card" and management_clicks:
                    return "management"
                elif trigger_id == "dashboard-card" and dashboard_clicks:
                    return "dashboard"
                elif trigger_id == "back-to-dashboard-btn" and back_clicks:
                    return "dashboard"
            
            return "dashboard"
        
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

                        for i, (band_name, band_color) in enumerate(zip(band_names, self.band_colors.values()),
                                                                    start=1):
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
                meditation_fig = create_gauge(meditation, "放鬆", "#2ca02c")

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
                            name=';放鬆',
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

                display_text = f"""
溫度: {sensor_data['temperature']:.1f}°C
濕度: {sensor_data['humidity']:.1f}%
光線: {sensor_data['light']}
更新: {datetime.now().strftime('%H:%M:%S')}
                """.strip()

                return display_text

            except Exception as e:
                return f"感測器錯誤: {e}"

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
                
                subject_options = [{'label': f"{s['subject_id']} ({s['gender']}, {s['age']}歲)", 'value': s['subject_id']} for s in subjects]
                
                # 獲取環境音效列表（同樣優先使用管理頁面的數據）
                if sounds_data and len(sounds_data) > 0:
                    sounds = sounds_data
                else:
                    sounds = self.db_writer.get_ambient_sounds()
                
                sound_options = [{'label': f"{s['sound_name']} ({s['style_category']})", 'value': s['id']} for s in sounds]
                
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
             Input("interval", "n_intervals")],
            [State("subject-dropdown", "value"),
             State("ambient-sound-dropdown", "value"),
             State("eye-state-dropdown", "value")],
            prevent_initial_call=True
        )
        def handle_experiment_control(start_exp_clicks, start_rec_clicks, stop_rec_clicks, stop_exp_clicks, n,
                                     subject_id, ambient_sound_id, eye_state):
            """處理實驗控制流程"""
            try:
                ctx = callback_context
                if not ctx.triggered:
                    # 定期狀態更新
                    if self.experiment_state['experiment_running']:
                        session_id = self.experiment_state['current_session_id']
                        recording_status = "🔴 錄音中" if self.experiment_state['recording_active'] else "⚪ 待機"
                        return f"📊 實驗進行中 | 會話: {session_id} | {recording_status}"
                    else:
                        return "⚪ 等待開始實驗..."

                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == "start-experiment-btn" and start_exp_clicks:
                    if not subject_id:
                        return "❌ 請先選擇受試者ID"
                    
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
                            return f"✅ 實驗已開始 | 會話ID: {session_id}"
                        else:
                            return "❌ 實驗啟動失敗"
                    else:
                        return "⚠️ 實驗已在進行中"
                
                elif button_id == "start-recording-btn" and start_rec_clicks:
                    if not self.experiment_state['experiment_running']:
                        return "❌ 請先開始實驗"
                    
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
                                return f"🔴 錄音已開始 | 群組ID: {recording_group_id}"
                            else:
                                return "❌ 錄音啟動失敗"
                        else:
                            return "❌ 音頻錄製器未初始化"
                    else:
                        return "⚠️ 已在錄音中"
                
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
                                return f"✅ 錄音已停止 | 檔案: {os.path.basename(filename)}"
                            else:
                                return "⚠️ 錄音停止，但儲存失敗"
                        else:
                            return "❌ 音頻錄製器未初始化"
                    else:
                        return "⚠️ 目前沒有錄音"
                
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
                            return f"✅ 實驗已結束 | 會話: {session_id}"
                        else:
                            return "❌ 實驗結束失敗"
                    else:
                        return "⚠️ 沒有進行中的實驗"
                
                return "⚪ 等待操作..."
                
            except Exception as e:
                logger.error(f"Error in handle_experiment_control: {e}")
                return f"❌ 實驗控制錯誤: {str(e)}"

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
                    return "❌ 音頻模組未安裝 (pip install sounddevice scipy)"

                ctx = callback_context
                if not ctx.triggered:
                    # 定期狀態更新
                    if status['is_recording']:
                        elapsed = status['elapsed_time']
                        group_id = status['current_group_id'] or "未知"
                        return f"🔴 錄音中... ({elapsed:.0f}秒) | 群組ID: {group_id}"
                    else:
                        device_info = self.audio_recorder.get_device_info()
                        if device_info.get('available', False) and 'error' not in device_info:
                            device_name = device_info.get('name', '未知設備')
                            return f"⚪ 待機中 | 設備: {device_name}"
                        else:
                            error_msg = device_info.get('error', '未知錯誤')
                            return f"⚠️ 設備錯誤: {error_msg}"

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
        logger.info(f"🚀 Starting EEG Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
# """EEG儀表板的Dash網頁介面"""
#
# import time
# import uuid
# import logging
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from datetime import datetime
# from typing import Dict, Any, Optional
#
# import dash
# from dash import dcc, html, Input, Output, State, callback_context
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# import numpy as np
# import psutil
#
# from core.eeg_processor import RealTimeEEGProcessor
# from models.data_buffer import EnhancedCircularBuffer
# from services.database_service import EnhancedDatabaseWriter
# from services.mqtt_client import MQTTSensorClient
# from services.audio_recorder import AudioRecorder
# from utils.data_utils import DataValidator, DataProcessor
# from resources.config.app_config import UI_CONFIG, PROCESSING_CONFIG, API_CONFIG
#
# logger = logging.getLogger(__name__)
#
#
# class EEGDashboardApp:
#     """EEG監控的主要Dash應用程式"""
#
#     def __init__(self, data_buffer: EnhancedCircularBuffer,
#                  db_writer: EnhancedDatabaseWriter,
#                  processor: RealTimeEEGProcessor,
#                  mqtt_client: Optional[MQTTSensorClient] = None,
#                  audio_recorder: Optional[AudioRecorder] = None):
#
#         self.data_buffer = data_buffer
#         self.db_writer = db_writer
#         self.processor = processor
#         self.mqtt_client = mqtt_client
#         self.audio_recorder = audio_recorder
#
#         # 初始化Dash應用程式
#         self.app = dash.Dash(__name__)
#
#         # 效能監控
#         self.performance_monitor = {
#             'last_update_time': time.time(),
#             'update_count': 0,
#             'avg_render_time': 0,
#             'adaptive_interval': UI_CONFIG['update_interval']
#         }
#
#         # EEG頻帶視覺化設定
#         self.bands = {
#             "Delta (0.5-4Hz)": (0.5, 4),
#             "Theta (4-8Hz)": (4, 8),
#             "Alpha (8-12Hz)": (8, 12),
#             "Beta (12-35Hz)": (12, 35),
#             "Gamma (35-50Hz)": (35, 50),
#         }
#
#         # 頻帶顏色
#         self.band_colors = {
#             "Delta (0.5-4Hz)": "#FF6B6B",
#             "Theta (4-8Hz)": "#4ECDC4",
#             "Alpha (8-12Hz)": "#45B7D1",
#             "Beta (12-35Hz)": "#96CEB4",
#             "Gamma (35-50Hz)": "#FFEAA7",
#         }
#
#         # ASIC頻帶名稱
#         self.asic_bands = ["Delta", "Theta", "Low-Alpha", "High-Alpha",
#                           "Low-Beta", "High-Beta", "Low-Gamma", "Mid-Gamma"]
#
#         # 設定版面配置和回呼函式
#         self._setup_layout()
#         self._setup_callbacks()
#
#     def _setup_layout(self):
#         """設定主要版面配置"""
#         self.app.layout = html.Div([
#             html.Div([
#                 # 標題
#                 html.H1(UI_CONFIG['title'],
#                         style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333'}),
#
#                 # 第一行：FFT頻帶分析
#                 html.Div([
#                     html.Div([
#                         html.Div([
#                             html.H3("FFT頻帶分析",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             dcc.Graph(id="fft-bands-main",
#                                      style={'height': f'{UI_CONFIG["chart_height"]}px'},
#                                      config={'displayModeBar': False}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
#                 ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
#
#                 # 第二行：認知指標
#                 html.Div([
#                     # 左側：趨勢圖表
#                     html.Div([
#                         html.Div([
#                             html.H3("認知指標趨勢",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             dcc.Graph(id="cognitive-trends", style={'height': '250px'},
#                                      config={'displayModeBar': False}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
#
#                     # 右側：儀表
#                     html.Div([
#                         html.Div([
#                             html.H3("即時數值",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             html.Div([
#                                 dcc.Graph(id="attention-gauge", style={'height': '120px'},
#                                          config={'displayModeBar': False}),
#                                 dcc.Graph(id="meditation-gauge", style={'height': '120px'},
#                                          config={'displayModeBar': False}),
#                             ]),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
#                 ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
#
#                 # 第三行：眨眼檢測
#                 html.Div([
#                     # 左側：事件時間軸
#                     html.Div([
#                         html.Div([
#                             html.H3("眨眼事件時間軸",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             dcc.Graph(id="blink-timeline", style={'height': '200px'},
#                                      config={'displayModeBar': False}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
#
#                     # 右側：眨眼計數
#                     html.Div([
#                         html.Div([
#                             html.H3("眨眼計數",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             dcc.Graph(id="blink-count-chart", style={'height': '200px'},
#                                      config={'displayModeBar': False}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
#                 ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
#
#                 # 第四行：ASIC頻帶
#                 html.Div([
#                     html.Div([
#                         html.Div([
#                             html.H3("ASIC頻帶分析",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             dcc.Graph(id="asic-bands-chart", style={'height': '300px'},
#                                      config={'displayModeBar': False}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
#                 ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
#
#                 # 第五行：感測器和錄音
#                 html.Div([
#                     # 左側：感測器數據
#                     html.Div([
#                         html.Div([
#                             html.H3("環境感測器",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             html.Div(id="sensor-display",
#                                      style={'fontSize': '12px', 'lineHeight': '1.5',
#                                             'fontFamily': 'monospace'}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '2', 'padding': '5px', 'minWidth': '300px'}),
#
#                     # 右側：錄音控制
#                     html.Div([
#                         html.Div([
#                             html.H3("錄音控制",
#                                     style={'fontSize': '18px', 'fontWeight': 'bold',
#                                            'marginBottom': '10px', 'color': '#555'}),
#                             html.Div([
#                                 html.Button("🎙️ 開始錄音", id="start-recording-btn",
#                                            style={'marginRight': '10px', 'padding': '10px 20px',
#                                                   'fontSize': '14px', 'backgroundColor': '#28a745',
#                                                   'color': 'white', 'border': 'none', 'borderRadius': '4px',
#                                                   'cursor': 'pointer'}),
#                                 html.Button("⏹️ 停止錄音", id="stop-recording-btn",
#                                            style={'padding': '10px 20px', 'fontSize': '14px',
#                                                   'backgroundColor': '#dc3545', 'color': 'white',
#                                                   'border': 'none', 'borderRadius': '4px',
#                                                   'cursor': 'pointer'}),
#                             ], style={'marginBottom': '10px'}),
#                             html.Div(id="recording-status",
#                                      style={'fontSize': '12px', 'color': '#666'}),
#                         ], style={'background': 'white', 'borderRadius': '8px',
#                                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
#                                  'padding': '15px', 'marginBottom': '15px'}),
#                     ], style={'flex': '1', 'padding': '5px', 'minWidth': '300px'}),
#                 ], style={'display': 'flex', 'flexWrap': 'wrap', 'margin': '-5px'}),
#
#                 # 狀態列
#                 html.Div([
#                     html.Div(id="performance-status",
#                              style={'fontSize': '12px', 'color': '#666',
#                                     'textAlign': 'center', 'padding': '10px',
#                                     'borderTop': '1px solid #eee'}),
#                 ]),
#
#                 # 間隔組件
#                 dcc.Interval(id="interval",
#                            interval=UI_CONFIG['update_interval'],
#                            n_intervals=0),
#                 dcc.Store(id="performance-store", data={}),
#
#             ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '10px'}),
#         ])
#
#     def _setup_callbacks(self):
#         """設定所有儀表板回呼函式"""
#
#         @self.app.callback(
#             Output("fft-bands-main", "figure"),
#             Input("interval", "n_intervals")
#         )
#         def update_fft_bands_main(n):
#             """更新FFT頻帶視覺化 (折線圖)"""
#             start_time = time.time()
#
#             try:
#                 # 取得目前視窗並進行處理
#                 processed_result = self.processor.process_current_window()
#
#                 if not processed_result:
#                     return go.Figure().add_annotation(
#                         text="EEG處理器錯誤<br>正在初始化...",
#                         showarrow=False, x=0.5, y=0.5,
#                         xref="paper", yref="paper",
#                         font=dict(size=16, color="red")
#                     )
#
#                 if 'fft_bands' not in processed_result:
#                     return go.Figure().add_annotation(
#                         text="FFT頻段數據缺失<br>正在生成測試數據...",
#                         showarrow=False, x=0.5, y=0.5,
#                         xref="paper", yref="paper",
#                         font=dict(size=16, color="orange")
#                     )
#
#                 fft_bands = processed_result['fft_bands']
#
#                 # 驗證 FFT 頻段數據
#                 if not fft_bands or all(len(band_data) == 0 for band_data in fft_bands.values()):
#                     return go.Figure().add_annotation(
#                         text="FFT頻段為空<br>正在重新生成數據...",
#                         showarrow=False, x=0.5, y=0.5,
#                         xref="paper", yref="paper",
#                         font=dict(size=16, color="orange")
#                     )
#
#                 # 建立多個子圖的折線圖
#                 band_names = list(self.bands.keys())
#                 fig = make_subplots(
#                     rows=len(band_names),
#                     cols=1,
#                     shared_xaxes=True,
#                     subplot_titles=band_names,
#                     vertical_spacing=0.05
#                 )
#
#                 # 計算時間軸
#                 if len(fft_bands) > 0:
#                     # 找到第一個非空的頻段來計算時間軸
#                     sample_length = 0
#                     for band_data in fft_bands.values():
#                         if len(band_data) > 0:
#                             sample_length = len(band_data)
#                             break
#
#                     if sample_length > 0:
#                         t = np.arange(sample_length) / self.processor.sample_rate
#
#                         for i, (band_name, band_color) in enumerate(zip(band_names, self.band_colors.values()), start=1):
#                             band_key = band_name.split(' ')[0].lower()
#                             band_signal = fft_bands.get(band_key, np.array([]))
#
#                             if len(band_signal) > 0:
#                                 fig.add_trace(
#                                     go.Scatter(
#                                         x=t,
#                                         y=band_signal,
#                                         mode="lines",
#                                         line=dict(color=band_color, width=1.5),
#                                         showlegend=False
#                                     ),
#                                     row=i, col=1
#                                 )
#                             else:
#                                 # 如果頻段數據為空，顯示零線
#                                 fig.add_trace(
#                                     go.Scatter(
#                                         x=t,
#                                         y=np.zeros(len(t)),
#                                         mode="lines",
#                                         line=dict(color="gray", width=1, dash="dash"),
#                                         showlegend=False
#                                     ),
#                                     row=i, col=1
#                                 )
#
#                 fig.update_layout(
#                     title="FFT頻帶分析 (時域波形)",
#                     height=UI_CONFIG['chart_height'],
#                     margin=dict(l=40, r=15, t=40, b=60),
#                     plot_bgcolor='white',
#                     showlegend=False
#                 )
#
#                 # 更新x軸標籤
#                 fig.update_xaxes(title_text="時間 (秒)", row=len(band_names), col=1)
#                 fig.update_yaxes(title_text="振幅")
#
#                 # 更新效能監控器
#                 render_time = time.time() - start_time
#                 self.performance_monitor['avg_render_time'] = (
#                     self.performance_monitor['avg_render_time'] * 0.9 + render_time * 0.1
#                 )
#
#                 return fig
#
#             except Exception as e:
#                 logger.error(f"Error in update_fft_bands_main: {e}")
#                 return go.Figure().add_annotation(
#                     text=f"頻帶分析錯誤: {str(e)}",
#                     showarrow=False, x=0.5, y=0.5,
#                     xref="paper", yref="paper"
#                 )
#
#         @self.app.callback(
#             [Output("attention-gauge", "figure"),
#              Output("meditation-gauge", "figure")],
#             Input("interval", "n_intervals")
#         )
#         def update_cognitive_gauges(n):
#             """更新認知指標儀表"""
#             try:
#                 cognitive_data = self.data_buffer.get_cognitive_data()
#                 attention = cognitive_data['attention']
#                 meditation = cognitive_data['meditation']
#
#                 def create_gauge(value, title, color):
#                     fig = go.Figure()
#                     fig.add_trace(go.Indicator(
#                         mode="gauge+number",
#                         value=value,
#                         title={'text': title, 'font': {'size': 12}},
#                         domain={'x': [0, 1], 'y': [0, 1]},
#                         gauge={
#                             'axis': {'range': [0, 100], 'tickwidth': 0},
#                             'bar': {'color': color, 'thickness': 0.4},
#                             'bgcolor': "white",
#                             'borderwidth': 1,
#                             'bordercolor': "lightgray"
#                         }
#                     ))
#                     fig.update_layout(
#                         height=120,
#                         margin=dict(l=5, r=5, t=15, b=5),
#                         font={'size': 9}
#                     )
#                     return fig
#
#                 attention_fig = create_gauge(attention, "注意力", "#1f77b4")
#                 meditation_fig = create_gauge(meditation, "冥想", "#2ca02c")
#
#                 return attention_fig, meditation_fig
#
#             except Exception as e:
#                 logger.error(f"Error in update_cognitive_gauges: {e}")
#                 empty_fig = go.Figure()
#                 empty_fig.update_layout(height=120)
#                 return empty_fig, empty_fig
#
#         @self.app.callback(
#             Output("cognitive-trends", "figure"),
#             Input("interval", "n_intervals")
#         )
#         def update_cognitive_trends(n):
#             """更新認知趨勢圖表"""
#             try:
#                 cognitive_data = self.data_buffer.get_cognitive_data()
#
#                 fig = go.Figure()
#
#                 max_points = UI_CONFIG['max_points']
#
#                 # 注意力趨勢
#                 if cognitive_data['attention_history']:
#                     history = list(cognitive_data['attention_history'])[-max_points:]
#                     if history:
#                         times, values = zip(*history)
#                         base_time = times[0] if times else 0
#                         rel_times = [(t - base_time) for t in times]
#                         fig.add_trace(go.Scatter(
#                             x=rel_times, y=values,
#                             mode='lines',
#                             name='注意力',
#                             line=dict(color='#1f77b4', width=2)
#                         ))
#
#                 # 冥想趨勢
#                 if cognitive_data['meditation_history']:
#                     history = list(cognitive_data['meditation_history'])[-max_points:]
#                     if history:
#                         times, values = zip(*history)
#                         base_time = times[0] if times else 0
#                         rel_times = [(t - base_time) for t in times]
#                         fig.add_trace(go.Scatter(
#                             x=rel_times, y=values,
#                             mode='lines',
#                             name='冥想',
#                             line=dict(color='#2ca02c', width=2)
#                         ))
#
#                 fig.update_layout(
#                     xaxis_title="時間 (秒)",
#                     yaxis_title="數值",
#                     yaxis_range=[0, 100],
#                     height=250,
#                     margin=dict(l=30, r=15, t=15, b=30),
#                     legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
#                     plot_bgcolor='white'
#                 )
#
#                 return fig
#
#             except Exception as e:
#                 logger.error(f"Error in update_cognitive_trends: {e}")
#                 return go.Figure()
#
#         @self.app.callback(
#             Output("blink-timeline", "figure"),
#             Input("interval", "n_intervals")
#         )
#         def update_blink_timeline(n):
#             """更新眨眼時間軸"""
#             try:
#                 blink_data = self.data_buffer.get_blink_data()
#                 events = list(blink_data['events'])[-10:]  # 最後10個事件
#
#                 fig = go.Figure()
#
#                 if events:
#                     times, intensities = zip(*events)
#                     base_time = times[0] if times else 0
#                     rel_times = [(t - base_time) for t in times]
#
#                     fig.add_trace(go.Scatter(
#                         x=rel_times, y=intensities,
#                         mode='markers',
#                         marker=dict(size=8, color='red', opacity=0.7),
#                         name='眨眼事件'
#                     ))
#
#                 fig.update_layout(
#                     xaxis_title="時間 (秒)",
#                     yaxis_title="強度",
#                     height=200,
#                     margin=dict(l=30, r=15, t=15, b=30),
#                     plot_bgcolor='white'
#                 )
#
#                 return fig
#
#             except Exception as e:
#                 logger.error(f"Error in update_blink_timeline: {e}")
#                 return go.Figure()
#
#         @self.app.callback(
#             Output("blink-count-chart", "figure"),
#             Input("interval", "n_intervals")
#         )
#         def update_blink_count_chart(n):
#             """更新眨眼計數圖表"""
#             try:
#                 blink_data = self.data_buffer.get_blink_data()
#                 count_history = blink_data['count_history']
#
#                 fig = go.Figure()
#
#                 if count_history:
#                     times, counts = zip(*count_history)
#                     base_time = times[0] if times else 0
#                     rel_times = [(t - base_time) for t in times]
#
#                     fig.add_trace(go.Scatter(
#                         x=rel_times, y=counts,
#                         mode='lines+markers',
#                         name='累計次數',
#                         line=dict(color='#9467bd', width=2),
#                         marker=dict(size=4)
#                     ))
#
#                 fig.update_layout(
#                     xaxis_title="時間 (秒)",
#                     yaxis_title="次數",
#                     height=200,
#                     margin=dict(l=40, r=20, t=20, b=40),
#                     plot_bgcolor='white'
#                 )
#
#                 return fig
#
#             except Exception as e:
#                 logger.error(f"Error in update_blink_count_chart: {e}")
#                 return go.Figure()
#
#         @self.app.callback(
#             Output("asic-bands-chart", "figure"),
#             Input("interval", "n_intervals")
#         )
#         def update_asic_bands_chart(n):
#             """更新ASIC頻帶圖表"""
#             try:
#                 asic_data = self.data_buffer.get_asic_data()
#                 current_bands = asic_data['current_bands']
#                 print(f"[ASIC DEBUG] DashApp: Retrieved ASIC bands for display: {current_bands}")
#
#                 fig = go.Figure()
#
#                 if all(band == 0 for band in current_bands):
#                     # 沒有ASIC數據
#                     print(f"[ASIC DEBUG] DashApp: No ASIC data - all bands are zero")
#                     fig.add_annotation(
#                         text="沒收到ASIC數據<br><br>可能原因:<br>• ThinkGear設備未連接<br>• 串口設定錯誤<br>• 電極接觸不良",
#                         showarrow=False, x=0.5, y=0.5,
#                         xref="paper", yref="paper",
#                         font=dict(size=14, color="red"),
#                         bgcolor="rgba(255,255,255,0.9)",
#                         bordercolor="red", borderwidth=2
#                     )
#                 else:
#                     # 顯示ASIC數據
#                     print(f"[ASIC DEBUG] DashApp: Displaying ASIC chart with data: {current_bands}")
#                     fig.add_trace(go.Bar(
#                         x=self.asic_bands,
#                         y=current_bands,
#                         marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
#                                      '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
#                         text=[f'{v}' if v > 0 else '0' for v in current_bands],
#                         textposition='auto',
#                         name="ASIC頻帶功率"
#                     ))
#
#                 fig.update_layout(
#                     title="ASIC EEG 8頻帶功率分布",
#                     xaxis_title="頻帶",
#                     yaxis_title="功率值",
#                     yaxis_range=[0, 100],
#                     height=300,
#                     margin=dict(l=30, r=15, t=30, b=30),
#                     plot_bgcolor='white',
#                     showlegend=False
#                 )
#
#                 return fig
#
#             except Exception as e:
#                 logger.error(f"Error in update_asic_bands_chart: {e}")
#                 return go.Figure()
#
#         @self.app.callback(
#             Output("sensor-display", "children"),
#             Input("interval", "n_intervals")
#         )
#         def update_sensor_display(n):
#             """更新感測器顯示"""
#             try:
#                 sensor_data = self.data_buffer.get_sensor_data()
#
#                 display_text = f"""
# 溫度: {sensor_data['temperature']:.1f}°C
# 濕度: {sensor_data['humidity']:.1f}%
# 光線: {sensor_data['light']}
# 更新: {datetime.now().strftime('%H:%M:%S')}
#                 """.strip()
#
#                 return display_text
#
#             except Exception as e:
#                 return f"感測器錯誤: {e}"
#
#         @self.app.callback(
#             Output("recording-status", "children"),
#             [Input("start-recording-btn", "n_clicks"),
#              Input("stop-recording-btn", "n_clicks"),
#              Input("interval", "n_intervals")],
#             prevent_initial_call=True
#         )
#         def handle_recording_control(start_clicks, stop_clicks, n):
#             """處理錄音控制"""
#             if not self.audio_recorder:
#                 return "❌ 音頻錄製器未初始化"
#
#             try:
#                 # 檢查音頻模組是否可用
#                 status = self.audio_recorder.get_recording_status()
#                 if not status.get('audio_available', False):
#                     return "❌ 音頻模組未安裝 (pip install sounddevice scipy)"
#
#                 ctx = callback_context
#                 if not ctx.triggered:
#                     # 定期狀態更新
#                     if status['is_recording']:
#                         elapsed = status['elapsed_time']
#                         group_id = status['current_group_id'] or "未知"
#                         return f"🔴 錄音中... ({elapsed:.0f}秒) | 群組ID: {group_id}"
#                     else:
#                         device_info = self.audio_recorder.get_device_info()
#                         if device_info.get('available', False) and 'error' not in device_info:
#                             device_name = device_info.get('name', '未知設備')
#                             return f"⚪ 待機中 | 設備: {device_name}"
#                         else:
#                             error_msg = device_info.get('error', '未知錯誤')
#                             return f"⚠️ 設備錯誤: {error_msg}"
#
#                 button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#
#                 if button_id == "start-recording-btn" and start_clicks:
#                     if not status['is_recording']:
#                         group_id = str(uuid.uuid4())[:8]
#                         success = self.audio_recorder.start_recording(group_id)
#                         if success:
#                             return f"🔴 錄音開始 | 群組ID: {group_id}"
#                         else:
#                             return "❌ 錄音啟動失敗 - 請檢查音頻設備"
#                     else:
#                         return "⚠️ 已在錄音中"
#
#                 elif button_id == "stop-recording-btn" and stop_clicks:
#                     if status['is_recording']:
#                         filename = self.audio_recorder.stop_recording(self.db_writer)
#                         if filename:
#                             return f"✅ 錄音已停止並儲存: {os.path.basename(filename)}"
#                         else:
#                             return "⚠️ 錄音停止，但儲存失敗"
#                     else:
#                         return "⚠️ 目前沒有錄音"
#
#                 return "⚪ 待機中"
#
#             except Exception as e:
#                 logger.error(f"Error in handle_recording_control: {e}")
#                 return f"❌ 錄音控制錯誤: {str(e)}"
#
#         @self.app.callback(
#             [Output("performance-status", "children"),
#              Output("interval", "interval")],
#             Input("interval", "n_intervals")
#         )
#         def update_performance_status(n):
#             """更新效能狀態"""
#             try:
#                 current_time = time.time()
#
#                 # 系統效能
#                 cpu_usage = psutil.cpu_percent(interval=None)
#                 memory_usage = psutil.virtual_memory().percent
#
#                 # 數據狀態
#                 data, timestamps = self.data_buffer.get_data()
#                 latency = (time.time() - timestamps[-1]) * 1000 if len(timestamps) > 0 else 0
#
#                 # 信號品質
#                 cognitive_data = self.data_buffer.get_cognitive_data()
#                 signal_quality = cognitive_data['signal_quality']
#
#                 # 效能統計
#                 avg_render = self.performance_monitor['avg_render_time'] * 1000
#
#                 status_text = (
#                     f"CPU: {cpu_usage:.1f}% | "
#                     f"Memory: {memory_usage:.1f}% | "
#                     f"Latency: {latency:.1f}ms | "
#                     f"Render: {avg_render:.1f}ms | "
#                     f"Signal: {signal_quality} | "
#                     f"Updates: {n}"
#                 )
#
#                 return status_text, UI_CONFIG['update_interval']
#
#             except Exception as e:
#                 return f"Status Error: {e}", UI_CONFIG['update_interval']
#
#     def run(self, host='0.0.0.0', port=8052, debug=False):
#         """執行Dash應用程式"""
#         logger.info(f"🚀 Starting EEG Dashboard on http://{host}:{port}")
#         self.app.run(host=host, port=port, debug=debug, use_reloader=False)