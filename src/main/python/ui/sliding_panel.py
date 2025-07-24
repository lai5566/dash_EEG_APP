"""滑動面板組件 - 整合管理頁面功能的左側滑動面板"""

import os
import time
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import dash
from dash import dcc, html, Input, Output, State, callback_context
from mutagen import File as MutagenFile

from services.database_service import EnhancedDatabaseWriter
from ui.session_history_page import SessionHistoryPage

logger = logging.getLogger(__name__)


class SlidingPanel:
    """左側滑動面板組件，整合受試者註冊和音效上傳功能"""
    
    def __init__(self, db_writer: EnhancedDatabaseWriter):
        self.db_writer = db_writer
        self.session_history_page = SessionHistoryPage(db_writer)
        
    def create_panel_layout(self):
        """創建滑動面板佈局"""
        return html.Div([
            # 滑動面板容器
            html.Div([
                # 可見的邊緣觸發器
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chevron-right", id="panel-icon",
                               style={'fontSize': '18px', 'color': 'white', 
                                     'transition': 'transform 0.3s ease'}),
                        html.Div("管理", style={'writingMode': 'vertical-rl', 
                                             'textOrientation': 'mixed',
                                             'fontSize': '14px', 'fontWeight': 'bold',
                                             'color': 'white', 'marginTop': '10px',
                                             'letterSpacing': '2px'})
                    ], style={'display': 'flex', 'flexDirection': 'column', 
                             'alignItems': 'center', 'padding': '15px 0'})
                ], id="panel-trigger", 
                   style={
                       'position': 'fixed', 'left': '0', 'top': '50%',
                       'transform': 'translateY(-50%)', 'width': '8vw', 'height': '120px',
                       'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                       'borderRadius': '0 15px 15px 0', 'cursor': 'pointer',
                       'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                       'boxShadow': '2px 0 10px rgba(0,0,0,0.3)', 'zIndex': '1001',
                       'transition': 'all 0.3s ease', 'opacity': '0.9'
                   }),
                
                # 滑動面板主體
                html.Div([
                    # 面板標題和關閉按鈕
                    html.Div([
                        html.H2("EEG 管理中心", 
                               style={'color': '#2c3e50', 'margin': '0', 'fontSize': '24px'}),
                        html.Button("×", id="close-panel-btn",
                                   style={'background': 'none', 'border': 'none', 
                                         'fontSize': '30px', 'cursor': 'pointer',
                                         'color': '#666', 'lineHeight': '1'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 
                             'alignItems': 'center', 'padding': '20px 30px',
                             'borderBottom': '2px solid #ecf0f1'}),
                    
                    # 面板內容區域
                    html.Div([
                        # 標籤切換
                        html.Div([
                            html.Button("👤 受試者管理", id="subjects-tab-btn", 
                                       className="tab-button active",
                                       style={'flex': '1', 'padding': '12px 16px', 
                                             'border': 'none', 'backgroundColor': '#3498db',
                                             'color': 'white', 'fontSize': '13px', 
                                             'cursor': 'pointer', 'fontWeight': 'bold',
                                             'borderRadius': '8px 0 0 8px'}),
                            html.Button("🎵 音效管理", id="sounds-tab-btn",
                                       className="tab-button",
                                       style={'flex': '1', 'padding': '12px 16px',
                                             'border': 'none', 'backgroundColor': '#95a5a6',
                                             'color': 'white', 'fontSize': '13px',
                                             'cursor': 'pointer', 'fontWeight': 'bold',
                                             'borderRadius': '0'}),
                            html.Button("📊 歷史記錄", id="history-tab-btn",
                                       className="tab-button",
                                       style={'flex': '1', 'padding': '12px 16px',
                                             'border': 'none', 'backgroundColor': '#95a5a6',
                                             'color': 'white', 'fontSize': '13px',
                                             'cursor': 'pointer', 'fontWeight': 'bold',
                                             'borderRadius': '0 8px 8px 0'})
                        ], style={'display': 'flex', 'margin': '20px 30px', 'gap': '2px'}),
                        
                        # 標籤內容區域
                        html.Div([
                            # 受試者管理標籤內容
                            html.Div([
                                # 受試者新增表單
                                html.Div([
                                    html.H4("新增受試者", 
                                           style={'color': '#34495e', 'marginBottom': '20px',
                                                 'borderBottom': '2px solid #3498db', 
                                                 'paddingBottom': '10px'}),
                                    
                                    # 受試者ID
                                    html.Div([
                                        html.Label("受試者ID:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Input(
                                            id="subject-id-input-panel",
                                            type="text",
                                            placeholder="例如: S001, P001",
                                            style={'width': '100%', 'padding': '14px 16px', 
                                                  'marginBottom': '20px', 'fontSize': '15px',
                                                  'border': '2px solid #e2e8f0', 
                                                  'borderRadius': '12px', 'background': 'white',
                                                  'transition': 'all 0.3s ease', 'fontWeight': '500'}
                                        ),
                                    ]),
                                    
                                    # 性別和年齡 (並排)
                                    html.Div([
                                        html.Div([
                                            html.Label("性別:", 
                                                      style={'fontWeight': '600', 'marginBottom': '8px',
                                                            'display': 'block', 'color': '#1e293b', 
                                                            'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                            dcc.Dropdown(
                                                id="subject-gender-dropdown-panel",
                                                options=[
                                                    {'label': '男性', 'value': 'M'},
                                                    {'label': '女性', 'value': 'F'},
                                                    {'label': '其他', 'value': 'Other'}
                                                ],
                                                placeholder="選擇性別",
                                                style={'marginBottom': '15px'}
                                            ),
                                        ], style={'flex': '1', 'marginRight': '10px'}),
                                        
                                        html.Div([
                                            html.Label("年齡:", 
                                                      style={'fontWeight': '600', 'marginBottom': '8px',
                                                            'display': 'block', 'color': '#1e293b', 
                                                            'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                            dcc.Input(
                                                id="subject-age-input-panel",
                                                type="number",
                                                min=0, max=150,
                                                placeholder="年齡",
                                                style={'width': '100%', 'padding': '14px 16px', 
                                                      'marginBottom': '20px', 'fontSize': '15px',
                                                      'border': '2px solid #e2e8f0', 
                                                      'borderRadius': '12px', 'background': 'white',
                                                      'transition': 'all 0.3s ease', 'fontWeight': '500'}
                                            ),
                                        ], style={'flex': '1', 'marginLeft': '10px'}),
                                    ], style={'display': 'flex'}),
                                    
                                    # 研究者姓名
                                    html.Div([
                                        html.Label("研究者姓名:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Input(
                                            id="researcher-name-input-panel",
                                            type="text",
                                            placeholder="研究者姓名",
                                            style={'width': '100%', 'padding': '14px 16px', 
                                                  'marginBottom': '20px', 'fontSize': '15px',
                                                  'border': '2px solid #e2e8f0', 
                                                  'borderRadius': '12px', 'background': 'white',
                                                  'transition': 'all 0.3s ease', 'fontWeight': '500'}
                                        ),
                                    ]),
                                    
                                    # 備註
                                    html.Div([
                                        html.Label("備註:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Textarea(
                                            id="subject-notes-input-panel",
                                            placeholder="受試者備註資訊...",
                                            style={'width': '100%', 'height': '90px', 'padding': '14px 16px',
                                                  'border': '2px solid #e2e8f0', 'borderRadius': '12px',
                                                  'marginBottom': '25px', 'resize': 'vertical', 'background': 'white',
                                                  'fontSize': '15px', 'transition': 'all 0.3s ease', 'fontWeight': '500'}
                                        ),
                                    ]),
                                    
                                    # 新增按鈕
                                    html.Button(
                                        "新增受試者",
                                        id="add-subject-btn-panel",
                                        style={'backgroundColor': '#27ae60', 'color': 'white',
                                              'border': 'none', 'padding': '15px 30px',
                                              'borderRadius': '8px', 'cursor': 'pointer',
                                              'width': '100%', 'marginBottom': '20px',
                                              'fontSize': '16px', 'fontWeight': 'bold',
                                              'transition': 'background-color 0.3s ease'}
                                    ),
                                    
                                    # 操作狀態顯示
                                    html.Div(id="subject-status-panel", 
                                            style={'marginBottom': '0', 'padding': '16px',
                                                  'borderRadius': '12px', 'textAlign': 'center',
                                                  'fontSize': '14px', 'fontWeight': '600'}),
                                    
                                ], style={'background': 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
                                         'padding': '30px', 'borderRadius': '16px', 'marginBottom': '25px',
                                         'border': '1px solid rgba(148, 163, 184, 0.2)',
                                         'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.05)'}),
                                
                                # 現有受試者列表
                                html.Div([
                                    html.H4("現有受試者", 
                                           style={'color': '#34495e', 'marginBottom': '15px',
                                                 'borderBottom': '2px solid #3498db', 
                                                 'paddingBottom': '10px'}),
                                    html.Div(id="subjects-list-panel", 
                                            style={'maxHeight': '300px', 'overflowY': 'auto',
                                                  'border': '1px solid #ecf0f1', 'borderRadius': '8px',
                                                  'padding': '10px'}),
                                ])
                                
                            ], id="subjects-content", style={'display': 'block'}),
                            
                            # 音效管理標籤內容
                            html.Div([
                                # 音效上傳表單
                                html.Div([
                                    html.H4("上傳音效檔案", 
                                           style={'color': '#34495e', 'marginBottom': '20px',
                                                 'borderBottom': '2px solid #e74c3c', 
                                                 'paddingBottom': '10px'}),
                                    
                                    # 音效名稱
                                    html.Div([
                                        html.Label("音效名稱:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Input(
                                            id="sound-name-input-panel",
                                            type="text",
                                            placeholder="例如: 森林白噪音, 海浪聲",
                                            style={'width': '100%', 'padding': '14px 16px', 
                                                  'marginBottom': '20px', 'fontSize': '15px',
                                                  'border': '2px solid #e2e8f0', 
                                                  'borderRadius': '12px', 'background': 'white',
                                                  'transition': 'all 0.3s ease', 'fontWeight': '500'}
                                        ),
                                    ]),
                                    
                                    # 音效類別
                                    html.Div([
                                        html.Label("音效類別:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Dropdown(
                                            id="sound-category-dropdown-panel",
                                            options=[
                                                {'label': '自然聲音', 'value': '自然聲音'},
                                                {'label': '白噪音', 'value': '白噪音'},
                                                {'label': '粉紅噪音', 'value': '粉紅噪音'},
                                                {'label': '音樂', 'value': '音樂'},
                                                {'label': '環境音', 'value': '環境音'},
                                                {'label': '其他', 'value': '其他'}
                                            ],
                                            placeholder="選擇音效類別",
                                            style={'marginBottom': '15px'}
                                        ),
                                    ]),
                                    
                                    # 檔案上傳
                                    html.Div([
                                        html.Label("選擇音效檔案:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Upload(
                                            id="sound-upload-panel",
                                            children=html.Div([
                                                html.Div("📁", style={'fontSize': '32px', 'marginBottom': '10px'}),
                                                html.Div("拖拽檔案到此處或點擊選擇", 
                                                        style={'fontSize': '14px', 'fontWeight': 'bold'}),
                                                html.Div("支援格式: .wav, .mp3, .flac", 
                                                        style={'fontSize': '12px', 'color': '#7f8c8d', 
                                                              'marginTop': '5px'})
                                            ]),
                                            style={
                                                'width': '100%', 'height': '140px',
                                                'lineHeight': '140px', 'borderWidth': '3px',
                                                'borderStyle': 'dashed', 'borderRadius': '16px',
                                                'textAlign': 'center', 'margin': '15px 0',
                                                'cursor': 'pointer', 'borderColor': '#f472b6',
                                                'background': 'linear-gradient(135deg, #fef7ff 0%, #fae8ff 100%)', 
                                                'color': '#831843', 'transition': 'all 0.3s ease',
                                                'boxShadow': '0 4px 15px rgba(244, 114, 182, 0.1)'
                                            },
                                            multiple=False,
                                            accept=".wav,.mp3,.flac"
                                        ),
                                    ]),
                                    
                                    # 音效描述
                                    html.Div([
                                        html.Label("音效描述:", 
                                                  style={'fontWeight': '600', 'marginBottom': '8px',
                                                        'display': 'block', 'color': '#1e293b', 
                                                        'fontSize': '14px', 'letterSpacing': '0.5px'}),
                                        dcc.Textarea(
                                            id="sound-description-input-panel",
                                            placeholder="音效描述和用途說明...",
                                            style={'width': '100%', 'height': '90px', 'padding': '14px 16px',
                                                  'border': '2px solid #e2e8f0', 'borderRadius': '12px',
                                                  'marginBottom': '25px', 'resize': 'vertical', 'background': 'white',
                                                  'fontSize': '15px', 'transition': 'all 0.3s ease', 'fontWeight': '500'}
                                        ),
                                    ]),
                                    
                                    # 上傳按鈕
                                    html.Button(
                                        "上傳音效",
                                        id="upload-sound-btn-panel",
                                        style={'backgroundColor': '#e67e22', 'color': 'white',
                                              'border': 'none', 'padding': '15px 30px',
                                              'borderRadius': '8px', 'cursor': 'pointer',
                                              'width': '100%', 'marginBottom': '20px',
                                              'fontSize': '16px', 'fontWeight': 'bold',
                                              'transition': 'background-color 0.3s ease'}
                                    ),
                                    
                                    # 上傳狀態顯示
                                    html.Div(id="sound-status-panel", 
                                            style={'marginBottom': '0', 'padding': '16px',
                                                  'borderRadius': '12px', 'textAlign': 'center',
                                                  'fontSize': '14px', 'fontWeight': '600'}),
                                    
                                ], style={'background': 'linear-gradient(135deg, #fef3f2 0%, #fee2e2 100%)',
                                         'padding': '30px', 'borderRadius': '16px', 'marginBottom': '25px',
                                         'border': '1px solid rgba(239, 68, 68, 0.2)',
                                         'boxShadow': '0 4px 20px rgba(239, 68, 68, 0.08)'}),
                                
                                # 現有音效列表
                                html.Div([
                                    html.H4("現有音效", 
                                           style={'color': '#34495e', 'marginBottom': '15px',
                                                 'borderBottom': '2px solid #e74c3c', 
                                                 'paddingBottom': '10px'}),
                                    html.Div(id="sounds-list-panel", 
                                            style={'maxHeight': '300px', 'overflowY': 'auto',
                                                  'border': '1px solid #ecf0f1', 'borderRadius': '8px',
                                                  'padding': '10px'}),
                                ])
                                
                            ], id="sounds-content", style={'display': 'none'}),
                            
                            # Session 歷史記錄標籤內容
                            html.Div([
                                self.session_history_page.create_layout()
                            ], id="history-content", style={'display': 'none'})
                            
                        ], style={'padding': '0 30px 30px', 'height': 'calc(100vh - 200px)', 
                                 'overflowY': 'auto'})
                        
                    ])
                    
                ], id="sliding-panel-content",
                   style={
                       'position': 'fixed', 'left': '-100vw', 'top': '0',
                       'width': '100vw', 'height': '100vh', 'backgroundColor': 'white',
                       'zIndex': '1000', 'transition': 'left 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                       'boxShadow': '2px 0 20px rgba(0,0,0,0.3)', 'overflow': 'hidden'
                   })
                
            ]),
            
            # 背景遮罩
            html.Div(id="panel-overlay",
                    style={
                        'position': 'fixed', 'top': '0', 'left': '0',
                        'width': '100vw', 'height': '100vh',
                        'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '999',
                        'display': 'none', 'transition': 'opacity 0.3s ease'
                    }),
            
            # 數據存儲
            dcc.Store(id="subjects-store-panel", data=[]),
            dcc.Store(id="sounds-store-panel", data=[]),
            dcc.Store(id="panel-state", data={"is_open": False})
            
        ])
    
    def register_callbacks(self, app):
        """註冊滑動面板的回調函數"""
        
        # 面板開關控制
        @app.callback(
            [Output("sliding-panel-content", "style"),
             Output("panel-overlay", "style"),
             Output("panel-icon", "style"),
             Output("panel-state", "data")],
            [Input("panel-trigger", "n_clicks"),
             Input("close-panel-btn", "n_clicks"),
             Input("panel-overlay", "n_clicks")],
            [State("panel-state", "data")],
            prevent_initial_call=True
        )
        def toggle_panel(trigger_clicks, close_clicks, overlay_clicks, panel_state):
            """切換面板開關狀態"""
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            is_open = panel_state.get("is_open", False)
            
            if trigger_id == "panel-trigger" or \
               (trigger_id in ["close-panel-btn", "panel-overlay"] and is_open):
                new_state = not is_open
                
                if new_state:  # 打開面板
                    panel_style = {
                        'position': 'fixed', 'left': '0', 'top': '0',
                        'width': '90vw', 'height': '100vh', 'backgroundColor': 'white',
                        'zIndex': '1000', 'transition': 'left 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                        'boxShadow': '2px 0 20px rgba(0,0,0,0.3)', 'overflow': 'hidden'
                    }
                    overlay_style = {
                        'position': 'fixed', 'top': '0', 'left': '0',
                        'width': '100vw', 'height': '100vh',
                        'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '999',
                        'display': 'block', 'transition': 'opacity 0.3s ease'
                    }
                    icon_style = {
                        'fontSize': '18px', 'color': 'white', 
                        'transition': 'transform 0.3s ease',
                        'transform': 'rotate(180deg)'
                    }
                else:  # 關閉面板
                    panel_style = {
                        'position': 'fixed', 'left': '-100vw', 'top': '0',
                        'width': '100vw', 'height': '100vh', 'backgroundColor': 'white',
                        'zIndex': '1000', 'transition': 'left 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
                        'boxShadow': '2px 0 20px rgba(0,0,0,0.3)', 'overflow': 'hidden'
                    }
                    overlay_style = {
                        'position': 'fixed', 'top': '0', 'left': '0',
                        'width': '100vw', 'height': '100vh',
                        'backgroundColor': 'rgba(0,0,0,0.5)', 'zIndex': '999',
                        'display': 'none', 'transition': 'opacity 0.3s ease'
                    }
                    icon_style = {
                        'fontSize': '18px', 'color': 'white', 
                        'transition': 'transform 0.3s ease',
                        'transform': 'rotate(0deg)'
                    }
                
                return panel_style, overlay_style, icon_style, {"is_open": new_state}
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # 標籤切換控制
        @app.callback(
            [Output("subjects-tab-btn", "style"),
             Output("sounds-tab-btn", "style"),
             Output("history-tab-btn", "style"),
             Output("subjects-content", "style"),
             Output("sounds-content", "style"),
             Output("history-content", "style")],
            [Input("subjects-tab-btn", "n_clicks"),
             Input("sounds-tab-btn", "n_clicks"),
             Input("history-tab-btn", "n_clicks")],
            prevent_initial_call=True
        )
        def switch_tabs(subjects_clicks, sounds_clicks, history_clicks):
            """切換標籤內容"""
            ctx = callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == "subjects-tab-btn":
                # 切換到受試者標籤
                subjects_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': 'none', 
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                    'color': 'white', 'fontSize': '14px', 'cursor': 'pointer', 
                    'fontWeight': '600', 'borderRadius': '12px', 'margin': '0 4px',
                    'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)', 'transition': 'all 0.3s ease'
                }
                sounds_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': '2px solid #e8ecf0', 
                    'backgroundColor': 'white', 'color': '#64748b', 'fontSize': '14px',
                    'cursor': 'pointer', 'fontWeight': '600', 'borderRadius': '12px', 
                    'margin': '0 4px', 'transition': 'all 0.3s ease'
                }
                history_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': '2px solid #e8ecf0', 
                    'backgroundColor': 'white', 'color': '#64748b', 'fontSize': '14px',
                    'cursor': 'pointer', 'fontWeight': '600', 'borderRadius': '12px', 
                    'margin': '0 4px', 'transition': 'all 0.3s ease'
                }
                subjects_content_style = {'display': 'block'}
                sounds_content_style = {'display': 'none'}
                history_content_style = {'display': 'none'}
                
            elif trigger_id == "sounds-tab-btn":
                # 切換到音效標籤
                subjects_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': '2px solid #e8ecf0', 
                    'backgroundColor': 'white', 'color': '#64748b', 'fontSize': '14px', 
                    'cursor': 'pointer', 'fontWeight': '600', 'borderRadius': '12px', 
                    'margin': '0 4px', 'transition': 'all 0.3s ease'
                }
                sounds_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': 'none', 
                    'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
                    'color': 'white', 'fontSize': '14px', 'cursor': 'pointer', 
                    'fontWeight': '600', 'borderRadius': '12px', 'margin': '0 4px',
                    'boxShadow': '0 4px 15px rgba(245, 87, 108, 0.3)', 'transition': 'all 0.3s ease'
                }
                history_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': '2px solid #e8ecf0', 
                    'backgroundColor': 'white', 'color': '#64748b', 'fontSize': '14px',
                    'cursor': 'pointer', 'fontWeight': '600', 'borderRadius': '12px', 
                    'margin': '0 4px', 'transition': 'all 0.3s ease'
                }
                subjects_content_style = {'display': 'none'}
                sounds_content_style = {'display': 'block'}
                history_content_style = {'display': 'none'}
                
            else:  # history-tab-btn
                # 切換到歷史記錄標籤
                subjects_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': '2px solid #e8ecf0', 
                    'backgroundColor': 'white', 'color': '#64748b', 'fontSize': '14px', 
                    'cursor': 'pointer', 'fontWeight': '600', 'borderRadius': '12px', 
                    'margin': '0 4px', 'transition': 'all 0.3s ease'
                }
                sounds_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': '2px solid #e8ecf0', 
                    'backgroundColor': 'white', 'color': '#64748b', 'fontSize': '14px',
                    'cursor': 'pointer', 'fontWeight': '600', 'borderRadius': '12px', 
                    'margin': '0 4px', 'transition': 'all 0.3s ease'
                }
                history_tab_style = {
                    'flex': '1', 'padding': '16px 20px', 'border': 'none', 
                    'background': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', 
                    'color': 'white', 'fontSize': '14px', 'cursor': 'pointer', 
                    'fontWeight': '600', 'borderRadius': '12px', 'margin': '0 4px',
                    'boxShadow': '0 4px 15px rgba(79, 172, 254, 0.3)', 'transition': 'all 0.3s ease'
                }
                subjects_content_style = {'display': 'none'}
                sounds_content_style = {'display': 'none'}
                history_content_style = {'display': 'block'}
            
            return subjects_tab_style, sounds_tab_style, history_tab_style, subjects_content_style, sounds_content_style, history_content_style
        
        # 受試者管理回調
        @app.callback(
            [Output("subject-status-panel", "children"),
             Output("subject-status-panel", "style"),
             Output("subjects-store-panel", "data"),
             Output("subject-id-input-panel", "value"),
             Output("subject-gender-dropdown-panel", "value"),
             Output("subject-age-input-panel", "value"),
             Output("researcher-name-input-panel", "value"),
             Output("subject-notes-input-panel", "value")],
            [Input("add-subject-btn-panel", "n_clicks")],
            [State("subject-id-input-panel", "value"),
             State("subject-gender-dropdown-panel", "value"),
             State("subject-age-input-panel", "value"),
             State("researcher-name-input-panel", "value"),
             State("subject-notes-input-panel", "value")]
        )
        def add_subject_panel(n_clicks, subject_id, gender, age, researcher, notes):
            """新增受試者 (面板版本)"""
            if not n_clicks:
                subjects = self.db_writer.get_subjects()
                return "", {}, subjects, "", None, "", "", ""
            
            # 驗證輸入
            if not subject_id or not subject_id.strip():
                return ("❌ 請輸入受試者ID", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], subject_id, gender, age, researcher, notes)
            
            if not gender:
                return ("❌ 請選擇性別", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], subject_id, gender, age, researcher, notes)
            
            if not age or age < 0 or age > 150:
                return ("❌ 請輸入有效年齡 (0-150)", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], subject_id, gender, age, researcher, notes)
            
            try:
                # 新增受試者到資料庫
                success = self.db_writer.add_subject(
                    subject_id=subject_id.strip(),
                    gender=gender,
                    age=int(age),
                    researcher_name=researcher.strip() if researcher else None,
                    notes=notes.strip() if notes else None
                )
                
                if success:
                    subjects = self.db_writer.get_subjects()
                    return (f"✅ 受試者 {subject_id} 新增成功！", 
                           {'backgroundColor': '#e8f5e8', 'color': '#2e7d32', 'border': '1px solid #66bb6a'},
                           subjects, "", None, "", "", "")
                else:
                    return ("❌ 受試者ID已存在，請使用不同的ID", 
                           {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                           [], subject_id, gender, age, researcher, notes)
                    
            except Exception as e:
                logger.error(f"新增受試者失敗: {e}")
                return (f"❌ 新增失敗: {str(e)}", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], subject_id, gender, age, researcher, notes)
        
        @app.callback(
            Output("subjects-list-panel", "children"),
            [Input("subjects-store-panel", "data")]
        )
        def update_subjects_list_panel(subjects_data):
            """更新受試者列表顯示 (面板版本)"""
            if not subjects_data:
                return html.Div("尚無受試者資料", 
                               style={'textAlign': 'center', 'color': '#7f8c8d', 
                                     'padding': '20px', 'fontStyle': 'italic'})
            
            subjects_cards = []
            for subject in subjects_data:
                card = html.Div([
                    html.Div([
                        html.Div([
                            html.Strong(f"ID: {subject['subject_id']}", 
                                       style={'fontSize': '16px', 'color': '#2c3e50'}),
                            html.Span(f" ({subject['gender']}, {subject['age']}歲)", 
                                     style={'fontSize': '14px', 'color': '#7f8c8d', 'marginLeft': '8px'})
                        ], style={'marginBottom': '8px'}),
                        
                        html.Div([
                            html.Span("👨‍🔬 ", style={'marginRight': '5px'}),
                            html.Span(f"研究者: {subject['researcher_name'] or 'N/A'}", 
                                     style={'fontSize': '13px', 'color': '#34495e'})
                        ], style={'marginBottom': '5px'}),
                        
                        html.Div([
                            html.Span("📅 ", style={'marginRight': '5px'}),
                            html.Span(f"創建: {subject['created_at']}", 
                                     style={'fontSize': '12px', 'color': '#95a5a6'})
                        ], style={'marginBottom': '5px'}),
                        
                        html.Div([
                            html.Span("📝 ", style={'marginRight': '5px'}),
                            html.Span(f"備註: {subject['notes'] or 'N/A'}", 
                                     style={'fontSize': '12px', 'color': '#95a5a6'})
                        ])
                    ])
                ], style={
                    'border': '1px solid #ecf0f1', 'borderRadius': '8px',
                    'padding': '15px', 'marginBottom': '12px', 
                    'backgroundColor': '#fdfdfd', 'transition': 'all 0.2s ease',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
                })
                subjects_cards.append(card)
            
            return subjects_cards
        
        # 音效管理回調
        @app.callback(
            [Output("sound-status-panel", "children"),
             Output("sound-status-panel", "style"),
             Output("sounds-store-panel", "data"),
             Output("sound-name-input-panel", "value"),
             Output("sound-category-dropdown-panel", "value"),
             Output("sound-description-input-panel", "value")],
            [Input("upload-sound-btn-panel", "n_clicks")],
            [State("sound-name-input-panel", "value"),
             State("sound-category-dropdown-panel", "value"),
             State("sound-upload-panel", "contents"),
             State("sound-upload-panel", "filename"),
             State("sound-description-input-panel", "value")]
        )
        def upload_sound_panel(n_clicks, sound_name, category, contents, filename, description):
            """上傳音效檔案 (面板版本)"""
            if not n_clicks:
                sounds = self.db_writer.get_ambient_sounds()
                return "", {}, sounds, "", None, ""
            
            # 驗證輸入
            if not sound_name or not sound_name.strip():
                return ("❌ 請輸入音效名稱", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], sound_name, category, description)
            
            if not category:
                return ("❌ 請選擇音效類別", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], sound_name, category, description)
            
            if not contents or not filename:
                return ("❌ 請選擇音效檔案", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], sound_name, category, description)
            
            try:
                # 解析上傳的檔案
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # 創建音效存儲目錄
                sounds_dir = "src/main/resources/sounds"
                os.makedirs(sounds_dir, exist_ok=True)
                
                # 生成唯一檔名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_extension = os.path.splitext(filename)[1]
                unique_filename = f"{timestamp}_{filename}"
                file_path = os.path.join(sounds_dir, unique_filename)
                
                # 儲存檔案
                with open(file_path, 'wb') as f:
                    f.write(decoded)
                
                # 獲取音頻檔案資訊
                duration = 0.0
                file_size = len(decoded)
                
                try:
                    # 使用mutagen獲取音頻資訊
                    audio_file = MutagenFile(file_path)
                    if audio_file is not None:
                        duration = audio_file.info.length
                except Exception as e:
                    logger.warning(f"無法獲取音頻檔案資訊: {e}")
                    duration = 60.0  # 預設60秒
                
                # 新增到資料庫
                sound_id = self.db_writer.add_ambient_sound(
                    sound_name=sound_name.strip(),
                    style_category=category,
                    filename=unique_filename,
                    file_path=file_path,
                    duration_seconds=duration,
                    description=description.strip() if description else None,
                    file_size_bytes=file_size
                )
                
                if sound_id:
                    sounds = self.db_writer.get_ambient_sounds()
                    return (f"✅ 音效 '{sound_name}' 上傳成功！", 
                           {'backgroundColor': '#e8f5e8', 'color': '#2e7d32', 'border': '1px solid #66bb6a'},
                           sounds, "", None, "")
                else:
                    # 刪除已上傳的檔案
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return ("❌ 音效名稱已存在，請使用不同的名稱", 
                           {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                           [], sound_name, category, description)
                    
            except Exception as e:
                logger.error(f"上傳音效失敗: {e}")
                return (f"❌ 上傳失敗: {str(e)}", 
                       {'backgroundColor': '#ffebee', 'color': '#c62828', 'border': '1px solid #ef5350'},
                       [], sound_name, category, description)
        
        @app.callback(
            Output("sounds-list-panel", "children"),
            [Input("sounds-store-panel", "data")]
        )
        def update_sounds_list_panel(sounds_data):
            """更新音效列表顯示 (面板版本)"""
            if not sounds_data:
                return html.Div("尚無音效檔案", 
                               style={'textAlign': 'center', 'color': '#7f8c8d', 
                                     'padding': '20px', 'fontStyle': 'italic'})
            
            sounds_cards = []
            for sound in sounds_data:
                card = html.Div([
                    html.Div([
                        html.Div([
                            html.Span("🎵 ", style={'fontSize': '18px', 'marginRight': '8px'}),
                            html.Strong(f"{sound['sound_name']}", 
                                       style={'fontSize': '16px', 'color': '#2c3e50'}),
                            html.Span(f" ({sound['style_category']})", 
                                     style={'fontSize': '14px', 'color': '#e74c3c', 
                                           'marginLeft': '8px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '8px'}),
                        
                        html.Div([
                            html.Span("📁 ", style={'marginRight': '5px'}),
                            html.Span(f"檔案: {sound['filename']}", 
                                     style={'fontSize': '13px', 'color': '#34495e'})
                        ], style={'marginBottom': '5px'}),
                        
                        html.Div([
                            html.Span("⏱️ ", style={'marginRight': '5px'}),
                            html.Span(f"時長: {sound['duration_seconds']:.1f}秒", 
                                     style={'fontSize': '13px', 'color': '#34495e'}),
                            html.Span(f" | ID: {sound['id']}", 
                                     style={'fontSize': '12px', 'color': '#95a5a6', 'marginLeft': '15px'})
                        ], style={'marginBottom': '5px'}),
                        
                        html.Div([
                            html.Span("📝 ", style={'marginRight': '5px'}),
                            html.Span(f"描述: {sound['description'] or 'N/A'}", 
                                     style={'fontSize': '12px', 'color': '#95a5a6'})
                        ])
                    ])
                ], style={
                    'border': '1px solid #ecf0f1', 'borderRadius': '8px',
                    'padding': '15px', 'marginBottom': '12px', 
                    'backgroundColor': '#fdfdfd', 'transition': 'all 0.2s ease',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
                })
                sounds_cards.append(card)
            
            return sounds_cards
        
        # 註冊 Session 歷史頁面的回調
        self.session_history_page.register_callbacks(app)