"""EEG數據管理頁面"""

import os
import time
import base64
import io
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
from mutagen import File as MutagenFile

from services.database_service import EnhancedDatabaseWriter

logger = logging.getLogger(__name__)


class ManagementPage:
    """EEG數據管理頁面類"""
    
    def __init__(self, db_writer: EnhancedDatabaseWriter):
        self.db_writer = db_writer
        
    def create_layout(self):
        """創建管理頁面佈局"""
        return html.Div([
            # 頁面標題
            html.Div([
                html.H1("EEG 數據管理中心", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
                
                # 導航按鈕
                html.Div([
                    html.Button("返回實驗頁面", id="back-to-dashboard-btn",
                               style={'backgroundColor': '#007bff', 'color': 'white',
                                     'border': 'none', 'padding': '10px 20px',
                                     'borderRadius': '4px', 'cursor': 'pointer',
                                     'fontSize': '14px', 'textDecoration': 'none'}),
                ], style={'textAlign': 'center', 'marginBottom': '30px'}),
            ]),
            
            # 主要內容區域
            html.Div([
                # 左側：受試者管理
                html.Div([
                    html.Div([
                        html.H3("👤 受試者管理", 
                               style={'color': '#34495e', 'marginBottom': '20px'}),
                        
                        # 受試者新增表單
                        html.Div([
                            html.H5("新增受試者", style={'color': '#555', 'marginBottom': '15px'}),
                            
                            # 受試者ID
                            html.Div([
                                html.Label("受試者ID:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Input(
                                    id="subject-id-input",
                                    type="text",
                                    placeholder="例如: S001, P001",
                                    style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                                          'border': '1px solid #ddd', 'borderRadius': '4px'}
                                ),
                            ]),
                            
                            # 性別選擇
                            html.Div([
                                html.Label("性別:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id="subject-gender-dropdown",
                                    options=[
                                        {'label': '男性 (M)', 'value': 'M'},
                                        {'label': '女性 (F)', 'value': 'F'},
                                        {'label': '其他 (Other)', 'value': 'Other'}
                                    ],
                                    placeholder="選擇性別",
                                    style={'marginBottom': '10px'}
                                ),
                            ]),
                            
                            # 年齡
                            html.Div([
                                html.Label("年齡:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Input(
                                    id="subject-age-input",
                                    type="number",
                                    min=0, max=150,
                                    placeholder="年齡",
                                    style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                                          'border': '1px solid #ddd', 'borderRadius': '4px'}
                                ),
                            ]),
                            
                            # 研究者姓名
                            html.Div([
                                html.Label("研究者姓名:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Input(
                                    id="researcher-name-input",
                                    type="text",
                                    placeholder="研究者姓名",
                                    style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                                          'border': '1px solid #ddd', 'borderRadius': '4px'}
                                ),
                            ]),
                            
                            # 備註
                            html.Div([
                                html.Label("備註:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Textarea(
                                    id="subject-notes-input",
                                    placeholder="受試者備註資訊...",
                                    style={'width': '100%', 'height': '80px', 'padding': '8px',
                                          'border': '1px solid #ddd', 'borderRadius': '4px',
                                          'marginBottom': '15px', 'resize': 'vertical'}
                                ),
                            ]),
                            
                            # 新增按鈕
                            html.Button(
                                "新增受試者",
                                id="add-subject-btn",
                                style={'backgroundColor': '#28a745', 'color': 'white',
                                      'border': 'none', 'padding': '10px 20px',
                                      'borderRadius': '4px', 'cursor': 'pointer',
                                      'width': '100%', 'marginBottom': '15px'}
                            ),
                            
                            # 操作狀態顯示
                            html.Div(id="subject-status", 
                                    style={'marginTop': '10px', 'padding': '10px',
                                          'borderRadius': '4px', 'textAlign': 'center'}),
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px',
                                 'marginBottom': '20px'}),
                        
                        # 現有受試者列表
                        html.Div([
                            html.H5("現有受試者", style={'color': '#555', 'marginBottom': '15px'}),
                            html.Div(id="subjects-list", style={'maxHeight': '300px', 'overflowY': 'auto'}),
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px'}),
                        
                    ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px',
                             'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'height': 'fit-content'}),
                ], style={'flex': '1', 'marginRight': '20px'}),
                
                # 右側：環境音效管理
                html.Div([
                    html.Div([
                        html.H3("🎵 環境音效管理", 
                               style={'color': '#34495e', 'marginBottom': '20px'}),
                        
                        # 音效上傳表單
                        html.Div([
                            html.H5("上傳音效檔案", style={'color': '#555', 'marginBottom': '15px'}),
                            
                            # 音效名稱
                            html.Div([
                                html.Label("音效名稱:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Input(
                                    id="sound-name-input",
                                    type="text",
                                    placeholder="例如: 森林白噪音, 海浪聲",
                                    style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                                          'border': '1px solid #ddd', 'borderRadius': '4px'}
                                ),
                            ]),
                            
                            # 音效類別
                            html.Div([
                                html.Label("音效類別:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id="sound-category-dropdown",
                                    options=[
                                        {'label': '自然聲音', 'value': '自然聲音'},
                                        {'label': '白噪音', 'value': '白噪音'},
                                        {'label': '粉紅噪音', 'value': '粉紅噪音'},
                                        {'label': '音樂', 'value': '音樂'},
                                        {'label': '環境音', 'value': '環境音'},
                                        {'label': '其他', 'value': '其他'}
                                    ],
                                    placeholder="選擇音效類別",
                                    style={'marginBottom': '10px'}
                                ),
                            ]),
                            
                            # 檔案上傳
                            html.Div([
                                html.Label("選擇音效檔案:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Upload(
                                    id="sound-upload",
                                    children=html.Div([
                                        html.I(className="fas fa-cloud-upload-alt", 
                                              style={'fontSize': '24px', 'marginBottom': '10px'}),
                                        html.Br(),
                                        "拖拽檔案到此處或點擊選擇",
                                        html.Br(),
                                        html.Small("支援格式: .wav, .mp3, .flac", 
                                                  style={'color': '#6c757d'})
                                    ]),
                                    style={
                                        'width': '100%', 'height': '100px',
                                        'lineHeight': '100px', 'borderWidth': '2px',
                                        'borderStyle': 'dashed', 'borderRadius': '8px',
                                        'textAlign': 'center', 'margin': '10px 0',
                                        'cursor': 'pointer', 'borderColor': '#007bff',
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    multiple=False,
                                    accept=".wav,.mp3,.flac"
                                ),
                            ]),
                            
                            # 音效描述
                            html.Div([
                                html.Label("音效描述:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                                dcc.Textarea(
                                    id="sound-description-input",
                                    placeholder="音效描述和用途說明...",
                                    style={'width': '100%', 'height': '80px', 'padding': '8px',
                                          'border': '1px solid #ddd', 'borderRadius': '4px',
                                          'marginBottom': '15px', 'resize': 'vertical'}
                                ),
                            ]),
                            
                            # 上傳按鈕
                            html.Button(
                                "上傳音效",
                                id="upload-sound-btn",
                                style={'backgroundColor': '#17a2b8', 'color': 'white',
                                      'border': 'none', 'padding': '10px 20px',
                                      'borderRadius': '4px', 'cursor': 'pointer',
                                      'width': '100%', 'marginBottom': '15px'}
                            ),
                            
                            # 上傳狀態顯示
                            html.Div(id="sound-status", 
                                    style={'marginTop': '10px', 'padding': '10px',
                                          'borderRadius': '4px', 'textAlign': 'center'}),
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px',
                                 'marginBottom': '20px'}),
                        
                        # 現有音效列表
                        html.Div([
                            html.H5("現有音效", style={'color': '#555', 'marginBottom': '15px'}),
                            html.Div(id="sounds-list", style={'maxHeight': '300px', 'overflowY': 'auto'}),
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px'}),
                        
                    ], style={'background': 'white', 'padding': '20px', 'borderRadius': '8px',
                             'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'height': 'fit-content'}),
                ], style={'flex': '1'}),
                
            ], style={'display': 'flex', 'margin': '0 20px'}),
            
            # 隱藏存儲組件
            dcc.Store(id="subjects-store-mgmt", data=[]),
            dcc.Store(id="sounds-store-mgmt", data=[]),
            
        ], style={'minHeight': '100vh', 'backgroundColor': '#ecf0f1', 'padding': '20px 0'})
    
    def register_callbacks(self, app):
        """註冊管理頁面的回調函數"""
        
        @app.callback(
            [Output("subject-status", "children"),
             Output("subject-status", "style"),
             Output("subjects-store-mgmt", "data"),
             Output("subject-id-input", "value"),
             Output("subject-gender-dropdown", "value"),
             Output("subject-age-input", "value"),
             Output("researcher-name-input", "value"),
             Output("subject-notes-input", "value")],
            [Input("add-subject-btn", "n_clicks")],
            [State("subject-id-input", "value"),
             State("subject-gender-dropdown", "value"),
             State("subject-age-input", "value"),
             State("researcher-name-input", "value"),
             State("subject-notes-input", "value")]
        )
        def add_subject(n_clicks, subject_id, gender, age, researcher, notes):
            """新增受試者"""
            if not n_clicks:
                subjects = self.db_writer.get_subjects()
                return "", {}, subjects, "", None, "", "", ""
            
            # 驗證輸入
            if not subject_id or not subject_id.strip():
                return "❌ 請輸入受試者ID", {'backgroundColor': '#f8d7da', 'color': '#721c24'}, [], subject_id, gender, age, researcher, notes
            
            if not gender:
                return "❌ 請選擇性別", {'backgroundColor': '#f8d7da', 'color': '#721c24'}, [], subject_id, gender, age, researcher, notes
            
            if not age or age < 0 or age > 150:
                return "❌ 請輸入有效年齡 (0-150)", {'backgroundColor': '#f8d7da', 'color': '#721c24'}, [], subject_id, gender, age, researcher, notes
            
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
                           {'backgroundColor': '#d4edda', 'color': '#155724'},
                           subjects, "", None, "", "", "")
                else:
                    return ("❌ 受試者ID已存在，請使用不同的ID", 
                           {'backgroundColor': '#f8d7da', 'color': '#721c24'},
                           [], subject_id, gender, age, researcher, notes)
                    
            except Exception as e:
                logger.error(f"新增受試者失敗: {e}")
                return (f"❌ 新增失敗: {str(e)}", 
                       {'backgroundColor': '#f8d7da', 'color': '#721c24'},
                       [], subject_id, gender, age, researcher, notes)
        
        @app.callback(
            Output("subjects-list", "children"),
            [Input("subjects-store-mgmt", "data")]
        )
        def update_subjects_list(subjects_data):
            """更新受試者列表顯示"""
            if not subjects_data:
                return html.P("尚無受試者資料", style={'textAlign': 'center', 'color': '#6c757d'})
            
            subjects_cards = []
            for subject in subjects_data:
                card = html.Div([
                    html.Div([
                        html.H6(f"ID: {subject['subject_id']}", 
                               style={'margin': '0', 'color': '#495057'}),
                        html.P(f"{subject['gender']} | {subject['age']}歲", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '14px'}),
                        html.P(f"研究者: {subject['researcher_name'] or 'N/A'}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'}),
                        html.P(f"創建時間: {subject['created_at']}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'}),
                        html.P(f"備註: {subject['notes'] or 'N/A'}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'})
                    ])
                ], style={'border': '1px solid #dee2e6', 'borderRadius': '4px',
                         'padding': '10px', 'marginBottom': '10px', 'backgroundColor': 'white'})
                subjects_cards.append(card)
            
            return subjects_cards
        
        @app.callback(
            [Output("sound-status", "children"),
             Output("sound-status", "style"),
             Output("sounds-store-mgmt", "data"),
             Output("sound-name-input", "value"),
             Output("sound-category-dropdown", "value"),
             Output("sound-description-input", "value")],
            [Input("upload-sound-btn", "n_clicks")],
            [State("sound-name-input", "value"),
             State("sound-category-dropdown", "value"),
             State("sound-upload", "contents"),
             State("sound-upload", "filename"),
             State("sound-description-input", "value")]
        )
        def upload_sound(n_clicks, sound_name, category, contents, filename, description):
            """上傳音效檔案"""
            if not n_clicks:
                sounds = self.db_writer.get_ambient_sounds()
                return "", {}, sounds, "", None, ""
            
            # 驗證輸入
            if not sound_name or not sound_name.strip():
                return "❌ 請輸入音效名稱", {'backgroundColor': '#f8d7da', 'color': '#721c24'}, [], sound_name, category, description
            
            if not category:
                return "❌ 請選擇音效類別", {'backgroundColor': '#f8d7da', 'color': '#721c24'}, [], sound_name, category, description
            
            if not contents or not filename:
                return "❌ 請選擇音效檔案", {'backgroundColor': '#f8d7da', 'color': '#721c24'}, [], sound_name, category, description
            
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
                    return (f"✅ 音效 '{sound_name}' 上傳成功！ID: {sound_id}", 
                           {'backgroundColor': '#d4edda', 'color': '#155724'},
                           sounds, "", None, "")
                else:
                    # 刪除已上傳的檔案
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return ("❌ 音效名稱已存在，請使用不同的名稱", 
                           {'backgroundColor': '#f8d7da', 'color': '#721c24'},
                           [], sound_name, category, description)
                    
            except Exception as e:
                logger.error(f"上傳音效失敗: {e}")
                return (f"❌ 上傳失敗: {str(e)}", 
                       {'backgroundColor': '#f8d7da', 'color': '#721c24'},
                       [], sound_name, category, description)
        
        @app.callback(
            Output("sounds-list", "children"),
            [Input("sounds-store-mgmt", "data")]
        )
        def update_sounds_list(sounds_data):
            """更新音效列表顯示"""
            if not sounds_data:
                return html.P("尚無音效檔案", style={'textAlign': 'center', 'color': '#6c757d'})
            
            sounds_cards = []
            for sound in sounds_data:
                card = html.Div([
                    html.Div([
                        html.H6(f"🎵 {sound['sound_name']}", 
                               style={'margin': '0', 'color': '#495057'}),
                        html.P(f"類別: {sound['style_category']}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '14px'}),
                        html.P(f"檔案: {sound['filename']}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'}),
                        html.P(f"時長: {sound['duration_seconds']:.1f}秒", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'}),
                        html.P(f"ID: {sound['id']}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'}),
                        html.P(f"描述: {sound['description'] or 'N/A'}", 
                              style={'margin': '5px 0', 'color': '#6c757d', 'fontSize': '12px'})
                    ])
                ], style={'border': '1px solid #dee2e6', 'borderRadius': '4px',
                         'padding': '10px', 'marginBottom': '10px', 'backgroundColor': 'white'})
                sounds_cards.append(card)
            
            return sounds_cards