"""EEG Session 歷史管理頁面"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
from services.database_service import EnhancedDatabaseWriter

logger = logging.getLogger(__name__)


class SessionHistoryPage:
    """Session 歷史管理頁面類"""
    
    def __init__(self, db_writer: EnhancedDatabaseWriter):
        self.db_writer = db_writer
        
    def create_layout(self):
        """創建 Session 歷史頁面佈局"""
        return html.Div([
            # 頁面標題區域
            html.Div([
                html.Div([
                    html.H2([
                        html.I(className="fas fa-history", 
                               style={'marginRight': '10px', 'color': '#007bff'}),
                        "Session 歷史記錄"
                    ], style={'color': '#2c3e50', 'marginBottom': '10px', 'fontSize': '24px'}),
                    html.P("瀏覽和管理所有實驗會話記錄，支援篩選和 CSV 匯出功能",
                           style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                    'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px',
                    'border': '1px solid #dee2e6'
                })
            ]),
            
            # 篩選控制區域
            html.Div([
                html.Div([
                    html.H4([
                        html.I(className="fas fa-filter", 
                               style={'marginRight': '8px', 'color': '#28a745'}),
                        "篩選控制"
                    ], style={'fontSize': '18px', 'color': '#495057', 'marginBottom': '15px'}),
                    
                    # 篩選控制項
                    html.Div([
                        # 受試者篩選
                        html.Div([
                            html.Label("受試者 ID:", 
                                      style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Input(
                                id="history-subject-filter",
                                type="text",
                                placeholder="輸入受試者 ID 進行篩選",
                                style={
                                    'width': '100%', 'padding': '8px 12px', 
                                    'border': '1px solid #ced4da', 'borderRadius': '4px',
                                    'fontSize': '14px'
                                }
                            )
                        ], style={'flex': '1', 'marginRight': '15px'}),
                        
                        # 數量限制
                        html.Div([
                            html.Label("顯示數量:", 
                                      style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                            dcc.Dropdown(
                                id="history-limit-dropdown",
                                options=[
                                    {'label': '20 筆', 'value': 20},
                                    {'label': '50 筆', 'value': 50},
                                    {'label': '100 筆', 'value': 100},
                                    {'label': '200 筆', 'value': 200}
                                ],
                                value=50,
                                style={'fontSize': '14px'}
                            )
                        ], style={'flex': '0 0 120px', 'marginRight': '15px'}),
                        
                        # 刷新按鈕
                        html.Div([
                            html.Button([
                                html.I(className="fas fa-sync-alt", style={'marginRight': '5px'}),
                                "刷新"
                            ], id="history-refresh-btn",
                               style={
                                   'backgroundColor': '#17a2b8', 'color': 'white',
                                   'border': 'none', 'padding': '8px 16px',
                                   'borderRadius': '4px', 'cursor': 'pointer',
                                   'fontSize': '14px', 'fontWeight': 'bold',
                                   'marginTop': '24px'
                               })
                        ], style={'flex': '0 0 auto'})
                        
                    ], style={'display': 'flex', 'alignItems': 'flex-end'})
                    
                ], style={
                    'background': 'white', 'padding': '20px', 'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'
                })
            ]),
            
            # Session 清單區域
            html.Div([
                html.Div([
                    html.H4([
                        html.I(className="fas fa-list", 
                               style={'marginRight': '8px', 'color': '#dc3545'}),
                        "Session 清單"
                    ], style={'fontSize': '18px', 'color': '#495057', 'marginBottom': '15px'}),
                    
                    # 載入指示器
                    dcc.Loading(
                        id="history-loading",
                        type="default",
                        children=[
                            html.Div(id="session-history-table")
                        ]
                    ),
                    
                    # 狀態顯示
                    html.Div(id="history-status", 
                            style={'marginTop': '15px', 'fontSize': '14px', 'color': '#6c757d'})
                    
                ], style={
                    'background': 'white', 'padding': '20px', 'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'
                })
            ]),
            
            # CSV 下載組件
            dcc.Download(id="download-session-csv"),
            
            # 隱藏的數據存儲
            dcc.Store(id="session-history-store", data=[]),
            
        ], style={'padding': '10px'})
    
    def create_session_table(self, sessions: List[Dict]) -> html.Div:
        """創建 Session 清單表格"""
        if not sessions:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle", 
                           style={'fontSize': '48px', 'color': '#6c757d', 'marginBottom': '15px'}),
                    html.H5("無資料", style={'color': '#6c757d', 'marginBottom': '10px'}),
                    html.P("目前沒有找到符合條件的 Session 記錄", 
                           style={'color': '#868e96', 'fontSize': '14px'})
                ], style={
                    'textAlign': 'center', 'padding': '40px',
                    'border': '2px dashed #dee2e6', 'borderRadius': '8px'
                })
            ])
        
        # 創建表格行
        table_rows = []
        for session in sessions:
            # 格式化時間顯示
            start_time = session['start_time'][:19] if session['start_time'] else 'N/A'
            end_time = session['end_time'][:19] if session['end_time'] else 'N/A'
            duration = f"{session['duration']:.1f}s" if session['duration'] else 'N/A'
            
            # 狀態指示器
            if session['status'] == '已完成':
                status_badge = html.Span("已完成", 
                    style={'background': '#28a745', 'color': 'white', 'padding': '4px 8px',
                           'borderRadius': '12px', 'fontSize': '12px', 'fontWeight': 'bold'})
            else:
                status_badge = html.Span("進行中", 
                    style={'background': '#ffc107', 'color': '#212529', 'padding': '4px 8px',
                           'borderRadius': '12px', 'fontSize': '12px', 'fontWeight': 'bold'})
            
            # 操作按鈕
            download_btn = html.Button([
                html.I(className="fas fa-download", style={'marginRight': '5px'}),
                "CSV"
            ], id={'type': 'download-csv-btn', 'index': session['session_id']},
               style={
                   'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                   'padding': '6px 12px', 'borderRadius': '4px', 'cursor': 'pointer',
                   'fontSize': '12px', 'fontWeight': 'bold'
               })
            
            table_rows.append(html.Tr([
                html.Td(session['session_id'], 
                        style={'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': 'bold'}),
                html.Td(session['subject_id'], 
                        style={'fontWeight': 'bold', 'color': '#495057'}),
                html.Td(start_time, 
                        style={'fontSize': '13px', 'color': '#6c757d'}),
                html.Td(duration, 
                        style={'fontSize': '13px', 'textAlign': 'right'}),
                html.Td(session['eye_state'] or 'N/A', 
                        style={'fontSize': '13px'}),
                html.Td(status_badge),
                html.Td(session['researcher_name'] or 'System', 
                        style={'fontSize': '13px'}),
                html.Td(download_btn, style={'textAlign': 'center'})
            ], style={'borderBottom': '1px solid #dee2e6'}))
        
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Session ID", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                                'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057'}),
                    html.Th("受試者", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                           'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057'}),
                    html.Th("開始時間", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                            'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057'}),
                    html.Th("持續時間", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                            'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057',
                                            'textAlign': 'right'}),
                    html.Th("眼睛狀態", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                            'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057'}),
                    html.Th("狀態", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                         'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057'}),
                    html.Th("研究者", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                          'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057'}),
                    html.Th("操作", style={'padding': '12px 8px', 'backgroundColor': '#f8f9fa',
                                         'fontWeight': 'bold', 'fontSize': '14px', 'color': '#495057',
                                         'textAlign': 'center'})
                ])
            ]),
            html.Tbody(table_rows)
        ], style={
            'width': '100%', 'borderCollapse': 'collapse',
            'border': '1px solid #dee2e6', 'borderRadius': '8px',
            'overflow': 'hidden'
        })
    
    def export_session_to_csv(self, session_id: str) -> tuple:
        """匯出 Session 數據為 CSV"""
        try:
            # 獲取會話數據
            session_data = self.db_writer.get_session_data_for_export(session_id)
            
            if not session_data:
                return None, f"找不到 Session: {session_id}"
            
            # 準備 CSV 數據
            csv_rows = []
            session_info = session_data['session_info']
            unified_data = session_data['unified_data']
            
            # 添加會話信息標頭
            csv_rows.append([
                '# Session Information',
                f"Session ID: {session_info['session_id']}",
                f"Subject: {session_info['subject_id']}",
                f"Start Time: {session_info['start_time']}",
                f"Duration: {session_info['duration']}s" if session_info['duration'] else "N/A",
                f"Eye State: {session_info['eye_state']}",
                f"Researcher: {session_info['researcher_name'] or 'System'}"
            ])
            csv_rows.append([])  # 空行
            
            # 添加數據標頭
            headers = [
                'timestamp', 'attention', 'meditation', 'signal_quality',
                'temperature', 'humidity', 'light', 'blink_intensity',
                'delta_power', 'theta_power', 'low_alpha_power', 'high_alpha_power',
                'low_beta_power', 'high_beta_power', 'low_gamma_power', 'mid_gamma_power',
                'voltage_samples_count'
            ]
            csv_rows.append(headers)
            
            # 添加統一記錄數據
            for row in unified_data:
                # 處理電壓數據
                voltage_data = row[8]  # voltage_data 欄位
                voltage_count = 0
                if voltage_data:
                    try:
                        voltage_array = json.loads(voltage_data)
                        voltage_count = len(voltage_array)
                    except:
                        voltage_count = 0
                
                csv_row = [
                    row[0],  # timestamp_local
                    row[1] or '',  # attention
                    row[2] or '',  # meditation
                    row[3] or '',  # signal_quality
                    row[4] or '',  # temperature
                    row[5] or '',  # humidity
                    row[6] or '',  # light
                    row[7] or '',  # blink_intensity
                    row[9] or '',  # delta_power
                    row[10] or '',  # theta_power
                    row[11] or '',  # low_alpha_power
                    row[12] or '',  # high_alpha_power
                    row[13] or '',  # low_beta_power
                    row[14] or '',  # high_beta_power
                    row[15] or '',  # low_gamma_power
                    row[16] or '',  # mid_gamma_power
                    voltage_count  # voltage_samples_count
                ]
                csv_rows.append(csv_row)
            
            # 轉換為 DataFrame 並產生 CSV
            df = pd.DataFrame(csv_rows)
            csv_string = df.to_csv(index=False, header=False, encoding='utf-8-sig')
            
            filename = f"{session_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            return csv_string, filename
            
        except Exception as e:
            logger.error(f"Error exporting session to CSV: {e}")
            return None, f"匯出錯誤: {str(e)}"
    
    def register_callbacks(self, app: dash.Dash):
        """註冊頁面回調函數"""
        
        # 更新 Session 歷史清單
        @app.callback(
            [Output("session-history-table", "children"),
             Output("history-status", "children"),
             Output("session-history-store", "data")],
            [Input("history-refresh-btn", "n_clicks"),
             Input("history-subject-filter", "value"),
             Input("history-limit-dropdown", "value")],
            prevent_initial_call=False
        )
        def update_session_history(refresh_clicks, subject_filter, limit):
            """更新 Session 歷史清單"""
            try:
                # 獲取會話歷史
                sessions = self.db_writer.get_session_history(
                    limit=limit or 50,
                    subject_filter=subject_filter
                )
                
                # 創建表格
                table = self.create_session_table(sessions)
                
                # 狀態信息
                status_text = f"顯示 {len(sessions)} 筆記錄"
                if subject_filter:
                    status_text += f" (篩選: {subject_filter})"
                
                return table, status_text, sessions
                
            except Exception as e:
                logger.error(f"Error updating session history: {e}")
                error_msg = html.Div([
                    html.I(className="fas fa-exclamation-triangle", 
                           style={'color': '#dc3545', 'marginRight': '8px'}),
                    f"載入錯誤: {str(e)}"
                ], style={'color': '#dc3545', 'fontSize': '14px'})
                
                return error_msg, "載入失敗", []
        
        # CSV 下載處理
        @app.callback(
            Output("download-session-csv", "data"),
            [Input({'type': 'download-csv-btn', 'index': dash.dependencies.ALL}, "n_clicks")],
            [State("session-history-store", "data")],
            prevent_initial_call=True
        )
        def download_session_csv(download_clicks, sessions_data):
            """處理 CSV 下載請求"""
            ctx = callback_context
            if not ctx.triggered or not any(download_clicks):
                return dash.no_update
            
            # 找出被點擊的按鈕
            button_id = None
            for i, clicks in enumerate(download_clicks):
                if clicks:
                    button_id = ctx.inputs_list[0][i]['id']['index']
                    break
            
            if not button_id:
                return dash.no_update
            
            try:
                # 匯出 CSV
                csv_content, filename = self.export_session_to_csv(button_id)
                
                if csv_content:
                    return dict(content=csv_content, filename=filename)
                else:
                    logger.error(f"CSV export failed: {filename}")
                    return dash.no_update
                    
            except Exception as e:
                logger.error(f"Error in CSV download callback: {e}")
                return dash.no_update