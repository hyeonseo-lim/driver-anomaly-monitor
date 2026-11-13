import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from sklearn.preprocessing import RobustScaler

st.set_page_config(
    page_title="운전자 이상행동 진단 모니터 | 충북대학교 정보통신공학부",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 모바일 화면 여백 및 가로 넘침 방지 CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 2.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 760px !important;
    }
    h1 {
        font-size: 1.32rem !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.3rem !important;
        word-break: keep-all;
    }
    h3 {
        font-size: 0.96rem !important;
        font-weight: 600 !important;
        color: #1e293b;
        margin-top: 0.8rem !important;
        margin-bottom: 0.3rem !important;
        word-break: keep-all;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
    }
    .affil-badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }
    .footer-credit {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.5;
        word-break: keep-all;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="affil-badge">충북대학교 정보통신공학부  |  임현서 · 전민서</div>', unsafe_allow_html=True)
st.title("운전자 이상행동 탐지 및 원인 분별 모니터")
st.caption("Stage 1: GRU-Autoencoder 이상 탐지 | Stage 2: 1D-CNN+GRU (MC-Dropout)")

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = base_dir

EVENT_NAMES = [
    "Normal (정상 주행)",
    "Sudden Acceleration (급가속)",
    "Intermittent Braking (간헐적 급제동)",
    "Sudden Steering (급조향)",
    "Rapid Lane Change (급차선 변경)",
    "Irregular Steering (주의분산 조향)",
    "Drowsy Recovery (졸음 복원 조타)"
]

SHORT_LABELS = [
    "정상 주행",
    "급가속",
    "간헐적 급제동",
    "급조향",
    "급차선 변경",
    "주의분산 조향",
    "졸음 복원 조타"
]

# ==============================================================================
# 1. 모델 정의
# ==============================================================================
class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, hn = self.encoder(x)
        z = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(z)
        return self.fc_out(dec_out)

class CNNGRUCauseClassifier(nn.Module):
    def __init__(self, input_dim=13, num_classes=7, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(64, 48, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        c = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        _, hn = self.gru(c)
        return self.fc(self.dropout(hn[-1]))

@st.cache_resource
def load_models_and_scaler():
    s1 = GRUAutoencoder(input_dim=13, hidden_dim=32)
    s2 = CNNGRUCauseClassifier(input_dim=13, num_classes=7, dropout_rate=0.3)
    
    s1_path = os.path.join(dataset_dir, "stage1_gru_ae_carla_v2.pth")
    s2_path = os.path.join(dataset_dir, "stage2_cnngru_carla_v2.pth")
    
    if os.path.exists(s1_path) and os.path.exists(s2_path):
        s1.load_state_dict(torch.load(s1_path, map_location="cpu"))
        s2.load_state_dict(torch.load(s2_path, map_location="cpu"))
        loaded = True
    else:
        loaded = False
        
    base_cols = [
        'speed', 'steer', 'throttle', 'brake',
        'accel_longitudinal', 'accel_lateral', 'yaw_rate_z',
        'lane_deviation', 'lateral_rate'
    ]
    all_cols = base_cols + ['steer_diff', 'throttle_diff', 'brake_diff', 'lane_deviation_diff']
    
    dfs = []
    for i in range(1, 5):
        p = os.path.join(dataset_dir, f"risky_data_sim_driver_0{i}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d['accel_longitudinal'] = d['accel_longitudinal'].clip(-15.0, 15.0)
            d['accel_lateral'] = d['accel_lateral'].clip(-15.0, 15.0)
            d['yaw_rate_z'] = d['yaw_rate_z'].clip(-60.0, 60.0)
            d['lateral_rate'] = d['lateral_rate'].clip(-15.0, 15.0)
            for c in ['steer', 'throttle', 'brake', 'lane_deviation']:
                d[f'{c}_diff'] = d[c].diff().fillna(0.0)
            dfs.append(d)
            
    full_df = pd.concat(dfs, ignore_index=True)
    scaler = RobustScaler().fit(full_df[all_cols].values)
    driver_map = {f"sim_driver_0{i}": dfs[i-1] for i in range(1, len(dfs)+1)}
    return s1, s2, scaler, driver_map, all_cols, loaded

s1_model, s2_model, scaler, driver_map, all_cols, is_loaded = load_models_and_scaler()

# ==============================================================================
# 2. 상단 제어 바
# ==============================================================================
with st.expander("진단 대상 및 임계값 설정", expanded=False):
    selected_driver = st.selectbox("주행 로그 (LOGOCV)", list(driver_map.keys()), index=len(driver_map) - 1)
    mse_threshold = st.slider("이상 판정 임계값 (MSE)", 0.5, 4.0, 1.45, 0.05)
    sim_speed = st.slider("갱신 주기 (초)", 0.01, 0.20, 0.04, 0.01)

test_df = driver_map[selected_driver]
start_btn = st.button("주행 스트리밍 진단 실행", type="primary", use_container_width=True)

# ==============================================================================
# 3. 반응형 모니터링 레이아웃
# ==============================================================================
status_banner = st.empty()

r1_c1, r1_c2 = st.columns(2)
m_time = r1_c1.empty()
m_speed = r1_c2.empty()

r2_c1, r2_c2 = st.columns(2)
m_mse = r2_c1.empty()
m_uncert = r2_c2.empty()

st.subheader("Stage 2 세부 원인 확률 분포 (%)")
chart_probs = st.empty()
gt_info = st.empty()

st.subheader("차량 동역학 시계열 신호")
chart_stream = st.empty()

st.subheader("Stage 1 복원 오차 (MSE)")
chart_mse = st.empty()

st.subheader("이상 감지 이력")
log_box = st.empty()

st.markdown("""
    <div class="footer-credit">
        <strong>충북대학교 정보통신공학부</strong>  |  개발 및 연구: <strong>임현서 · 전민서</strong><br>
        시계열 운전자 행동 데이터 기반 다중 운전자 이상행동 탐지 및 원인 분별 확률 모델링
    </div>
""", unsafe_allow_html=True)

scaled_arr = np.clip(scaler.transform(test_df[all_cols].values), -5.0, 5.0).astype(np.float32)

if start_btn:
    s1_model.eval()
    s2_model.train()
    
    WINDOW = 50
    mse_history = []
    event_logs = []
    
    for idx in range(WINDOW, len(test_df), 5):
        window_x = torch.tensor(scaled_arr[idx-WINDOW:idx]).unsqueeze(0)
        
        with torch.no_grad():
            recon = s1_model(window_x)
            mse_val = torch.mean((window_x - recon) ** 2).item()
            mc_probs = torch.stack([F.softmax(s2_model(window_x), dim=1) for _ in range(20)]).squeeze(1).numpy()
            
        mean_prob = np.mean(mc_probs, axis=0)
        std_uncert = np.std(mc_probs, axis=0)
        
        pred_idx = int(np.argmax(mean_prob))
        pred_label = SHORT_LABELS[pred_idx]
        pred_conf = mean_prob[pred_idx] * 100
        pred_std = std_uncert[pred_idx]
        
        cur_row = test_df.iloc[idx]
        is_anomaly = (mse_val > mse_threshold) or (pred_idx != 0)
        
        if is_anomaly:
            status_banner.error(
                f"이상 감지: {pred_label} ({pred_conf:.1f}% / σ={pred_std:.3f})"
            )
            event_logs.insert(0, f"[{cur_row['time']:05.1f}s] {pred_label} | {pred_conf:.1f}% (σ={pred_std:.3f})")
        else:
            status_banner.success(
                f"정상 주행 (MSE: {mse_val:.3f} / σ={pred_std:.3f})"
            )
            
        m_time.metric("경과 시간", f"{cur_row['time']:.2f} s")
        m_speed.metric("차량 속도", f"{cur_row['speed']:.1f} km/h")
        m_mse.metric("복원 오차 (MSE)", f"{mse_val:.4f}", delta=f"{mse_val - mse_threshold:+.3f}", delta_color="inverse")
        m_uncert.metric("불확실성 (σ)", f"{pred_std:.4f}")
        
        # 가로 막대그래프 적용으로 모바일에서도 클래스명 잘림 없음
        prob_df = pd.DataFrame({
            "원인": SHORT_LABELS,
            "확률 (%)": mean_prob * 100
        }).set_index("원인")
        chart_probs.bar_chart(prob_df, horizontal=True, height=230, use_container_width=True)
        
        gt_info.caption(f"실제 라벨: {cur_row['event_type']} / 상태: {cur_row['driving_state']}")
        
        stream_df = test_df.iloc[max(0, idx-100):idx][['steer', 'lane_deviation', 'accel_lateral']].reset_index(drop=True)
        chart_stream.line_chart(stream_df, height=180, use_container_width=True)
        
        mse_history.append({"MSE": mse_val, "Threshold": mse_threshold})
        chart_mse.line_chart(pd.DataFrame(mse_history[-60:]), height=150, use_container_width=True)
        
        if event_logs:
            log_box.code("\n".join(event_logs[:5]), language="text")
        else:
            log_box.code("감지된 이상 이벤트 없음", language="text")
            
        time.sleep(sim_speed)
else:
    status_banner.info("상단의 '주행 스트리밍 진단 실행' 버튼을 누르면 실시간 추론이 시작됩니다.")
