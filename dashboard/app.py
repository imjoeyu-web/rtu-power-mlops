"""
RTU 전력 예측 대시보드
실행: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# 한글 폰트
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# src 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title='RTU 전력 예측 대시보드',
    page_icon='⚡',
    layout='wide'
)

st.title('⚡ RTU 스마트팩토리 전력 예측 대시보드')
st.markdown('---')


# 데이터 로드 (캐싱으로 속도 개선)
@st.cache_data
def load_data():
    from src.preprocess import load_and_aggregate, make_features
    hourly = load_and_aggregate('data/rtu_data_full.csv')
    hourly = make_features(hourly)
    return hourly

@st.cache_data
def load_submission():
    return pd.read_csv('output/submission.csv')

@st.cache_data
def load_anomaly():
    return pd.read_csv('output/anomaly_result.csv', parse_dates=['dt'])


# 로딩
with st.spinner('데이터 로딩 중...'):
    hourly    = load_data()
    submission = load_submission()
    anomaly   = load_anomaly()


# ─── 상단 KPI 카드 ───
st.subheader('📊 5월 예측 요약')
col1, col2, col3, col4 = st.columns(4)

agg_pow    = submission['agg_pow'].iloc[0]
may_bill   = submission['may_bill'].iloc[0]
may_carbon = submission['may_carbon'].iloc[0]
anomaly_count = int(anomaly['zscore_anomaly'].sum())

col1.metric('5월 누적 전력', f'{agg_pow:,.0f} kWh')
col2.metric('5월 전기요금', f'{may_bill/1e8:.2f} 억원')
col3.metric('5월 탄소배출', f'{may_carbon/1e4:.1f} 만 kgCO₂')
col4.metric('이상탐지 건수', f'{anomaly_count} 건')

st.markdown('---')


# ─── 5월 예측 그래프 ───
st.subheader('📈 5월 시간별 전력 예측')

submission['id'] = pd.to_datetime(submission['id'])

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(submission['id'], submission['hourly_pow'],
        color='tomato', linewidth=0.8, label='5월 예측값')
ax.set_xlabel('날짜')
ax.set_ylabel('hourly_pow (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close()


# ─── 이상탐지 그래프 ───
st.subheader('🚨 이상탐지 현황')

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('**Rolling Z-score 이상탐지**')
    zscore_anomalies = anomaly[anomaly['zscore_anomaly'] == True]
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(anomaly['dt'], anomaly['hourly_pow'],
             color='steelblue', linewidth=0.5, label='hourly_pow')
    ax2.scatter(zscore_anomalies['dt'], zscore_anomalies['hourly_pow'],
                color='red', s=15, zorder=5, label=f'이상 ({len(zscore_anomalies)}건)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

with col_b:
    st.markdown('**Isolation Forest 이상탐지**')
    iso_anomalies = anomaly[anomaly['iso_anomaly'] == True]
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(anomaly['dt'], anomaly['hourly_pow'],
             color='steelblue', linewidth=0.5, label='hourly_pow')
    ax3.scatter(iso_anomalies['dt'], iso_anomalies['hourly_pow'],
                color='orange', s=15, zorder=5, label=f'이상 ({len(iso_anomalies)}건)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close()


# ─── 이상탐지 상세 테이블 ───
st.subheader('📋 이상탐지 상세 목록')

tab1, tab2 = st.tabs(['Z-score 이상', 'Isolation Forest 이상'])

with tab1:
    st.dataframe(
        zscore_anomalies[['dt', 'hourly_pow', 'z_score']]
        .rename(columns={'dt': '시간', 'hourly_pow': '전력(kW)', 'z_score': 'Z-score'})
        .reset_index(drop=True),
        use_container_width=True
    )

with tab2:
    st.dataframe(
        iso_anomalies[['dt', 'hourly_pow']]
        .rename(columns={'dt': '시간', 'hourly_pow': '전력(kW)'})
        .reset_index(drop=True),
        use_container_width=True
    )


# ─── 시간대별 패턴 ───
st.subheader('🕐 시간대별 평균 전력 패턴')

hourly['hour'] = pd.to_datetime(hourly['dt']).dt.hour
hour_avg = hourly.groupby('hour')['hourly_pow'].mean()

fig4, ax4 = plt.subplots(figsize=(14, 3))
ax4.plot(hour_avg.index, hour_avg.values, marker='o', color='steelblue')
ax4.set_xlabel('시간대')
ax4.set_ylabel('평균 hourly_pow (kW)')
ax4.set_xticks(range(24))
ax4.grid(True, alpha=0.3)
st.pyplot(fig4)
plt.close()

st.markdown('---')
st.caption('RTU 스마트팩토리 전력 예측 파이프라인 | 인하대학교 DeCo팀 참고')
