import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.font_manager as fm
import plotly.express as px


nanum_font = None
for font in fm.fontManager.ttflist:
    if 'Nanum' in font.name:
        nanum_font = font.name
        break

if nanum_font:
    matplotlib.rcParams['font.family'] = nanum_font
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title='RTU Power Prediction Dashboard',
    page_icon='⚡',
    layout='wide'
)

st.title('⚡ RTU Smart Factory Power Prediction Dashboard')
st.markdown('---')


@st.cache_data
def load_submission():
    return pd.read_csv('output/submission.csv')

@st.cache_data
def load_anomaly():
    return pd.read_csv('output/anomaly_result.csv', parse_dates=['dt'])

@st.cache_data
def load_equip_anomaly():
    path = 'output/anomaly_per_equipment.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['dt'])
        return df[df['dt'] < '2025-05-01']  # 5월 이후 제거
    return pd.DataFrame()


with st.spinner('데이터 로딩 중...'):
    submission    = load_submission()
    anomaly       = load_anomaly()
    equip_anomaly = load_equip_anomaly()


# KPI
st.subheader('📊 5월 예측 요약')
col1, col2, col3, col4 = st.columns(4)

agg_pow       = submission['agg_pow'].iloc[0]
may_bill      = submission['may_bill'].iloc[0]
may_carbon    = submission['may_carbon'].iloc[0]
anomaly_count = int(anomaly['zscore_anomaly'].sum())

col1.metric('5월 누적 전력',  f'{agg_pow:,.0f} kWh')
col2.metric('5월 전기요금',   f'{may_bill/1e8:.2f} 억원')
col3.metric('5월 탄소배출',   f'{may_carbon/1e4:.1f} 만 kgCO2')
col4.metric('이상탐지 건수',  f'{anomaly_count} 건')

st.markdown('---')


# 5월 예측 그래프
st.subheader('📈 5월 시간별 전력 예측')

submission['id'] = pd.to_datetime(submission['id'])

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(submission['id'], submission['hourly_pow'],
        color='tomato', linewidth=0.8, label='May Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('hourly_pow (kW)')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close()

st.markdown('---')


# 전체 합산 이상탐지
st.subheader('🚨 이상탐지 현황 (전체 합산)')

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('**Rolling Z-score**')
    zscore_anomalies = anomaly[anomaly['zscore_anomaly'] == True]
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(anomaly['dt'], anomaly['hourly_pow'],
             color='steelblue', linewidth=0.5, label='hourly_pow')
    ax2.scatter(zscore_anomalies['dt'], zscore_anomalies['hourly_pow'],
                color='red', s=15, zorder=5, label=f'Anomaly ({len(zscore_anomalies)})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

with col_b:
    st.markdown('**Isolation Forest**')
    iso_anomalies = anomaly[anomaly['iso_anomaly'] == True]
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    ax3.plot(anomaly['dt'], anomaly['hourly_pow'],
             color='steelblue', linewidth=0.5, label='hourly_pow')
    ax3.scatter(iso_anomalies['dt'], iso_anomalies['hourly_pow'],
                color='orange', s=15, zorder=5, label=f'Anomaly ({len(iso_anomalies)})')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close()

st.markdown('---')


# 설비별 이상탐지
st.subheader('🔧 설비별 이상탐지 현황')

if not equip_anomaly.empty:
    equip_summary = (
        equip_anomaly[equip_anomaly['zscore_anomaly'] == True]
        .groupby('equipment')
        .size()
        .reset_index(name='anomaly_count')
        .sort_values('anomaly_count', ascending=False)
    )


    col_c, col_d = st.columns([1, 1])

    with col_c:
        st.markdown('**설비별 이상 건수 (두 방법 교집합)**')
        fig5 = px.bar(
            equip_summary,
            x='anomaly_count',
            y='equipment',
            orientation='h',
            color=equip_summary['anomaly_count'].apply(
                lambda x: 'max' if x == equip_summary['anomaly_count'].max() else 'normal'
            ),
            color_discrete_map={'max': 'crimson', 'normal': 'steelblue'},
            text='anomaly_count'
        )
        fig5.update_layout(
            showlegend=False,
            xaxis_title='이상 건수',
            yaxis_title='',
            yaxis=dict(autorange='reversed'),
            height=400
        )
        fig5.update_traces(textposition='outside')
        st.plotly_chart(fig5, use_container_width=True)

    with col_d:
        st.markdown('**설비별 이상탐지 상세**')
        st.dataframe(
            equip_summary.rename(columns={'equipment': '설비명', 'anomaly_count': '이상 건수'}),
            use_container_width=True,
            hide_index=True
        )

        selected = st.selectbox("설비 선택", equip_summary["equipment"].tolist())
        if selected:
            sel_all  = equip_anomaly[equip_anomaly['equipment'] == selected].sort_values('dt')
            sel_anom = sel_all[sel_all['zscore_anomaly'] == True]
            fig6, ax6 = plt.subplots(figsize=(7, 3))
            ax6.plot(sel_all['dt'], sel_all['hourly_pow'],
                    color='steelblue', linewidth=0.5, label=selected)
            ax6.scatter(sel_anom['dt'], sel_anom['hourly_pow'],
                        color='red', s=20, zorder=5, label=f'Anomaly ({len(sel_anom)})')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            st.pyplot(fig6)
            plt.close()
else:
    st.info('output/anomaly_per_equipment.csv 파일이 없어요.')

st.markdown('---')


# 이상탐지 상세 테이블
st.subheader('📋 이상탐지 상세 목록 (전체 합산)')

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

st.markdown('---')
st.caption('RTU Smart Factory Power Prediction Pipeline')