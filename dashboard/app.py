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
        color='steelblue', linewidth=0.8, label='May Prediction')
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
    surge = zscore_anomalies[zscore_anomalies['z_score'] > 0]
    drop  = zscore_anomalies[zscore_anomalies['z_score'] < 0]

    fig2 = px.line(anomaly, x='dt', y='hourly_pow',
                   labels={'dt': '', 'hourly_pow': '전력(kW)'},
                   color_discrete_sequence=['steelblue'])
    fig2.update_traces(line_width=0.5)
    fig2.add_scatter(x=surge['dt'], y=surge['hourly_pow'],
                     mode='markers', marker=dict(color='crimson', size=5),
                     name=f'급등 ({len(surge)}건)')
    fig2.add_scatter(x=drop['dt'], y=drop['hourly_pow'],
                     mode='markers', marker=dict(color='navy', size=5),
                     name=f'급락 ({len(drop)}건)')
    fig2.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown('**Isolation Forest**')
    iso_anomalies = anomaly[anomaly['iso_anomaly'] == True]
    iso_surge = iso_anomalies[iso_anomalies['hourly_pow'] > anomaly['hourly_pow'].mean()]
    iso_drop  = iso_anomalies[iso_anomalies['hourly_pow'] <= anomaly['hourly_pow'].mean()]

    fig3 = px.line(anomaly, x='dt', y='hourly_pow',
                   labels={'dt': '', 'hourly_pow': '전력(kW)'},
                   color_discrete_sequence=['steelblue'])
    fig3.update_traces(line_width=0.5)
    fig3.add_scatter(x=iso_surge['dt'], y=iso_surge['hourly_pow'],
                     mode='markers', marker=dict(color='crimson', size=5),
                     name=f'급등 ({len(iso_surge)}건)')
    fig3.add_scatter(x=iso_drop['dt'], y=iso_drop['hourly_pow'],
                     mode='markers', marker=dict(color='navy', size=5),
                     name=f'급락 ({len(iso_drop)}건)')
    fig3.update_layout(showlegend=True, height=300, margin=dict(t=10, b=10))
    fig3.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

# 이상탐지 요약 표
both_surge = anomaly[(anomaly['zscore_anomaly'] == True) & (anomaly['iso_anomaly'] == True) & (anomaly['z_score'] > 0)]
both_drop  = anomaly[(anomaly['zscore_anomaly'] == True) & (anomaly['iso_anomaly'] == True) & (anomaly['z_score'] < 0)]

summary_df = pd.DataFrame({
    '구분':             ['급등', '급락', '합계'],
    'Z-score':         [f'{len(surge)}건', f'{len(drop)}건', f'{len(surge)+len(drop)}건'],
    'Isolation Forest':[f'{len(iso_surge)}건', f'{len(iso_drop)}건', f'{len(iso_surge)+len(iso_drop)}건'],
    '교집합':           [f'{len(both_surge)}건', f'{len(both_drop)}건', f'{len(both_surge)+len(both_drop)}건'],
})

st.dataframe(summary_df, use_container_width=False, hide_index=True)
st.markdown('---')


# 설비별 이상탐지
st.subheader('🔧 설비별 이상탐지 현황')
st.caption('두 이상탐지 방법(Z-score · Isolation Forest)이 동시에 감지한 교집합만 표시. 좌측 막대 클릭 시 우측 타임라인 연동.')

if not equip_anomaly.empty:
    equip_summary = (
        equip_anomaly[equip_anomaly['zscore_anomaly'] == True]
        .groupby('equipment')
        .size()
        .reset_index(name='anomaly_count')
        .sort_values('anomaly_count', ascending=False)
    )


    col_c, col_d = st.columns([5, 5])

    with col_c:
        st.markdown('**설비별 이상 건수**')
        fig5 = px.bar(
            equip_summary,
            x='anomaly_count',
            y='equipment',
            orientation='h',
            color_discrete_sequence=['steelblue'],
            text='anomaly_count'
        )
        fig5.update_layout(
            margin=dict(t=10, b=40),
            xaxis_title='이상 건수',
            yaxis_title='',
            yaxis=dict(autorange='reversed'),
            height=350
        )
        fig5.update_traces(textposition='outside')
        selected_event = st.plotly_chart(fig5, use_container_width=True, on_select='rerun', key='equip_bar')

    with col_d:

        clicked = None
        if selected_event and selected_event.selection and selected_event.selection.points:
            clicked = selected_event.selection.points[0]['y']

        selected = clicked if clicked else equip_summary['equipment'].iloc[0]

        sel_all  = equip_anomaly[equip_anomaly['equipment'] == selected].sort_values('dt')
        sel_anom = sel_all[sel_all['zscore_anomaly'] == True]

        fig6 = px.line(sel_all, x='dt', y='hourly_pow',
                       labels={'dt': '', 'hourly_pow': '전력(kW)'},
                       color_discrete_sequence=['steelblue'])
        fig6.update_traces(line_width=0.8)
        fig6.add_scatter(x=sel_anom['dt'], y=sel_anom['hourly_pow'],
                         mode='markers', marker=dict(color='red', size=6),
                         name=f'이상 ({len(sel_anom)}건)')
        fig6.update_layout(
            showlegend=False,
            height=400,
            title=f'{selected}의 시간별 전력 추이 및 이상 시점',
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig6, use_container_width=True)

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