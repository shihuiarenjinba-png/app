import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from sklearn.decomposition import PCA
import io

# 将来の警告を無視する設定
warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================
# ⚙️ ページ設定 (最初に行う必要があります)
# =========================================================
st.set_page_config(page_title="Factor Simulator V19", layout="wide", page_icon="🧬")

# =========================================================
# 🔗 モジュール読み込みチェック
# =========================================================
try:
    from simulation_engine import MarketDataEngine, PortfolioAnalyzer, PortfolioDiagnosticEngine
    from pdf_generator import create_pdf_report
except ImportError as e:
    st.error(f"❌ 重要ファイルが見つかりません: {e}")
    st.info("app.py と同じフォルダに 'simulation_engine.py' と 'pdf_generator.py' があるか確認してください。")
    st.stop()

# 多言語辞書モジュールの読み込みチェック
try:
    from i18n import ja, en
except ImportError as e:
    st.error(f"❌ 翻訳ファイルが見つかりません: {e}")
    st.info("app.py と同じ階層に 'i18n' フォルダを作成し、'ja.py' と 'en.py' を配置してください。")
    st.stop()

# =========================================================
# 🎨 定数・スタイル設定
# =========================================================
COLORS = {
    'main': '#00FFFF',      # Neon Cyan
    'benchmark': '#FF69B4', # Hot Pink
    'principal': '#FFFFFF', # White
    'median': '#32CD32',    # Lime Green
    'mean': '#FFD700',      # Gold
    'p10': '#FF6347',       # Pessimistic
    'p90': '#00BFFF',       # Optimistic
    'hist_bar': '#42A5F5',  # Mid Blue
    'cost_net': '#FF6347',  # Tomato Red
    'bg_fill': 'rgba(0, 255, 255, 0.1)'
}

# CSSスタイリング
st.markdown("""
<style>
    .metric-card { background-color: #262730; border: 1px solid #444; padding: 15px; border-radius: 8px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1E1E1E; border-radius: 5px 5px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #00FFFF; color: black; font-weight: bold; }
    .report-box { border-left: 5px solid #00FFFF; padding-left: 15px; margin-top: 10px; background-color: rgba(0, 255, 255, 0.05); }
    .factor-box { border-left: 5px solid #FF69B4; padding-left: 15px; margin-top: 10px; background-color: rgba(255, 105, 180, 0.05); }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    h1, h2, h3 { color: #E0E0E0; font-family: 'Hiragino Kaku Gothic Pro', 'Meiryo', sans-serif; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🌍 共通言語辞書（Dictionary）の実装
# =========================================================
def t(key):
    lang = st.session_state.get('lang', 'JA')
    if lang == 'JA':
        return ja.TEXTS.get(key, key)
    else:
        return en.TEXTS.get(key, key)


def ui_text(ja_text, en_text):
    return ja_text if st.session_state.get('lang', 'JA') == 'JA' else en_text


ANALYSIS_REQUIREMENTS = {
    'info_ratio': {'min_months': 12, 'stable_months': 24},
    'factor_regression': {'min_months': 24, 'stable_months': 36},
    'rolling': {'min_months': 24, 'stable_months': 48},
    'distribution': {'min_months': 12, 'stable_months': 24},
    'simulation': {'min_months': 12, 'stable_months': 36},
    'pca': {'min_months': 12, 'stable_months': 24},
    'attribution': {'min_months': 12, 'stable_months': 24},
}


def quality_label(sample_count, key):
    req = ANALYSIS_REQUIREMENTS[key]
    if sample_count < req['min_months']:
        return ui_text("不足", "Insufficient")
    if sample_count < req['stable_months']:
        return ui_text("参考値", "Indicative")
    return ui_text("安定", "Stable")


def build_window_text(start_date, end_date, months):
    if not start_date or not end_date or not months:
        return ui_text("期間情報なし", "No window information")
    return ui_text(
        f"{start_date} から {end_date} / {months} ヶ月",
        f"{start_date} to {end_date} / {months} months",
    )


def count_factor_months(port_ret, factor_df):
    if port_ret.empty or factor_df is None or factor_df.empty:
        return 0
    return len(pd.concat([port_ret.to_frame(name='y'), factor_df], axis=1).dropna())


def dynamic_hist_bins(sample_count):
    if sample_count <= 8:
        return max(4, sample_count)
    if sample_count <= 24:
        return max(6, min(12, sample_count // 2))
    return min(40, max(10, int(np.sqrt(sample_count) * 2)))

# =========================================================
# 🛠️ セッション状態の初期化
# =========================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'JA'
if 'base_currency' not in st.session_state:
    st.session_state.base_currency = 'JPY'
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'payload' not in st.session_state:
    st.session_state.payload = None
if 'figs' not in st.session_state:
    st.session_state.figs = {}
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'practical'

# タイトル表示（辞書適用）
st.title(t('title'))
st.caption(t('caption'))

# =========================================================
# 🏗️ サイドバー: ポートフォリオ設定
# =========================================================
with st.sidebar:
    st.header(t('sidebar_global'))
    
    # 🌍 UI（サイドバー）の設置: 言語と通貨の切り替えスイッチ
    c_lang, c_curr = st.columns(2)
    with c_lang:
        selected_lang = st.selectbox("Language / 言語", ["JA", "EN"], index=0 if st.session_state.lang == 'JA' else 1)
    with c_curr:
        selected_curr = st.selectbox("Currency / 通貨", ["JPY", "USD"], index=0 if st.session_state.base_currency == 'JPY' else 1)
    
    # セッションの更新
    st.session_state.lang = selected_lang
    st.session_state.base_currency = selected_curr
    
    st.markdown("---")
    st.header(t('sidebar_settings'))
    st.markdown(f"### {t('sb_sec1')}")
    
    uploaded_file = st.file_uploader(t('sb_upload_csv'), type=['csv'], help=t('sb_upload_help'))
    default_input = "SPY: 40, VWO: 20, 7203.T: 20, GLD: 20"
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if df_upload.shape[1] >= 2:
                tickers_up = df_upload.iloc[:, 0].astype(str)
                weights_up = df_upload.iloc[:, 1].astype(str)
                formatted_list = [f"{t}: {w}" for t, w in zip(tickers_up, weights_up)]
                default_input = ", ".join(formatted_list)
                st.success(t('sb_csv_success'))
            else:
                st.error(t('sb_csv_error'))
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

    input_text = st.text_area(t('sb_ticker_input'), value=default_input, height=100)

    st.markdown(f"### {t('sb_sec2')}")
    target_region = st.selectbox(t('sb_region'), ["US (米国)", "Japan (日本)", "Global (全世界)"], index=0)
    region_code = target_region.split()[0]
    
    bench_options = {
        'US': {'S&P 500 (SPY)': 'SPY', 'NASDAQ 100 (^NDX)': '^NDX'},
        'Japan': {'TOPIX (1306 ETF)': '1306.T', '日経平均 (^N225)': '^N225'},
        'Global': {'VT (全世界株式)': 'VT', 'MSCI ACWI (指数)': 'ACWI'}
    }
    
    current_bench_options = list(bench_options[region_code].keys()) + ["Custom"]
    selected_bench_label = st.selectbox(t('sb_bench'), current_bench_options, index=0)

    if selected_bench_label == "Custom":
        bench_ticker = st.text_input(t('sb_custom_bench'), value="SPY")
    else:
        bench_ticker = bench_options[region_code][selected_bench_label]

    st.markdown(f"### {t('sb_sec3')}")
    cost_tier = st.select_slider(t('sb_cost_tier'), options=["Low", "Medium", "High"], value="Medium")

    st.markdown("### 🔄 リバランス設定 (Rebalance)")
    rebalance_label = st.selectbox(
        "実行頻度 (Frequency)",
        ["月次 (Monthly)", "四半期 (Quarterly)", "年次 (Yearly)", "なし (Buy & Hold)"],
        index=0 
    )
    rebalance_map = {"月次 (Monthly)": 'M', "四半期 (Quarterly)": 'Q', "年次 (Yearly)": 'Y', "なし (Buy & Hold)": None}
    rebalance_freq = rebalance_map[rebalance_label]

    st.markdown("### 🧪 分析モード")
    mode_labels = {
        'JA': {
            "Practical (実務用・柔軟)": 'practical',
            "Strict (研究用・共通期間)": 'strict',
        },
        'EN': {
            "Practical (Flexible)": 'practical',
            "Strict (Common Window)": 'strict',
        },
    }
    selected_mode_label = st.selectbox(
        ui_text("分析方針", "Analysis Policy"),
        list(mode_labels[st.session_state.lang].keys()),
        index=0 if st.session_state.analysis_mode == 'practical' else 1,
    )
    analysis_mode = mode_labels[st.session_state.lang][selected_mode_label]
    st.session_state.analysis_mode = analysis_mode
    st.caption(
        ui_text(
            "Practical はその月に存在する銘柄だけで計算します。Strict は全銘柄の共通期間だけで計算します。",
            "Practical uses only the assets available in each month. Strict uses only the common history shared by all assets.",
        )
    )

    st.markdown(f"### {t('sb_sec4')}")
    st.caption(t('sb_adv_caption'))
    
    default_note = t('default_advisor_note')
    advisor_note = st.text_area(t('sb_adv_label'), value=default_note, height=100)

    st.markdown("---")
    # Streamlit最新仕様に合わせて use_container_width=True を維持（ボタンの場合は許容されます）
    analyze_btn = st.button(t('btn_analyze'), type="primary", use_container_width=True)

# =========================================================
# 🚀 メインロジック (計算実行)
# =========================================================

if analyze_btn:
    with st.spinner(t('msg_fetching_data')):
        try:
            # 1. 入力解析
            raw_items = [item.strip() for item in input_text.split(',')]
            parsed_dict = {}
            for item in raw_items:
                try:
                    k, v = item.split(':')
                    parsed_dict[k.strip()] = float(v.strip())
                except ValueError: 
                    pass # フォーマットが不正な場合はスキップ

            if not parsed_dict: st.stop()

            # 🚀 Engine 呼び出し
            engine = MarketDataEngine()
            valid_assets, _ = engine.validate_tickers(parsed_dict)
            if not valid_assets:
                st.error(t('msg_err_no_ticker'))
                st.stop()

            tickers = list(valid_assets.keys())
            hist_returns = engine.fetch_historical_prices(
                tickers,
                base_currency=st.session_state.base_currency,
            )

            # 💡 除外銘柄のUIへの通知（クリーンアップ済み）
            fetched_tickers = list(hist_returns.columns)
            excluded_tickers = [t for t in tickers if t not in fetched_tickers]
            
            if excluded_tickers:
                st.warning(f"⚠️ **データ期間不足・取得エラーによる除外:** 以下の資産はデータが取得できなかったため分析から除外されました: **{', '.join(excluded_tickers)}**")
                
                for t in excluded_tickers:
                    del valid_assets[t]
                
                total_w = sum(v['weight'] for v in valid_assets.values())
                if total_w <= 0:
                    st.error("有効な資産が残っていません。分析を中止します。")
                    st.stop()
                
                for k in valid_assets.keys():
                    valid_assets[k]['weight'] /= total_w
                
                st.info("💡 残りの資産でウェイト比率を維持したまま再計算を行いました。")

            if hist_returns.empty:
                 st.error(t('msg_err_price_fetch'))
                 st.stop()

            # ベンチマーク取得
            is_jpy_bench = bench_ticker in ['^TPX', '^N225', '1306.T'] or bench_ticker.endswith('.T')
            bench_series = engine.fetch_benchmark_data(
                bench_ticker,
                is_jpy_asset=is_jpy_bench,
                base_currency=st.session_state.base_currency,
            )

            weights_clean = {k: v['weight'] for k, v in valid_assets.items()}
            
            # 合成ヒストリー作成
            port_series, final_weights, analysis_meta = PortfolioAnalyzer.create_synthetic_history(
                hist_returns,
                weights_clean,
                benchmark_ret=bench_series,
                rebalance_freq=rebalance_freq,
                analysis_mode=analysis_mode,
            )
            if port_series.empty:
                st.error(
                    ui_text(
                        "この設定では分析に必要な履歴を作れませんでした。分析モードや銘柄構成を見直してください。",
                        "This configuration could not produce enough history for analysis. Please review the analysis mode or asset mix.",
                    )
                )
                st.stop()

            # ファクター取得
            french_factors = engine.fetch_french_factors(region_code)
            factor_source = french_factors.attrs.get('factor_data_source', 'unavailable')
            analysis_components = hist_returns.loc[port_series.index]
            benchmark_common_months = len(port_series.index.intersection(bench_series.index)) if not bench_series.empty else 0
            factor_common_months = count_factor_months(port_series, french_factors)
            complete_component_months = len(analysis_components.dropna(how='any'))

            # データ保存
            st.session_state.portfolio_data = {
                'returns': port_series,
                'benchmark': bench_series,
                'components': analysis_components,
                'weights': final_weights,
                'factors': french_factors,
                'factor_source': factor_source,
                'asset_info': valid_assets,
                'cost_tier': cost_tier,
                'bench_name': selected_bench_label,
                'analysis_mode': analysis_mode,
                'analysis_meta': analysis_meta,
                'quality_summary': {
                    'portfolio_months': len(port_series),
                    'benchmark_common_months': benchmark_common_months,
                    'factor_common_months': factor_common_months,
                    'complete_component_months': complete_component_months,
                },
            }
            
            # キャッシュクリア
            st.session_state.pdf_bytes = None
            st.session_state.analysis_done = False
            st.session_state.figs = {}

        except Exception as e:
            st.error(f"{t('msg_err_analysis')}{e}")
            st.stop()

# =========================================================
# 📊 ダッシュボード表示 & PDF用データ準備
# =========================================================

if st.session_state.portfolio_data:
    data = st.session_state.portfolio_data
    analyzer = PortfolioAnalyzer()
    port_ret = data['returns']
    bench_ret = data['benchmark']
    analysis_meta = data.get('analysis_meta', {})
    quality_summary = data.get('quality_summary', {})
    analysis_mode = data.get('analysis_mode', 'practical')

    curr_unit = t('currency_jpy') if st.session_state.base_currency == 'JPY' else t('currency_usd')
    init_inv = 1000000 if st.session_state.base_currency == 'JPY' else 10000

    # --- 1. 基本指標 ---
    total_ret_cum = (1 + port_ret).cumprod()
    cagr = (total_ret_cum.iloc[-1])**(12/len(port_ret)) - 1
    vol = port_ret.std() * np.sqrt(12)
    max_dd = (total_ret_cum / total_ret_cum.cummax() - 1).min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    try:
        omega = analyzer.calculate_omega_ratio(port_ret, threshold=0.0)
    except Exception:
        omega = 0.0
        
    try:
        info_ratio, track_err = analyzer.calculate_information_ratio(port_ret, bench_ret)
    except Exception:
        info_ratio, track_err = np.nan, np.nan

    # ゼロ除算対策
    sharpe_ratio = (cagr - 0.02) / vol if vol > 0 else 0.0

    # --- 2. 高度計算 & 分析レポート ---
    params, r_sq, factor_sample_count = analyzer.perform_factor_regression(
        port_ret,
        data['factors'],
        min_months=ANALYSIS_REQUIREMENTS['factor_regression']['min_months'],
    )
    if params is not None and not params.empty:
        factor_comment = PortfolioDiagnosticEngine.generate_factor_report(params, lang=st.session_state.lang)
    else:
        factor_comment = ui_text(
            "ファクターデータまたは共通期間が不足しているため分析できません。",
            "Factor data or overlap is insufficient for regression.",
        )

    sim_years = 20
    simulation_months = len(port_ret.dropna())
    df_stats, final_values = (None, None)
    if simulation_months >= ANALYSIS_REQUIREMENTS['simulation']['min_months']:
        df_stats, final_values = analyzer.run_monte_carlo_simulation(
            port_ret,
            n_years=sim_years,
            n_simulations=7500,
            initial_investment=init_inv,
            random_seed=42,
        )
    
    final_median = np.median(final_values) if final_values is not None else 0.0
    final_p10 = np.percentile(final_values, 10) if final_values is not None else 0.0
    final_p90 = np.percentile(final_values, 90) if final_values is not None else 0.0
    
    component_returns = data['components']
    comp_clean_for_analysis = component_returns.dropna(how='any')
    
    corr_matrix = analyzer.calculate_correlation_matrix(
        component_returns,
        min_periods=ANALYSIS_REQUIREMENTS['pca']['min_months'],
    )
    fig_corr_report = None
    if not corr_matrix.empty:
        fig_corr_report = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)

    pca_available = len(comp_clean_for_analysis) >= ANALYSIS_REQUIREMENTS['pca']['min_months'] and comp_clean_for_analysis.shape[1] > 1
    if pca_available:
        pca_ratio, _ = analyzer.perform_pca(comp_clean_for_analysis)
        report = PortfolioDiagnosticEngine.generate_report(data['weights'], pca_ratio, port_ret, lang=st.session_state.lang)
    else:
        pca_ratio = np.nan
        report = {
            'type': ui_text("🧭 分散診断は保留", "🧭 Diversification diagnosis pending"),
            'diversification_comment': ui_text("PCA に必要な完全月数が不足しているため、分散診断はまだ参考表示にしません。", "PCA does not yet have enough complete history, so diversification diagnosis is deferred."),
            'risk_comment': ui_text("ポートフォリオ自体の計算は継続していますが、この診断だけは保守的に保留しています。", "The portfolio calculations still run, but this diagnosis is conservatively deferred."),
            'action_plan': ui_text("完全月数が増えたら自動で診断を再開できます。", "The diagnosis can resume automatically once the complete overlap grows."),
        }

    detailed_review = []
    if sharpe_ratio > 1.0:
        detailed_review.append(t('review_eff_high').format(sharpe=sharpe_ratio))
    elif sharpe_ratio > 0.6:
        detailed_review.append(t('review_eff_mid').format(sharpe=sharpe_ratio))
    else:
        detailed_review.append(t('review_eff_low').format(sharpe=sharpe_ratio))

    if vol < 0.12:
        detailed_review.append(t('review_vol_low').format(vol=vol))
    elif vol < 0.18:
        detailed_review.append(t('review_vol_mid').format(vol=vol))
    else:
        detailed_review.append(t('review_vol_high').format(vol=vol))

    detailed_review.append(t('review_dd').format(max_dd=max_dd))
    detailed_review_str = "\n".join(detailed_review)

    # 🛡️ Payload 作成
    st.session_state.payload = {
        'lang': st.session_state.lang,
        'currency': st.session_state.base_currency,
        'curr_unit': curr_unit,
        'raw_metrics': {'CAGR': cagr, 'Vol': vol, 'MaxDD': max_dd, 'Sharpe': sharpe_ratio},
        'raw_mc_stats': {'median': final_median, 'p10': final_p10, 'p90': final_p90, 'init_inv': init_inv},
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'metrics': {
            'CAGR': f"{cagr:.2%}", 'Vol': f"{vol:.2%}", 'MaxDD': f"{max_dd:.2%}",
            'Sharpe': f"{sharpe_ratio:.2f}", 'Calmar Ratio': f"{calmar:.2f}",
            'Information Ratio': f"{info_ratio:.2f}" if not np.isnan(info_ratio) else "N/A"
        },
        'factor_comment': factor_comment,
        'diagnosis': {
            'type': report['type'],
            'diversification_comment': report['diversification_comment'],
            'risk_comment': report['risk_comment'],
            'action_plan': report['action_plan']
        },
        'detailed_review': detailed_review_str,
        'mc_stats': t('pdf_mc_stats_values').format(median=final_median, p10=final_p10, p90=final_p90, curr=curr_unit)
    }

    figs_for_report = {}
    if fig_corr_report:
        figs_for_report['correlation'] = fig_corr_report

    st.markdown("---")
    
    # 💡 メトリクス同士の重なりを防ぐ
    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    c1.metric(t('metric_cagr'), f"{cagr:.2%}")
    c2.metric(t('metric_vol'), f"{vol:.2%}")
    c3.metric(t('metric_maxdd'), f"{max_dd:.2%}", delta_color="inverse")
    c4.metric(t('metric_sharpe'), f"{sharpe_ratio:.2f}")
    c5.metric(t('metric_omega'), f"{omega:.2f}")

    if not np.isnan(info_ratio):
        st.caption(t('cap_info_ratio').format(bench=data['bench_name'], info_ratio=info_ratio, track_err=track_err))
    else:
        st.caption(
            ui_text(
                f"ベンチマーク比較は {ANALYSIS_REQUIREMENTS['info_ratio']['min_months']} ヶ月以上で安定します。現在の共通月数は {quality_summary.get('benchmark_common_months', 0)} です。",
                f"Benchmark comparison stabilizes after {ANALYSIS_REQUIREMENTS['info_ratio']['min_months']} months. Current overlap: {quality_summary.get('benchmark_common_months', 0)} months.",
            )
        )

    mode_description = {
        'strict': ui_text("Strict: 全銘柄の共通期間だけで厳密計算", "Strict: common history across all assets"),
        'practical': ui_text("Practical: その月に存在する銘柄だけで柔軟計算", "Practical: flexible monthly available-assets mode"),
    }
    st.info(
        ui_text(
            f"分析モード: {mode_description.get(analysis_mode, analysis_mode)} / 使用期間: {build_window_text(analysis_meta.get('start_date'), analysis_meta.get('end_date'), analysis_meta.get('months'))}",
            f"Analysis mode: {mode_description.get(analysis_mode, analysis_mode)} / Window: {build_window_text(analysis_meta.get('start_date'), analysis_meta.get('end_date'), analysis_meta.get('months'))}",
        )
    )
    st.caption(
        ui_text(
            f"共通ベンチマーク月数: {quality_summary.get('benchmark_common_months', 0)} ({quality_label(quality_summary.get('benchmark_common_months', 0), 'info_ratio')}) / "
            f"ファクター回帰月数: {factor_sample_count} ({quality_label(factor_sample_count, 'factor_regression')}) / "
            f"PCA完全月数: {quality_summary.get('complete_component_months', 0)} ({quality_label(quality_summary.get('complete_component_months', 0), 'pca')})",
            f"Benchmark overlap: {quality_summary.get('benchmark_common_months', 0)} ({quality_label(quality_summary.get('benchmark_common_months', 0), 'info_ratio')}) / "
            f"Factor months: {factor_sample_count} ({quality_label(factor_sample_count, 'factor_regression')}) / "
            f"Complete PCA months: {quality_summary.get('complete_component_months', 0)} ({quality_label(quality_summary.get('complete_component_months', 0), 'pca')})",
        )
    )

    tabs = st.tabs(t('tab_names'))

    with tabs[0]:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.subheader(t('sub_pca'))
            if len(comp_clean_for_analysis) >= ANALYSIS_REQUIREMENTS['pca']['min_months'] and comp_clean_for_analysis.shape[1] > 1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = pca_ratio * 100, 
                    title = {'text': t('pca_gauge_title')},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['main']},
                             'steps': [{'range': [0, 60], 'color': "#333"}, {'range': [60, 100], 'color': "#555"}],
                             'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}
                ))
                st.plotly_chart(fig_gauge, width="stretch")
                st.caption(
                    ui_text(
                        f"PCA に使えた完全月数: {len(comp_clean_for_analysis)} ({quality_label(len(comp_clean_for_analysis), 'pca')})",
                        f"Complete months used for PCA: {len(comp_clean_for_analysis)} ({quality_label(len(comp_clean_for_analysis), 'pca')})",
                    )
                )
            else:
                st.info(
                    ui_text(
                        f"PCA は完全月数が {ANALYSIS_REQUIREMENTS['pca']['min_months']} ヶ月以上必要です。現在は {len(comp_clean_for_analysis)} ヶ月です。",
                        f"PCA needs at least {ANALYSIS_REQUIREMENTS['pca']['min_months']} complete months. Current: {len(comp_clean_for_analysis)} months.",
                    )
                )
            
            st.markdown(t('sub_pca_map'))
            try:
                if len(comp_clean_for_analysis) >= ANALYSIS_REQUIREMENTS['pca']['min_months'] and comp_clean_for_analysis.shape[1] > 1:
                    pca = PCA(n_components=2)
                    pca_coords = pca.fit_transform(comp_clean_for_analysis.T)
                    labels = comp_clean_for_analysis.columns
                    
                    fig_pca = px.scatter(x=pca_coords[:, 0], y=pca_coords[:, 1], text=labels, 
                                         color=labels, title=t('graph_pca'))
                    fig_pca.update_traces(textposition='top center', marker=dict(size=12))
                    fig_pca.update_layout(xaxis_title=t('pca_pc1'), yaxis_title=t('pca_pc2'), showlegend=False)
                    st.plotly_chart(fig_pca, width="stretch")
            except Exception as e:
                st.warning(f"{t('msg_err_pca')}{e}")

        with c2:
            st.subheader(t('graph_alloc'))
            fig_pie = px.pie(values=list(data['weights'].values()), names=list(data['weights'].keys()), hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, width="stretch")
            figs_for_report['allocation'] = fig_pie
            
            st.markdown("---")
            st.subheader(t('sub_ai_diag'))
            st.markdown(f"""
            <div class="report-box">
                <h3 style="color: #00FFFF; margin-bottom:0px;">{report['type']}</h3>
                <hr style="margin-top:5px; margin-bottom:10px; border-color: #555;">
                <p><b>🧐 {t('pdf_diag_div')}:</b><br>{report['diversification_comment']}</p>
                <p><b>⚠️ {t('pdf_diag_risk')}:</b><br>{report['risk_comment']}</p>
                <p><b>💡 {t('pdf_diag_action')}:</b><br>{report['action_plan']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if fig_corr_report:
            st.markdown("---")
            st.markdown(f"#### 🔥 {t('graph_corr')}")
            num_assets = len(data['components'].columns)
            corr_height = max(400, 200 + (num_assets * 30))
            fig_corr_report.update_layout(height=corr_height)
            st.plotly_chart(fig_corr_report, width="stretch")

    with tabs[1]:
        factor_source_messages = {
            'JA': {
                'datareader': "ファクターデータ取得元: pandas-datareader 経由の Ken French Data Library",
                'official_zip': "ファクターデータ取得元: Ken French 公式 ZIP ダウンロード",
                'local_cache': "ファクターデータ取得元: ローカルキャッシュ（通信失敗時の退避データ）",
            },
            'EN': {
                'datareader': "Factor data source: Ken French Data Library via pandas-datareader",
                'official_zip': "Factor data source: Ken French official ZIP download",
                'local_cache': "Factor data source: local cache fallback after network retrieval failed",
            },
        }
        factor_source_key = data.get('factor_source')
        if factor_source_key in factor_source_messages.get(st.session_state.lang, {}):
            st.caption(factor_source_messages[st.session_state.lang][factor_source_key])

        # 💡 ファクターデータが無い時の安全なガードレール
        if data['factors'].empty:
            err_msg = t('msg_err_factor')
            if err_msg == 'msg_err_factor': 
                err_msg = "ファクターデータの取得に失敗したため、このタブの分析はスキップされます。"
            st.warning(err_msg)
        else:
            st.subheader(t('sub_style'))
            if params is not None and not params.empty:
                c1, c2 = st.columns([1, 1], gap="medium")
                with c1:
                    beta_df = params.drop('const') if 'const' in params else params
                    colors = ['#00CC96' if x > 0 else '#FF4B4B' for x in beta_df.values]
                    fig_beta = go.Figure(go.Bar(
                        x=beta_df.values, y=beta_df.index, orientation='h', 
                        marker_color=colors, text=[f"{x:.2f}" for x in beta_df.values], textposition='auto'
                    ))
                    fig_beta.update_layout(title=t('graph_beta'), xaxis_title="感応度", height=300)
                    st.plotly_chart(fig_beta, width="stretch")
                    st.caption(
                        ui_text(
                            f"決定係数 (R²): {r_sq:.2%} / 回帰月数: {factor_sample_count} ({quality_label(factor_sample_count, 'factor_regression')})",
                            f"R²: {r_sq:.2%} / Regression months: {factor_sample_count} ({quality_label(factor_sample_count, 'factor_regression')})",
                        )
                    )
                    figs_for_report['factors'] = fig_beta
                
                with c2:
                    style_title = "🧠 AIスタイル分析" if st.session_state.lang == 'JA' else "🧠 AI Style Analysis"
                    st.markdown(f"""
                    <div class="factor-box">
                        <h4 style="color: #FF69B4; margin-bottom:10px;">{style_title}</h4>
                        <div style="white-space: pre-wrap;">{factor_comment}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(
                    ui_text(
                        f"回帰分析には少なくとも {ANALYSIS_REQUIREMENTS['factor_regression']['min_months']} ヶ月の共通データが必要です。現在は {factor_sample_count} ヶ月です。",
                        f"Factor regression needs at least {ANALYSIS_REQUIREMENTS['factor_regression']['min_months']} overlapping months. Current: {factor_sample_count}.",
                    )
                )
            
            st.markdown("---")
            st.subheader(t('sub_rolling'))
            rolling_betas, rolling_sample_count, rolling_window = analyzer.rolling_beta_analysis(
                port_ret,
                data['factors'],
                window=24,
                min_months=ANALYSIS_REQUIREMENTS['rolling']['min_months'],
            )
            
            if not rolling_betas.empty:
                fig_roll = go.Figure()
                cols = rolling_betas.columns
                if 'Mkt-RF' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['Mkt-RF'], name=t('factor_mkt'), line=dict(width=3, color=COLORS['main'])))
                if 'SMB' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['SMB'], name=t('factor_smb'), line=dict(dash='dot', color='orange')))
                if 'HML' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['HML'], name=t('factor_hml'), line=dict(dash='dot', color='yellow')))
                if 'RMW' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['RMW'], name=t('factor_rmw'), line=dict(dash='dot', color='#00FF00')))
                if 'CMA' in cols: 
                    fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas['CMA'], name=t('factor_cma'), line=dict(dash='dot', color='#FF00FF')))

                if not any(x in cols for x in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']):
                    for c in cols:
                        fig_roll.add_trace(go.Scatter(x=rolling_betas.index, y=rolling_betas[c], name=c))

                fig_roll.update_layout(title=t('graph_roll'), yaxis_title="Beta", height=400)
                st.plotly_chart(fig_roll, width="stretch")
                if rolling_window:
                    st.caption(
                        ui_text(
                            f"ローリング窓: {rolling_window} ヶ月 / 利用月数: {rolling_sample_count}",
                            f"Rolling window: {rolling_window} months / Sample count: {rolling_sample_count}",
                        )
                    )
            else:
                st.info(
                    ui_text(
                        f"ローリング分析には少なくとも {ANALYSIS_REQUIREMENTS['rolling']['min_months']} ヶ月が必要です。現在は {rolling_sample_count} ヶ月です。",
                        f"Rolling analysis needs at least {ANALYSIS_REQUIREMENTS['rolling']['min_months']} months. Current: {rolling_sample_count}.",
                    )
                )

    with tabs[2]:
        st.subheader(t('graph_hist'))
        
        if not bench_ret.empty:
            common_idx = port_ret.index.intersection(bench_ret.index)
            port_ret_sync = port_ret.loc[common_idx]
            bench_ret_sync = bench_ret.loc[common_idx]
        else:
            port_ret_sync = port_ret
            bench_ret_sync = pd.Series(dtype=float)

        # 💡 ここのスケール計算（* 10000）は現状で大正解です。そのまま活かします。
        cum_ret_sync = (1 + port_ret_sync).cumprod() * 10000
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=cum_ret_sync.index, y=[10000]*len(cum_ret_sync), 
            mode='lines', name=f"{t('label_principal')} (10,000)", 
            line=dict(color=COLORS['principal'], width=1, dash='dot')
        ))

        if not bench_ret_sync.empty:
            bench_cum_sync = (1 + bench_ret_sync).cumprod() * 10000
            fig_hist.add_trace(go.Scatter(
                x=bench_cum_sync.index, y=bench_cum_sync, 
                mode='lines', name=t('leg_bench').format(bench=data['bench_name']), 
                line=dict(color=COLORS['benchmark'], width=1.5)
            ))

        fig_hist.add_trace(go.Scatter(
            x=cum_ret_sync.index, y=cum_ret_sync, fill='tozeroy', 
            fillcolor=COLORS['bg_fill'], mode='lines', name=t('leg_port'), 
            line=dict(color=COLORS['main'], width=2.5)
        ))
        
        st.plotly_chart(fig_hist, width="stretch")
        figs_for_report['cumulative'] = fig_hist

        fig_dd = go.Figure()
        dd_series = (cum_ret_sync / cum_ret_sync.cummax() - 1)
        fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', name='Drawdown', line=dict(color='red')))
        fig_dd.update_layout(title=t('graph_dd'))
        st.plotly_chart(fig_dd, width="stretch")
        figs_for_report['drawdown'] = fig_dd

        st.markdown("---")
        st.subheader(t('sub_ret_dist'))
        mu, std = port_ret.mean(), port_ret.std()
        dist_sample_count = len(port_ret.dropna())

        if dist_sample_count < ANALYSIS_REQUIREMENTS['distribution']['min_months']:
            st.info(
                ui_text(
                    f"ヒストグラムは少なくとも {ANALYSIS_REQUIREMENTS['distribution']['min_months']} ヶ月ほしいです。現在は {dist_sample_count} ヶ月なので参考表示に留めます。",
                    f"The histogram works best with at least {ANALYSIS_REQUIREMENTS['distribution']['min_months']} months. Current: {dist_sample_count} months.",
                )
            )
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=port_ret, 
            histnorm='probability density', 
            name=t('leg_hist_ret'), 
            marker_color=COLORS['hist_bar'], 
            opacity=0.75, 
            nbinsx=dynamic_hist_bins(dist_sample_count)
        ))
        
        if dist_sample_count >= ANALYSIS_REQUIREMENTS['distribution']['min_months'] and not np.isnan(std) and std > 0:
            x_range = np.linspace(port_ret.min(), port_ret.max(), 100)
            y_norm = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x_range - mu) / std) ** 2)
            fig_dist.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', name=t('leg_norm_dist'), line=dict(color='white', dash='dash', width=2)))
        
        fig_dist.update_layout(title=t('graph_dist'), xaxis_title=t('dist_ret'), yaxis_title=t('dist_density'), height=400)
        st.plotly_chart(fig_dist, width="stretch")
        st.caption(
            ui_text(
                f"ヒストグラム月数: {dist_sample_count} ({quality_label(dist_sample_count, 'distribution')})",
                f"Histogram months: {dist_sample_count} ({quality_label(dist_sample_count, 'distribution')})",
            )
        )

    with tabs[3]:
        st.subheader(t('sub_cost_sim'))
        
        sim_res = analyzer.cost_drag_simulation(port_ret, data['cost_tier'])
        if len(sim_res) == 4:
            gross, net, loss, cost_pct = sim_res
        else:
            gross, net, loss = sim_res
            cost_pct = 0.0 
        
        loss_amount = init_inv * loss
        final_amount_net = init_inv * net.iloc[-1]
        
        c1, c2 = st.columns([3, 1], gap="medium")
        with c1:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=net.index, y=net, 
                mode='lines', stackgroup='one', 
                name=t('leg_net_asset'), 
                line=dict(color=COLORS['main'], width=2),
                fillcolor='rgba(0, 255, 255, 0.2)'
            ))
            loss_series = gross - net
            fig_cost.add_trace(go.Scatter(
                x=gross.index, y=loss_series, 
                mode='lines', stackgroup='one', 
                name=t('leg_cost_loss'), 
                line=dict(color='rgba(255, 99, 71, 0.5)', width=0),
                fillcolor='rgba(255, 99, 71, 0.3)'
            ))
            
            fig_cost.update_layout(title=t('graph_cost'), xaxis_title=t('label_months'), yaxis_title=t('label_multiple'))
            st.plotly_chart(fig_cost, width="stretch")
            
        with c2:
            st.error(t('msg_lost_val').format(loss_amount=f"{loss_amount:,.0f}", curr_unit=curr_unit))
            st.markdown(t('msg_final_val').format(init_inv=f"{init_inv:,.0f}", final_amount=f"{final_amount_net:,.0f}", curr_unit=curr_unit))
            st.info(t('msg_est_cost').format(cost_pct=f"{cost_pct:.2%}"))

    with tabs[4]:
        st.subheader(t('sub_attr'))
        attrib_input = data['components'].dropna(how='any')
        attrib = pd.Series(dtype=float)
        if len(attrib_input) >= ANALYSIS_REQUIREMENTS['attribution']['min_months']:
            attrib = analyzer.calculate_strict_attribution(attrib_input, data['weights'])
        
        if not attrib.empty:
            weights_series = pd.Series(data['weights'])
            common_idx = weights_series.index.intersection(attrib.index)
            
            total_risk = attrib[common_idx].sum()
            if total_risk != 0:
                r_relative = (attrib[common_idx] / total_risk) * 100
            else:
                r_relative = attrib[common_idx] * 0
            w_aligned = weights_series[common_idx] * 100 
            
            r_absolute = attrib[common_idx] * 100

            st.markdown(t('sub_attr_rel'))
            st.caption(t('cap_attr_rel'))
            
            fig_rel = go.Figure()
            fig_rel.add_trace(go.Bar(
                y=w_aligned.index, x=w_aligned.values, 
                name=t('leg_alloc'), orientation='h', 
                marker_color='rgba(200, 200, 200, 0.6)'
            ))
            fig_rel.add_trace(go.Bar(
                y=r_relative.index, x=r_relative.values, 
                name=t('leg_rel_risk'), orientation='h', 
                marker_color=COLORS['hist_bar']
            ))
            
            dynamic_height = max(400, 100 + (len(w_aligned) * 30))
            fig_rel.update_layout(
                barmode='group', 
                title=t('graph_attr_rel'),
                xaxis_title=t('label_ratio'),
                yaxis={'categoryorder':'total ascending'},
                height=dynamic_height
            )
            st.plotly_chart(fig_rel, width="stretch")
            
            st.markdown(t('sub_attr_abs'))
            st.caption(t('cap_attr_abs'))
            
            fig_abs = go.Figure()
            fig_abs.add_trace(go.Bar(
                y=r_absolute.index, x=r_absolute.values, 
                name=t('leg_abs_risk'), orientation='h', 
                marker_color='#FF6347'
            ))
            fig_abs.update_layout(
                title=t('graph_attr_abs'),
                xaxis_title=t('label_risk'),
                yaxis={'categoryorder':'total ascending'},
                height=dynamic_height
            )
            st.plotly_chart(fig_abs, width="stretch")

            figs_for_report['attribution'] = fig_rel
        else:
            st.info(
                ui_text(
                    f"寄与度分析には完全な月次履歴が {ANALYSIS_REQUIREMENTS['attribution']['min_months']} ヶ月以上必要です。現在は {len(attrib_input)} ヶ月です。",
                    f"Attribution needs at least {ANALYSIS_REQUIREMENTS['attribution']['min_months']} complete monthly observations. Current: {len(attrib_input)}.",
                )
            )

    with tabs[5]:
        st.subheader(t('sub_mc'))
        if df_stats is not None:
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p50'], mode='lines', name=t('leg_mc_med'), line=dict(color=COLORS['median'], width=3)))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p10'], mode='lines', name=t('leg_mc_p10'), line=dict(color=COLORS['p10'], width=1, dash='dot')))
            fig_mc.add_trace(go.Scatter(x=df_stats.index, y=df_stats['p90'], mode='lines', name=t('leg_mc_p90'), line=dict(color=COLORS['p90'], width=1, dash='dot')))
            
            fig_mc.update_layout(title=f"{t('graph_mc')} ({t('label_principal')}: {init_inv:,} {curr_unit})", yaxis_title=f"{t('label_val')} ({curr_unit})", height=500)
            st.plotly_chart(fig_mc, width="stretch")
            figs_for_report['monte_carlo'] = fig_mc

            st.markdown(t('sub_mc_dist'))
            final_mean = np.mean(final_values)

            mc1, mc2, mc3, mc4 = st.columns(4, gap="small")
            mc1.metric(t('mc_pes'), f"{final_p10:,.0f}", delta_color="inverse")
            mc2.metric(t('mc_med'), f"{final_median:,.0f}")
            mc3.metric(t('mc_mean'), f"{final_mean:,.0f}")
            mc4.metric(t('mc_opt'), f"{final_p90:,.0f}")

            fig_mc_hist = go.Figure()
            counts, _ = np.histogram(final_values, bins=100)
            y_max_freq = counts.max()
            x_max_view = np.percentile(final_values, 98)

            fig_mc_hist.add_trace(go.Histogram(
                x=final_values, nbinsx=100, name=t('mc_freq'), 
                marker_color=COLORS['hist_bar'], opacity=0.85
            ))
            
            lines_config = [
                (final_p10, COLORS['p10'], t('mc_pes_label').format(val=f"{final_p10:,.0f}"), 1.05, "dash", 2),
                (final_median, COLORS['median'], t('mc_med_label').format(val=f"{final_median:,.0f}"), 1.25, "solid", 3), 
                (final_mean, COLORS['mean'], t('mc_mean_label').format(val=f"{final_mean:,.0f}"), 1.15, "dot", 2),      
                (final_p90, COLORS['p90'], t('mc_opt_label').format(val=f"{final_p90:,.0f}"), 1.05, "dash", 2),
            ]
            
            for val, color, label, h_rate, dash, width in lines_config:
                fig_mc_hist.add_vline(x=val, line_width=width, line_dash=dash, line_color=color)
                fig_mc_hist.add_annotation(
                    x=val, y=y_max_freq * h_rate,
                    text=label, showarrow=False, font=dict(color=color)
                )

            fig_mc_hist.update_layout(
                xaxis_title=f"{t('label_final_val')} ({curr_unit})", yaxis_title=t('mc_freq'), showlegend=False,
                xaxis=dict(range=[0, x_max_view]), 
                yaxis=dict(range=[0, y_max_freq * 1.4])
            )
            st.plotly_chart(fig_mc_hist, width="stretch")
            
            st.success(t('msg_sim_complete'))
            st.caption(
                ui_text(
                    f"モンテカルロ月数: {simulation_months} ({quality_label(simulation_months, 'simulation')}) / 固定シード 42",
                    f"Monte Carlo months: {simulation_months} ({quality_label(simulation_months, 'simulation')}) / fixed seed 42",
                )
            )
        else:
            st.info(
                ui_text(
                    f"モンテカルロには少なくとも {ANALYSIS_REQUIREMENTS['simulation']['min_months']} ヶ月ほしいです。現在は {simulation_months} ヶ月です。",
                    f"Monte Carlo needs at least {ANALYSIS_REQUIREMENTS['simulation']['min_months']} months. Current: {simulation_months}.",
                )
            )

    # セッションへの保存 (分析完了フラグとグラフデータ)
    st.session_state.analysis_done = True
    st.session_state.figs = figs_for_report

# =========================================================
# 📄 PDF ダウンロードセクション
# =========================================================
st.markdown("---")

if st.session_state.analysis_done:
    st.header(t('pdf_section_title'))
    st.caption(t('pdf_section_caption'))

    col_gen, col_dl = st.columns([1, 1], gap="medium")

    with col_gen:
        if st.button(t('btn_generate_pdf')):
            with st.spinner(t('msg_pdf_spinning')):
                try:
                    final_payload = st.session_state.payload.copy()
                    # ※もし前半コードで advisor_note を定義していない場合は適宜空文字にしてください
                    final_payload['advisor_note'] = advisor_note if 'advisor_note' in locals() else ""
                    
                    if final_payload and st.session_state.figs:
                        pdf_buffer = create_pdf_report(final_payload, st.session_state.figs)
                        
                        if pdf_buffer:
                            st.session_state.pdf_bytes = pdf_buffer.getvalue()
                            st.success(f"{t('msg_pdf_ready')} ({len(st.session_state.pdf_bytes):,} bytes)")
                        else:
                            st.error(t('msg_pdf_err_empty'))
                    else:
                        st.error(t('msg_pdf_err_nodata'))
                        
                except Exception as e:
                    st.error(f"{t('msg_pdf_err_gen')}{e}")

    with col_dl:
        if st.session_state.pdf_bytes is not None:
            st.download_button(
                label=t('btn_download_pdf'),
                data=st.session_state.pdf_bytes,
                file_name="Portfolio_Analysis_Report.pdf",
                mime="application/pdf",
                type="primary"
            )

else:
    st.info(t('msg_pdf_hint'))
