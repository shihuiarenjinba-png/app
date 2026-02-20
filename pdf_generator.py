import io
import os
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# 🔻追加: 多言語辞書モジュールの読み込み
try:
    from i18n import ja, en
except ImportError as e:
    st.error(f"⚠️ 翻訳モジュールの読み込みエラー: {e}")

def create_pdf_report(payload, figs_dict):
    """
    app.py から受け取ったデータ(payload)とグラフ(figs_dict)を元にPDFを作成する
    """
    # 言語設定の取得 (未指定時は 'JA')
    lang = payload.get('lang', 'JA')
    
    # 🔻修正: 外部の i18n モジュールからテキストを取得するヘルパー関数
    def t(key):
        lang_upper = str(lang).upper()
        if lang_upper == 'JA':
            return ja.TEXTS.get(key, key)
        else:
            return en.TEXTS.get(key, key)

    buffer = io.BytesIO()
    
    # 1. ドキュメント設定
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40,
        title="Portfolio Report"
    )
    
    # 2. 日本語フォント登録 (ReportLab用: 本文の表示に必要)
    # ※ここは既存の文字化け対策をそのまま完全に保持しています
    base_dir = os.path.dirname(os.path.abspath(__file__))
    font_filename = "ipaexg.ttf"
    font_path = os.path.join(base_dir, font_filename)
    
    font_name = 'IPAexGothic'
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except:
        # 万が一見つからない場合、カレントディレクトリも探す
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_filename))
        except:
            st.error(f"⚠️ フォントファイル '{font_filename}' が見つかりません。pdf_generator.pyと同じ場所に置いてください。")
            return None

    # 3. スタイル定義
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('JpTitle', parent=styles['Title'], fontName=font_name, fontSize=24, leading=30, spaceAfter=20)
    heading_style = ParagraphStyle('JpHeading', parent=styles['Heading2'], fontName=font_name, fontSize=14, leading=18, spaceBefore=15, spaceAfter=10, textColor=colors.darkblue)
    normal_style = ParagraphStyle('JpNormal', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=10)
    alert_style = ParagraphStyle('JpAlert', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14, textColor=colors.firebrick, spaceAfter=10)
    small_style = ParagraphStyle('JpSmall', parent=styles['Normal'], fontName=font_name, fontSize=9, leading=12, textColor=colors.gray, spaceAfter=5)
    # メッセージ用の強調スタイル (保持)
    message_style = ParagraphStyle('JpMessage', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=10, backColor=colors.aliceblue, borderColor=colors.steelblue, borderWidth=1, borderPadding=10, borderRadius=5)

    # 4. コンテンツ構築
    story = []

    # --- ヘッダー ---
    # 🔻修正: 辞書キーを i18n モジュールに合わせて変更
    story.append(Paragraph(t('pdf_title'), title_style))
    story.append(Paragraph(f"{t('pdf_date')}{payload.get('date', '-')}", normal_style))
    story.append(Spacer(1, 20))

    # --- アドバイザーメッセージ ---
    if 'advisor_note' in payload and payload['advisor_note']:
        story.append(Paragraph(t('pdf_advisor_title'), heading_style))
        note_content = payload['advisor_note'].replace('\n', '<br/>')
        story.append(Paragraph(note_content, message_style))
        story.append(Spacer(1, 15))

    # --- 第1章: サマリー ---
    story.append(Paragraph(t('pdf_ch1_title'), heading_style))
    
    # 基本メトリクス (言語ごとのフォーマットに代入)
    summary_text = t('pdf_summary_text').format(
        cagr=payload['metrics']['CAGR'],
        vol=payload['metrics']['Vol'],
        sharpe=payload['metrics']['Sharpe'],
        maxdd=payload['metrics']['MaxDD']
    )
    story.append(Paragraph(summary_text, normal_style))
    
    # モンテカルロ統計 (あれば表示)
    if 'mc_stats' in payload:
        story.append(Paragraph(f"{t('pdf_mc_stats_title')}{payload['mc_stats']}", small_style))

    # AI詳細レビュー
    if 'detailed_review' in payload:
        story.append(Spacer(1, 5))
        for line in payload['detailed_review'].split('\n'):
            story.append(Paragraph(line, normal_style))

    story.append(Spacer(1, 10))

    # --- 第2章: AI診断 ---
    story.append(Paragraph(t('pdf_ch2_title'), heading_style))
    diag = payload.get('diagnosis', {})
    if diag:
        story.append(Paragraph(f"<b>{t('pdf_diag_type')}: {diag.get('type', '-')}</b>", normal_style))
        story.append(Paragraph(f"{t('pdf_diag_div')}: {diag.get('diversification_comment', '-')}", normal_style))
        story.append(Paragraph(f"{t('pdf_diag_risk')}: {diag.get('risk_comment', '-')}", alert_style))
        story.append(Paragraph(f"{t('pdf_diag_action')}: {diag.get('action_plan', '-')}", normal_style))

    if 'factor_comment' in payload:
        story.append(Spacer(1, 10))
        story.append(Paragraph(t('pdf_factor_title'), normal_style))
        story.append(Paragraph(payload['factor_comment'], normal_style))

    story.append(PageBreak())

    # --- 第3章: チャート ---
    story.append(Paragraph(t('pdf_ch3_title'), heading_style))
    story.append(Paragraph(t('pdf_ch3_desc'), normal_style))
    story.append(Spacer(1, 10))

    # グラフの表示順序とタイトル定義
    plot_order = ['allocation', 'correlation', 'monte_carlo', 'cumulative', 'drawdown', 'factors', 'attribution']
    
    # 🔻修正: title_mapをモジュールから動的に構築
    title_map = {
        'allocation': t('pdf_map_alloc'),
        'correlation': t('pdf_map_corr'),
        'monte_carlo': t('pdf_map_mc'),
        'cumulative': t('pdf_map_cum'),
        'drawdown': t('pdf_map_dd'),
        'factors': t('pdf_map_factor'),
        'attribution': t('pdf_map_attr')
    }

    for key in plot_order:
        if key in figs_dict:
            # タイトル追加
            story.append(Paragraph(title_map.get(key, f"■ {key}"), heading_style))
            
            try:
                fig = figs_dict[key]
                
                # -------------------------------------------------------
                # 【重要修正】 グラフ画像のフォント設定 (言語で分岐)
                # -------------------------------------------------------
                if lang == 'JA':
                    # 日本語環境: 既存の文字化け対策をそのまま使用
                    fig.update_layout(
                        font=dict(family="Noto Sans CJK JP, sans-serif"),
                        title_font=dict(family="Noto Sans CJK JP, sans-serif")
                    )
                else:
                    # 英語環境: 標準的な英字フォントを使用してスタイリッシュに
                    fig.update_layout(
                        font=dict(family="Arial, Helvetica, sans-serif"),
                        title_font=dict(family="Arial, Helvetica, sans-serif")
                    )
                
                # -------------------------------------------------------
                # 5ファクターが重ならず綺麗に収まるようサイズ調整 (保持)
                # -------------------------------------------------------
                chart_width = 900
                chart_height = 500
                
                if key == 'factors':
                    # 5項目に増えたため、高さを拡張し、英語ラベルが切れないよう左余白(l)を多めに確保
                    chart_height = 650
                    fig.update_layout(margin=dict(t=50, b=50, l=180, r=50))
                
                img_bytes = fig.to_image(format="png", width=chart_width, height=chart_height, scale=2)
                img_io = io.BytesIO(img_bytes)
                
                # PDF上のサイズ (アスペクト比を維持しつつA4に収める)
                render_width = 460
                render_height = int(render_width * (chart_height / chart_width))
                im = RLImage(img_io, width=render_width, height=render_height) 
                story.append(im)
                story.append(Spacer(1, 15))
                
                # ページ区切りの調整
                if key in ['monte_carlo', 'drawdown', 'correlation']: 
                    story.append(PageBreak())
                    
            except Exception as e:
                # 画像生成に失敗してもPDF作成自体は止めない
                story.append(Paragraph(f"{t('pdf_plot_err')}{e}", alert_style))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDFビルドエラー: {e}")
        return None
