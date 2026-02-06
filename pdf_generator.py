import io
import os
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

def create_pdf_report(payload, figs_dict):
    """
    app.py から受け取ったデータ(payload)とグラフ(figs_dict)を元にPDFを作成する
    """
    buffer = io.BytesIO()
    
    # 1. ドキュメント設定
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40,
        title="Portfolio Report"
    )
    
    # 2. 日本語フォント登録 (パス取得ロジックを強化)
    # このファイル(pdf_generator.py)と同じ階層にある ipaexg.ttf を探しに行きます
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
    # メッセージ用の強調スタイルを追加
    message_style = ParagraphStyle('JpMessage', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=10, backColor=colors.aliceblue, borderColor=colors.steelblue, borderWidth=1, borderPadding=10, borderRadius=5)

    # 4. コンテンツ構築
    story = []

    # --- ヘッダー ---
    story.append(Paragraph("ポートフォリオ詳細分析レポート", title_style))
    story.append(Paragraph(f"作成日: {payload.get('date', '-')}", normal_style))
    story.append(Spacer(1, 20))

    # --- 【追加】アドバイザーメッセージ ---
    # app.py から送られてきた advisor_note があればここに表示
    if 'advisor_note' in payload and payload['advisor_note']:
        story.append(Paragraph("▼ アドバイザーからのメッセージ", heading_style))
        # 改行コードを <br/> に変換して、複数行メッセージに対応
        note_content = payload['advisor_note'].replace('\n', '<br/>')
        story.append(Paragraph(note_content, message_style))
        story.append(Spacer(1, 15))

    # --- 第1章: サマリー ---
    story.append(Paragraph("1. 分析サマリー", heading_style))
    
    # 基本メトリクス
    summary_text = f"""
    本ポートフォリオの年平均成長率(CAGR)は <b>{payload['metrics']['CAGR']}</b>、
    リスク(Volatility)は <b>{payload['metrics']['Vol']}</b> です。
    シャープレシオは <b>{payload['metrics']['Sharpe']}</b> を記録しており、
    最大ドローダウンは <b>{payload['metrics']['MaxDD']}</b> と予測されます。
    """
    story.append(Paragraph(summary_text, normal_style))
    
    # モンテカルロ統計 (あれば表示)
    if 'mc_stats' in payload:
        story.append(Paragraph(f"<b>将来シミュレーション(20年後):</b> {payload['mc_stats']}", small_style))

    # AI詳細レビュー
    if 'detailed_review' in payload:
        story.append(Spacer(1, 5))
        for line in payload['detailed_review'].split('\n'):
            story.append(Paragraph(line, normal_style))

    story.append(Spacer(1, 10))

    # --- 第2章: AI診断 ---
    story.append(Paragraph("2. AI ポートフォリオ診断", heading_style))
    diag = payload.get('diagnosis', {})
    if diag:
        story.append(Paragraph(f"<b>タイプ判定: {diag.get('type', '-')}</b>", normal_style))
        story.append(Paragraph(f"分散状況: {diag.get('diversification_comment', '-')}", normal_style))
        story.append(Paragraph(f"リスク評価: {diag.get('risk_comment', '-')}", alert_style))
        story.append(Paragraph(f"アクションプラン: {diag.get('action_plan', '-')}", normal_style))

    if 'factor_comment' in payload:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>▼ ファクター特性分析</b>", normal_style))
        story.append(Paragraph(payload['factor_comment'], normal_style))

    story.append(PageBreak())

    # --- 第3章: チャート ---
    story.append(Paragraph("3. 詳細チャート分析", heading_style))
    story.append(Paragraph("以下に主要な分析チャートを示します。", normal_style))
    story.append(Spacer(1, 10))

    # グラフの表示順序とタイトル定義
    # app.pyで生成されているキー: allocation, correlation, factors, cumulative, drawdown, attribution, monte_carlo
    plot_order = ['allocation', 'correlation', 'monte_carlo', 'cumulative', 'drawdown', 'factors', 'attribution']
    
    title_map = {
        'allocation': '■ 資産配分 (Allocation)',
        'correlation': '■ 相関マトリックス (Correlation)',
        'monte_carlo': '■ 将来シミュレーション (Monte Carlo)',
        'cumulative': '■ 累積リターン推移 (Cumulative Return)',
        'drawdown': '■ ドローダウン (Drawdown)',
        'factors': '■ ファクター感応度 (Factor Exposure)',
        'attribution': '■ 寄与度分析 (Attribution)'
    }

    for key in plot_order:
        if key in figs_dict:
            # タイトル追加
            story.append(Paragraph(title_map.get(key, f"■ {key}"), heading_style))
            
            try:
                # Plotly -> 画像変換
                fig = figs_dict[key]
                # 画像サイズ調整 (A4横幅に合わせるためwidth=800, scale=2などで高画質化して縮小表示)
                img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
                img_io = io.BytesIO(img_bytes)
                
                # PDF上のサイズ (アスペクト比を維持しつつA4に収める)
                im = RLImage(img_io, width=460, height=255) 
                story.append(im)
                story.append(Spacer(1, 15))
                
                # ページ区切りの調整 (大きなグラフの後は改ページを入れると見やすい)
                if key in ['monte_carlo', 'drawdown', 'correlation']: 
                    story.append(PageBreak())
                    
            except Exception as e:
                # 画像生成に失敗してもPDF作成自体は止めない
                story.append(Paragraph(f"※グラフ生成エラー: {e}", alert_style))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDFビルドエラー: {e}")
        return None
