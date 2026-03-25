import os
import io
import base64
from datetime import datetime
import pandas as pd
import plotly.io as pio
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ── Color palette ─────────────────────────────────────────────
PRIMARY = colors.HexColor("#1E3A5F")
ACCENT = colors.HexColor("#4F8EF7")
LIGHT_BG = colors.HexColor("#F0F4FA")
SUCCESS = colors.HexColor("#28A745")
WARNING = colors.HexColor("#FFC107")
DANGER = colors.HexColor("#DC3545")
TEXT = colors.HexColor("#222222")


def fig_to_image(fig, width=16 * cm, height=8 * cm):
    """Convert a Plotly figure to a ReportLab Image object."""
    try:
        img_bytes = pio.to_image(fig, format="png", width=900, height=450, scale=1.5)
        buf = io.BytesIO(img_bytes)
        return RLImage(buf, width=width, height=height)
    except Exception:
        return None


def build_styles():
    """Return custom paragraph styles."""
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", fontSize=26, textColor=PRIMARY,
            fontName="Helvetica-Bold", spaceAfter=6, alignment=TA_CENTER
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontSize=11, textColor=colors.grey,
            spaceAfter=20, alignment=TA_CENTER
        ),
        "h1": ParagraphStyle(
            "h1", fontSize=16, textColor=PRIMARY,
            fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=8
        ),
        "h2": ParagraphStyle(
            "h2", fontSize=13, textColor=ACCENT,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=6
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, textColor=TEXT,
            leading=15, spaceAfter=6
        ),
        "small": ParagraphStyle(
            "small", fontSize=9, textColor=colors.grey,
            spaceAfter=4
        ),
    }
    return styles


def build_report(
    filename,
    profile,
    univariate_results=None,
    bivariate_results=None,
    cluster_results=None,
    assoc_results=None,
    time_results=None,
    narrative=None,
):
    """
    Build the full PDF report and return bytes.

    Args:
        filename: Original uploaded filename
        profile: Output from profiler.profile_dataframe()
        univariate_results: Output from univariate.run_univariate_analysis()
        bivariate_results: Output from bivariate.run_bivariate_analysis()
        cluster_results: Output from clustering.run_clustering()
        assoc_results: Output from associations.run_association_analysis()
        time_results: Output from time_analysis.run_time_analysis()
        narrative: AI-generated executive summary string

    Returns:
        bytes of the PDF
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )
    styles = build_styles()
    story = []

    # ── Cover ─────────────────────────────────────────────────
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph("🤖 AutoAnalyst AI", styles["title"]))
    story.append(Paragraph("Automated Data Analysis Report", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"File: <b>{filename}</b>", styles["body"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles["small"]))
    story.append(Spacer(1, 1 * cm))

    # ── Metrics table ─────────────────────────────────────────
    metrics_data = [
        ["Metric", "Value"],
        ["Total Rows", f"{profile['num_rows']:,}"],
        ["Total Columns", str(profile["num_cols"])],
        ["Quality Score", f"{profile['quality_score']} / 100"],
        ["Duplicate Rows", str(profile["duplicate_rows"])],
        ["Memory Usage", f"{profile['memory_usage_kb']} KB"],
    ]
    metrics_table = Table(metrics_data, colWidths=[8 * cm, 8 * cm])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 1 * cm))

    # ── Executive Summary ─────────────────────────────────────
    if narrative:
        story.append(Paragraph("Executive Summary", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_BG))
        story.append(Spacer(1, 0.3 * cm))
        for para in narrative.split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles["body"]))
        story.append(PageBreak())

    # ── Column Overview ───────────────────────────────────────
    story.append(Paragraph("Column Overview", styles["h1"]))
    col_header = ["Column", "Type", "Unique", "Missing", "Top Value / Mean"]
    col_rows = [col_header]
    for col, cp in profile["column_profiles"].items():
        top = cp.get("mode", cp.get("mean", "—"))
        col_rows.append([
            col,
            cp["type"].capitalize(),
            str(cp["unique_values"]),
            f"{cp['missing_count']} ({cp['missing_pct']}%)",
            str(top)[:30],
        ])
    col_table = Table(col_rows, colWidths=[4 * cm, 3 * cm, 2.5 * cm, 3.5 * cm, 4 * cm])
    col_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(col_table)
    story.append(PageBreak())

    # ── Univariate Analysis ───────────────────────────────────
    if univariate_results:
        story.append(Paragraph("Univariate Analysis", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=1, color=LIGHT_BG))
        story.append(Spacer(1, 0.3 * cm))

        for col, res in univariate_results.items():
            story.append(Paragraph(f"Column: {col}", styles["h2"]))

            if res["type"] == "categorical":
                # Top values table
                tv_data = [["Value", "Count", "%"]]
                for val, cnt in list(res["top_values"].items())[:8]:
                    pct = res["top_pct"].get(val, 0)
                    tv_data.append([str(val)[:30], str(cnt), f"{pct:.1f}%"])
                tv_table = Table(tv_data, colWidths=[8 * cm, 4 * cm, 4 * cm])
                _apply_basic_table_style(tv_table)
                story.append(tv_table)

                # Chart
                if "fig_bar" in res:
                    img = fig_to_image(res["fig_bar"])
                    if img:
                        story.append(img)

            elif res["type"] == "numerical":
                s = res["stats"]
                stat_data = [
                    ["Mean", "Median", "Std Dev", "Min", "Max", "Skewness"],
                    [str(s["mean"]), str(s["median"]), str(s["std"]),
                    str(s["min"]), str(s["max"]), str(s["skewness"])],
                ]
                stat_table = Table(stat_data, colWidths=[2.7 * cm] * 6)
                _apply_basic_table_style(stat_table)
                story.append(stat_table)

                if "fig_hist" in res:
                    img = fig_to_image(res["fig_hist"])
                    if img:
                        story.append(img)

            elif res["type"] == "datetime":
                story.append(Paragraph(f"Date Range: {res['min_date']} → {res['max_date']}", styles["body"]))

            story.append(Spacer(1, 0.5 * cm))

        story.append(PageBreak())

    # ── Bivariate Analysis ────────────────────────────────────
    if bivariate_results:
        story.append(Paragraph("Bivariate Analysis", styles["h1"]))

        # Correlation heatmap
        if bivariate_results.get("correlation_fig"):
            story.append(Paragraph("Numerical Correlation Heatmap", styles["h2"]))
            img = fig_to_image(bivariate_results["correlation_fig"], height=9 * cm)
            if img:
                story.append(img)
            story.append(Spacer(1, 0.5 * cm))

        # Top categorical relationships
        cat_cat = bivariate_results.get("cat_cat", [])
        if cat_cat:
            story.append(Paragraph("Categorical Relationships (Top 5)", styles["h2"]))
            cr_data = [["Column 1", "Column 2", "Cramer's V", "p-value", "Significant"]]
            for r in cat_cat[:5]:
                cr_data.append([
                    r["col1"], r["col2"],
                    str(r["cramers_v"]), str(r["p_value"]),
                    "✓ Yes" if r["significant"] else "✗ No"
                ])
            cr_table = Table(cr_data, colWidths=[3.5 * cm, 3.5 * cm, 3 * cm, 3 * cm, 3 * cm])
            _apply_basic_table_style(cr_table)
            story.append(cr_table)

            for r in cat_cat[:3]:
                if "fig" in r:
                    img = fig_to_image(r["fig"])
                    if img:
                        story.append(img)
            story.append(Spacer(1, 0.5 * cm))

        story.append(PageBreak())

    # ── Clustering ────────────────────────────────────────────
    if cluster_results and "error" not in cluster_results:
        story.append(Paragraph("Customer / Data Segmentation", styles["h1"]))
        story.append(Paragraph(
            f"The algorithm identified <b>{cluster_results['n_clusters']} segments</b> in your data.",
            styles["body"]
        ))

        if "fig_pca" in cluster_results:
            img = fig_to_image(cluster_results["fig_pca"], height=9 * cm)
            if img:
                story.append(img)

        if "fig_elbow" in cluster_results:
            img = fig_to_image(cluster_results["fig_elbow"], height=7 * cm)
            if img:
                story.append(img)

        story.append(PageBreak())

    # ── Association Rules ──────────────────────────────────────
    if assoc_results and "error" not in assoc_results:
        story.append(Paragraph("Association Rule Mining", styles["h1"]))
        rules_df = assoc_results.get("rules_df")
        if rules_df is not None:
            ar_data = [["IF (Antecedent)", "THEN (Consequent)", "Support", "Confidence", "Lift"]]
            for _, row in rules_df.head(10).iterrows():
                ar_data.append([
                    str(row["antecedents"])[:30],
                    str(row["consequents"])[:30],
                    str(row["support"]),
                    str(row["confidence"]),
                    str(row["lift"]),
                ])
            ar_table = Table(ar_data, colWidths=[5 * cm, 5 * cm, 2 * cm, 2.5 * cm, 2.5 * cm])
            _apply_basic_table_style(ar_table)
            story.append(ar_table)

            if "fig_bar" in assoc_results:
                img = fig_to_image(assoc_results["fig_bar"])
                if img:
                    story.append(img)

        story.append(PageBreak())

    # ── Time Trends ────────────────────────────────────────────
    if time_results and time_results.get("available"):
        story.append(Paragraph("Time Trend Analysis", styles["h1"]))
        if "fig_count" in time_results:
            img = fig_to_image(time_results["fig_count"])
            if img:
                story.append(img)
        for trend in time_results.get("trends", [])[:4]:
            if "fig" in trend:
                img = fig_to_image(trend["fig"])
                if img:
                    story.append(img)
        story.append(PageBreak())

    # ── Footer ─────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Paragraph(
        f"Generated by AutoAnalyst AI · {datetime.now().strftime('%B %d, %Y')}",
        styles["small"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _apply_basic_table_style(table):
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
