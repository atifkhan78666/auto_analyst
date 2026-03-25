import streamlit as st
import pandas as pd
from utils.helpers import load_file
from engine.profiler import profile_dataframe
from engine.univariate import run_univariate_analysis
from engine.bivariate import run_bivariate_analysis
from engine.clustering import run_clustering
from engine.associations import run_association_analysis
from engine.time_analysis import run_time_analysis
from chat.chat_engine import chat_with_data
from ai.narrator import generate_report_narrative, generate_quick_profile_narrative
from report.builder import build_report

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AutoAnalyst AI",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFD; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        text-align: center;
    }
    .chat-bubble-user {
        background: #4F8EF7;
        color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-ai {
        background: white;
        color: #222;
        border-radius: 16px 16px 16px 4px;
        padding: 10px 16px;
        margin: 6px 0;
        max-width: 80%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────
for key in ["df", "profile", "filename", "univariate", "bivariate",
            "clustering", "associations", "time_analysis",
            "chat_history", "narrative"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.chat_history is None:
    st.session_state.chat_history = []

# ── Header ────────────────────────────────────────────────────
st.title("🤖 AutoAnalyst AI")
st.markdown("*Upload your data · Get instant AI-powered analysis · Chat with your dataset*")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Upload & Profile",
    "📊 Full Analysis",
    "💬 Chat with Data",
    "📄 Export Report",
])

# ══════════════════════════════════════════════════════════════
#  TAB 1 — Upload & Profile
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Drag and drop your CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file:
        df, error = load_file(uploaded_file)
        if error:
            st.error(f"❌ {error}")
        else:
            if st.session_state.filename != uploaded_file.name:
                # Fresh upload — reset all state
                st.session_state.df = df
                st.session_state.filename = uploaded_file.name
                st.session_state.profile = profile_dataframe(df)
                st.session_state.univariate = None
                st.session_state.bivariate = None
                st.session_state.clustering = None
                st.session_state.associations = None
                st.session_state.time_analysis = None
                st.session_state.chat_history = []
                st.session_state.narrative = None

            profile = st.session_state.profile
            st.success(f"✅ Loaded: **{uploaded_file.name}**")
            st.divider()

            # Metrics row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("📋 Rows", f"{profile['num_rows']:,}")
            c2.metric("📌 Columns", profile["num_cols"])
            c3.metric("🏆 Quality Score", f"{profile['quality_score']} / 100")
            c4.metric("⚠️ Duplicates", profile["duplicate_rows"])
            c5.metric("💾 Memory", f"{profile['memory_usage_kb']} KB")

            st.divider()

            # Column overview table
            st.subheader("📂 Column Overview")
            col_rows = []
            for col, cp in profile["column_profiles"].items():
                col_rows.append({
                    "Column": col,
                    "Type": cp["type"].capitalize(),
                    "Unique Values": cp["unique_values"],
                    "Missing": f"{cp['missing_count']} ({cp['missing_pct']}%)",
                    "Top Value / Mode": cp.get("mode", cp.get("mean", "—")),
                })
            st.dataframe(pd.DataFrame(col_rows), use_container_width=True)

            # Missing values
            if profile["missing_summary"]:
                st.divider()
                st.subheader("⚠️ Missing Values")
                miss = [
                    {"Column": c, "Missing Count": v["count"], "Missing %": f"{v['percent']}%"}
                    for c, v in profile["missing_summary"].items()
                ]
                st.dataframe(pd.DataFrame(miss), use_container_width=True)
            else:
                st.success("✅ No missing values found!")

            # AI summary
            st.divider()
            st.subheader("🤖 AI Profile Summary")
            if st.button("Generate AI Summary", key="ai_profile_btn"):
                with st.spinner("Generating AI summary..."):
                    try:
                        summary = generate_quick_profile_narrative(df, profile)
                        st.session_state.narrative = summary
                    except Exception as e:
                        st.error(f"AI error: {e}")

            if st.session_state.narrative:
                st.info(st.session_state.narrative)

            # Data preview
            st.divider()
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

    else:
        st.info("👆 Upload a CSV or Excel file to get started.")
        st.markdown("""
**What happens after upload?**
- ✅ Instant data quality score
- ✅ Column-by-column breakdown
- ✅ Missing value detection
- ✅ Full statistical + AI analysis
- ✅ Chat with your data using AI
- ✅ Downloadable PDF report
        """)

# ══════════════════════════════════════════════════════════════
#  TAB 2 — Full Analysis
# ══════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.df is None:
        st.info("📁 Please upload a dataset in the **Upload & Profile** tab first.")
    else:
        df = st.session_state.df

        st.subheader("📊 Full Analysis")

        if st.button("▶ Run Full Analysis", type="primary"):
            with st.spinner("Running univariate analysis..."):
                st.session_state.univariate = run_univariate_analysis(df)
            with st.spinner("Running bivariate analysis..."):
                st.session_state.bivariate = run_bivariate_analysis(df)
            with st.spinner("Running clustering..."):
                st.session_state.clustering = run_clustering(df)
            with st.spinner("Running association rules..."):
                st.session_state.associations = run_association_analysis(df)
            with st.spinner("Running time analysis..."):
                st.session_state.time_analysis = run_time_analysis(df)
            st.success("✅ Full analysis complete!")

        st.divider()

        # ── Univariate ─────────────────────────────────────────
        if st.session_state.univariate:
            st.subheader("📈 Univariate Analysis")
            uni = st.session_state.univariate

            for col, res in uni.items():
                with st.expander(f"🔹 {col} ({res['type'].capitalize()})", expanded=False):
                    if res["type"] == "categorical":
                        st.markdown(f"**Mode:** {res['mode']} &nbsp;|&nbsp; **Unique Values:** {res['unique_count']}")
                        c1, c2 = st.columns(2)
                        with c1:
                            if "fig_bar" in res:
                                st.plotly_chart(res["fig_bar"], use_container_width=True)
                        with c2:
                            if "fig_pie" in res:
                                st.plotly_chart(res["fig_pie"], use_container_width=True)

                    elif res["type"] == "numerical":
                        s = res["stats"]
                        mc = st.columns(5)
                        mc[0].metric("Mean", s["mean"])
                        mc[1].metric("Median", s["median"])
                        mc[2].metric("Std Dev", s["std"])
                        mc[3].metric("Min", s["min"])
                        mc[4].metric("Max", s["max"])
                        c1, c2 = st.columns(2)
                        with c1:
                            if "fig_hist" in res:
                                st.plotly_chart(res["fig_hist"], use_container_width=True)
                        with c2:
                            if "fig_box" in res:
                                st.plotly_chart(res["fig_box"], use_container_width=True)

                    elif res["type"] == "datetime":
                        st.markdown(f"**Range:** {res['min_date']} → {res['max_date']}")
                        if "fig_timeline" in res:
                            st.plotly_chart(res["fig_timeline"], use_container_width=True)

            st.divider()

        # ── Bivariate ──────────────────────────────────────────
        if st.session_state.bivariate:
            biv = st.session_state.bivariate
            st.subheader("🔗 Bivariate Analysis")

            if biv.get("correlation_fig"):
                st.markdown("**Numerical Correlation Heatmap**")
                st.plotly_chart(biv["correlation_fig"], use_container_width=True)

            if biv.get("cat_cat"):
                st.markdown("**Categorical Relationships (Cramer's V)**")
                cr_data = [
                    {
                        "Column 1": r["col1"],
                        "Column 2": r["col2"],
                        "Cramer's V": r["cramers_v"],
                        "p-value": r["p_value"],
                        "Significant": "✓ Yes" if r["significant"] else "✗ No",
                    }
                    for r in biv["cat_cat"][:10]
                ]
                st.dataframe(pd.DataFrame(cr_data), use_container_width=True)

                for r in biv["cat_cat"][:5]:
                    if "fig" in r:
                        with st.expander(f"📊 {r['col1']} vs {r['col2']} (V={r['cramers_v']})", expanded=False):
                            st.plotly_chart(r["fig"], use_container_width=True)

            if biv.get("cat_num"):
                st.markdown("**Categorical vs Numerical**")
                for r in biv["cat_num"][:6]:
                    if "fig" in r:
                        with st.expander(f"📊 {r['num_col']} by {r['cat_col']}", expanded=False):
                            st.plotly_chart(r["fig"], use_container_width=True)

            st.divider()

        # ── Clustering ─────────────────────────────────────────
        if st.session_state.clustering:
            cl = st.session_state.clustering
            st.subheader("🎯 Data Segmentation / Clustering")

            if "error" in cl:
                st.warning(cl["error"])
            else:
                st.markdown(f"**Optimal Segments Found: {cl['n_clusters']}**")
                cnt_df = pd.DataFrame(
                    list(cl["cluster_counts"].items()),
                    columns=["Cluster", "Count"]
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(cl["fig_pca"], use_container_width=True)
                with c2:
                    st.plotly_chart(cl["fig_elbow"], use_container_width=True)

                st.markdown("**Cluster Profiles**")
                for cluster_name, info in cl["cluster_summary"].items():
                    with st.expander(f"🔹 {cluster_name} — {info['size']} rows", expanded=False):
                        if info["numerical"]:
                            st.markdown("*Numerical Averages:*")
                            st.json(info["numerical"])
                        if info["categorical"]:
                            st.markdown("*Top Categories:*")
                            st.json(info["categorical"])

            st.divider()

        # ── Association Rules ──────────────────────────────────
        if st.session_state.associations:
            assoc = st.session_state.associations
            st.subheader("🔀 Association Rules (Market Basket Analysis)")

            if "error" in assoc:
                st.warning(assoc["error"])
            else:
                st.markdown(f"**{assoc['num_rules']} rules found**")
                st.dataframe(assoc["rules_df"], use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    if "fig_bar" in assoc:
                        st.plotly_chart(assoc["fig_bar"], use_container_width=True)
                with c2:
                    if "fig_scatter" in assoc:
                        st.plotly_chart(assoc["fig_scatter"], use_container_width=True)

            st.divider()

        # ── Time Analysis ──────────────────────────────────────
        if st.session_state.time_analysis:
            ta = st.session_state.time_analysis
            st.subheader("📅 Time Trend Analysis")

            if not ta.get("available"):
                st.info(ta.get("message", "No datetime columns found."))
            else:
                st.markdown(f"**Grouping: {ta.get('freq_label', 'Auto')}**")
                if "fig_count" in ta:
                    st.plotly_chart(ta["fig_count"], use_container_width=True)
                for trend in ta.get("trends", []):
                    if "fig" in trend:
                        st.plotly_chart(trend["fig"], use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 3 — Chat with Data
# ══════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.df is None:
        st.info("📁 Please upload a dataset in the **Upload & Profile** tab first.")
    else:
        df = st.session_state.df
        st.subheader("💬 Chat with Your Data")
        st.markdown("Ask anything about your dataset — the AI has full context of your data.")

        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
> 👋 **Hi! I'm your data analyst assistant.**
> I have full knowledge of your dataset.
> Ask me anything — for example:
> - *"Which category appears most often?"*
> - *"What is the average value of column X?"*
> - *"Are there any patterns between column A and B?"*
> - *"Summarize the key insights from this data"*
                """)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True
                        )

        st.divider()

        # Quick prompt suggestions
        st.markdown("**Quick prompts:**")
        qcols = st.columns(4)
        quick_prompts = [
            "Summarize this dataset",
            "What are the top categories?",
            "Are there any data quality issues?",
            "What insights can you find?",
        ]
        for i, prompt in enumerate(quick_prompts):
            if qcols[i].button(prompt, key=f"qp_{i}"):
                with st.spinner("Thinking..."):
                    try:
                        reply, updated_history = chat_with_data(
                            df,
                            st.session_state.chat_history,
                            prompt,
                            st.session_state.profile,
                        )
                        st.session_state.chat_history = updated_history
                        st.rerun()
                    except Exception as e:
                        st.error(f"Chat error: {e}")

        # User input
        user_input = st.chat_input("Ask a question about your data...")
        if user_input:
            with st.spinner("Thinking..."):
                try:
                    reply, updated_history = chat_with_data(
                        df,
                        st.session_state.chat_history,
                        user_input,
                        st.session_state.profile,
                    )
                    st.session_state.chat_history = updated_history
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {e}")

        # Clear chat
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

# ══════════════════════════════════════════════════════════════
#  TAB 4 — Export Report
# ══════════════════════════════════════════════════════════════
with tab4:
    if st.session_state.df is None:
        st.info("📁 Please upload a dataset in the **Upload & Profile** tab first.")
    else:
        st.subheader("📄 Export Full PDF Report")
        st.markdown("Generate a professional, AI-narrated PDF report of your complete analysis.")

        analysis_done = st.session_state.univariate is not None
        if not analysis_done:
            st.warning("⚠️ Run the **Full Analysis** in Tab 2 first for a richer report. Or generate a profile-only report below.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📄 Generate Full Report (with Analysis)", type="primary", disabled=not analysis_done):
                with st.spinner("Generating AI narrative..."):
                    try:
                        narrative = generate_report_narrative(
                            st.session_state.df,
                            st.session_state.profile,
                            st.session_state.bivariate,
                            st.session_state.clustering,
                            st.session_state.associations,
                        )
                    except Exception as e:
                        narrative = f"AI narrative unavailable: {e}"

                with st.spinner("Building PDF report..."):
                    try:
                        pdf_bytes = build_report(
                            filename=st.session_state.filename,
                            profile=st.session_state.profile,
                            univariate_results=st.session_state.univariate,
                            bivariate_results=st.session_state.bivariate,
                            cluster_results=st.session_state.clustering,
                            assoc_results=st.session_state.associations,
                            time_results=st.session_state.time_analysis,
                            narrative=narrative,
                        )
                        st.success("✅ Report generated!")
                        st.download_button(
                            label="⬇️ Download Full Report PDF",
                            data=pdf_bytes,
                            file_name=f"autoanalyst_report_{st.session_state.filename}.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"Report generation error: {e}")

        with col2:
            if st.button("📋 Generate Profile-Only Report"):
                with st.spinner("Generating quick report..."):
                    try:
                        narrative = generate_quick_profile_narrative(
                            st.session_state.df,
                            st.session_state.profile,
                        )
                    except Exception as e:
                        narrative = None

                    try:
                        pdf_bytes = build_report(
                            filename=st.session_state.filename,
                            profile=st.session_state.profile,
                            narrative=narrative,
                        )
                        st.success("✅ Profile report generated!")
                        st.download_button(
                            label="⬇️ Download Profile Report PDF",
                            data=pdf_bytes,
                            file_name=f"profile_report_{st.session_state.filename}.pdf",
                            mime="application/pdf",
                        )
                    except Exception as e:
                        st.error(f"Report error: {e}")

        st.divider()
        st.markdown("**Report Includes:**")
        st.markdown("""
- 📋 Executive Summary (AI-written)
- 📊 Dataset overview & quality score
- 🔍 Column-by-column analysis with charts
- 🔗 Relationship analysis (Cramer's V, correlations)
- 🎯 Clustering / segmentation results
- 🔀 Association rules
- 📅 Time trends (if date column present)
        """)