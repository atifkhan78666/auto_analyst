# 🤖 AutoAnalyst AI

An AI-powered automated data analysis platform built with Python and Streamlit.
Upload any CSV or Excel file and get instant analysis, visualizations, and insights — no coding required.

## Features

- 📊 **Full Auto Analysis** — Univariate, Bivariate, Clustering, Association Rules, Time Trends
- 💬 **Chat with Your Data** — Ask questions about your dataset in plain English (powered by Groq + LLaMA 3.3)
- 🤖 **AI Narratives** — Auto-generated executive summaries using Groq AI
- 📄 **PDF Report Export** — Download a professional report with all charts and insights
- 🆓 **100% Free Stack** — No paid APIs or subscriptions required

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Analysis | Pandas, SciPy, Scikit-learn, MLxtend |
| Visualization | Plotly, Matplotlib, Seaborn |
| AI / Chat | Groq API (LLaMA 3.3 70B) — Free tier |
| PDF Export | ReportLab + Kaleido |

## Project Structure
```
auto_analyst/
├── app.py                  # Main Streamlit app (4 tabs)
├── engine/
│   ├── profiler.py         # Data profiling
│   ├── univariate.py       # Per-column analysis + charts
│   ├── bivariate.py        # Column relationships (Cramer's V, correlation)
│   ├── clustering.py       # KMeans segmentation
│   ├── associations.py     # Apriori association rules
│   └── time_analysis.py    # Time trend detection
├── chat/
│   ├── context_builder.py  # DataFrame → LLM context
│   ├── chat_engine.py      # Groq API chat loop
│   └── function_calls.py   # Data query functions
├── ai/
│   └── narrator.py         # AI report narrative generator
├── report/
│   └── builder.py          # PDF report builder
└── utils/
    └── helpers.py          # Shared utilities
```

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/auto-analyst.git
cd auto-analyst
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
Create a `.env` file in the root folder:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

## Usage

1. Go to **Upload & Profile** tab → upload a CSV or Excel file
2. Go to **Full Analysis** tab → click **Run Full Analysis**
3. Go to **Chat with Data** tab → ask questions about your data
4. Go to **Export Report** tab → download your PDF report

## License

MIT License