# PaisaTrack

A student focused expense tracker MVP for a short hackathon build. It combines manual expense entry and CSV bank statement imports in one dashboard, with category summaries, a monthly budget, and simple spending insights.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow

1. Set a monthly budget in the sidebar.
2. Add a few expenses using **Add an expense**. Categories are suggested from merchant names and can be changed.
3. Open **Import statement** and upload a CSV. Choose its date, merchant/description, and amount columns, review the parsed rows, then add them.
4. Use **Transactions** to filter and inspect the entries.

## Use Google Gemma directly in the cloud

PaisaTrack calls Google's Gemini API directly with the hosted Gemma 4 model. No model or Ollama app needs to be installed. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/apikey), then store it in `.streamlit/secrets.toml` (never in `app.py`):

```toml
GEMINI_API_KEY = "your-Google-AI-Studio-key"
```

The secrets file is git-ignored. Restart Streamlit after changing it. For a hosted deployment, add `GEMINI_API_KEY` through the hosting service's secret settings instead.

Click **Ask Gemma for an insight** in the dashboard. The request includes the monthly budget and totals by category; transaction descriptions are not sent. Google's current pricing page lists Gemma 4 API input and output as free, though rate limits apply. Free-tier content may be used to improve Google products, so use sample data for demos unless you accept that data policy.

The starter transactions are illustrative. Data is kept in Streamlit session state for the demo and is not persisted after the session ends. CSV uploads are supported in this MVP; PDF extraction, authentication, and durable multi-user storage are intentionally outside the short-build scope.
