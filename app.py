import json
import os
import re
import socket
import ssl
import uuid
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st
import certifi
try:
    import truststore
except ImportError:
    truststore = None

st.set_page_config(page_title="PaisaTrack", page_icon="₹", layout="wide")

CATEGORIES = ["Food", "Transport", "Shopping", "Education", "Bills", "Health", "Entertainment", "Other"]
TRANSACTION_COLUMNS = ["Date", "Description", "Category", "Payment", "Amount"]
GEMMA_MODEL = "gemma-4-26b-a4b-it"
SEED = [
    ("Zomato", "Food", 250, "UPI"), ("DMRC", "Transport", 100, "UPI"),
    ("Amazon", "Shopping", 850, "Card"), ("College Canteen", "Food", 120, "Cash"),
    ("Airtel recharge", "Bills", 299, "UPI"), ("Notebook & pens", "Education", 180, "Cash"),
    ("Rapido", "Transport", 95, "UPI"), ("Movie ticket", "Entertainment", 320, "Card"),
]

def category_for(description):
    text = str(description).lower()
    rules = {
        "Food": ["food", "zomato", "swiggy", "canteen", "cafe", "restaurant", "blinkit", "zepto"],
        "Transport": ["metro", "dmrc", "uber", "ola", "rapido", "bus", "fuel", "petrol"],
        "Shopping": ["amazon", "flipkart", "myntra", "shopping", "store"],
        "Education": ["book", "course", "college", "tuition", "stationery", "notebook"],
        "Bills": ["recharge", "airtel", "jio", "electricity", "internet", "bill"],
        "Health": ["medical", "pharmacy", "doctor", "hospital", "health"],
        "Entertainment": ["movie", "netflix", "spotify", "game", "concert"],
    }
    return next((cat for cat, words in rules.items() if any(word in text for word in words)), "Other")

def money(value):
    return f"₹{value:,.0f}"

def google_api_key():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        return st.secrets.get("GEMINI_API_KEY", api_key)
    except (FileNotFoundError, AttributeError):
        return api_key

def categorize_descriptions_with_gemma(descriptions):
    """Suggest categories using only merchant descriptions; never send dates/amounts."""
    api_key = google_api_key()
    if not api_key:
        raise RuntimeError("missing_google_api_key")

    unique_descriptions = list(dict.fromkeys(str(item).strip() for item in descriptions if str(item).strip()))
    indexed_merchants = [
        {
            "id": index,
            # Remove long numeric references before sending merchant text to Google.
            "merchant": re.sub(r"\b\d{4,}\b", "[number]", description),
        }
        for index, description in enumerate(unique_descriptions)
    ]
    prompt = (
        "Categorize each merchant description for a student's expense tracker. "
        f"For each item choose exactly one category from: {', '.join(CATEGORIES)}. "
        "If the merchant is unclear, do not guess: set uncertain=true and confidence below 0.72. "
        "Return only valid JSON with this shape: "
        '{"categories":[{"id":0,"category":"Food","confidence":0.9,"uncertain":false}]}. '
        "The input contains merchant descriptions only; do not infer dates or amounts. Merchants: "
        + json.dumps(indexed_merchants, ensure_ascii=False)
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": max(300, len(indexed_merchants) * 35)},
    }).encode("utf-8")
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMMA_MODEL}:generateContent",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    if truststore is not None:
        tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=tls_context) as response:
        response_json = json.loads(response.read().decode("utf-8"))
    parts = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    response_text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
    start, end = response_text.find("{"), response_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemma did not return category JSON.")
    result = json.loads(response_text[start:end + 1])
    by_id = {
        item.get("id"): item
        for item in result.get("categories", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }

    suggestions = {}
    for index, description in enumerate(unique_descriptions):
        item = by_id.get(index, {})
        category = item.get("category")
        try:
            confidence = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        uncertain_value = item.get("uncertain", True)
        uncertain = uncertain_value is True or str(uncertain_value).strip().lower() in {"true", "yes", "1"}
        is_valid = category in CATEGORIES
        suggestions[description] = {
            "category": category if is_valid else category_for(description),
            "needs_review": (not is_valid) or uncertain or confidence < 0.72,
        }
    return suggestions

def clear_selected_upload():
    # File uploader values are read-only in session state. A new widget key
    # reliably clears the selected file and makes the uploader ready again.
    st.session_state.csv_upload_generation += 1

def delete_selected_transactions():
    selected_ids = set(st.session_state.get("delete_transaction_ids", []))
    if not selected_ids:
        return
    transactions = st.session_state.transactions
    st.session_state.transactions = transactions[
        ~transactions["_tx_id"].astype(str).isin(selected_ids)
    ].reset_index(drop=True)
    st.session_state.delete_notice = f"Removed {len(selected_ids)} transaction(s)."
    st.session_state.delete_transaction_ids = []

def remove_last_csv_import():
    batch_id = st.session_state.get("last_uploaded_batch")
    if not batch_id:
        return
    transactions = st.session_state.transactions
    batch_mask = transactions["_batch_id"].eq(batch_id)
    removed = int(batch_mask.sum())
    st.session_state.transactions = transactions.loc[~batch_mask].reset_index(drop=True)
    st.session_state.last_uploaded_batch = None
    st.session_state.last_uploaded_filename = None
    st.session_state.import_notice = f"Removed the last CSV import ({removed} transaction(s))."

def add_csv_transactions(imported, filename):
    imported = imported.copy()
    imported["_tx_id"] = [uuid.uuid4().hex for _ in range(len(imported))]
    batch_id = uuid.uuid4().hex
    imported["_batch_id"] = batch_id
    st.session_state.transactions = pd.concat([st.session_state.transactions, imported], ignore_index=True)
    st.session_state.last_uploaded_batch = batch_id
    st.session_state.last_uploaded_filename = filename
    st.session_state.import_notice = f"Added {len(imported)} transactions from {filename}. They are now in All transactions."
    # A fresh uploader key clears the file while keeping this import undoable.
    st.session_state.csv_upload_generation += 1

def fallback_spending_advice(summary):
    """Provide a concise, arithmetic-grounded answer if Gemma echoes instructions."""
    top_category = summary["top_category"]
    top_amount = summary["top_category_spent_inr"]
    share = summary["top_category_share_percent"]
    weekly_cap = summary["suggested_weekly_cap_inr"]
    return (
        f"- Your largest category so far is {top_category} at {money(top_amount)} "
        f"({share}% of this month's spending).\n"
        f"- Try keeping {top_category} near {money(weekly_cap)} per week and compare it again next week."
    )

def ask_gemma(category_totals, monthly_budget, spent_total, model_name, last_7_days=None):
    """Ask Google's hosted Gemma API for concise advice grounded in spending totals."""
    api_key = google_api_key()
    if not api_key:
        raise RuntimeError("missing_google_api_key")
    category_totals = {str(category): round(float(amount), 2) for category, amount in category_totals.items()}
    last_7_days = {
        str(category): round(float(amount), 2)
        for category, amount in (last_7_days or {}).items()
    }
    top_category = max(category_totals, key=category_totals.get)
    top_amount = category_totals[top_category]
    top_share = round(top_amount / spent_total * 100) if spent_total else 0
    weekly_cap = round(top_amount / max(date.today().day / 7, 1))
    summary = {
        "monthly_budget_inr": round(monthly_budget, 2),
        "spent_so_far_inr": round(spent_total, 2),
        "remaining_inr": round(monthly_budget - spent_total, 2),
        "spending_by_category_this_month_inr": category_totals,
        "spending_by_category_last_7_days_inr": last_7_days,
        "top_category": top_category,
        "top_category_spent_inr": top_amount,
        "top_category_share_percent": top_share,
        "suggested_weekly_cap_inr": weekly_cap,
    }
    prompt = (
        "You are PaisaTrack, a practical and supportive student expense coach. Analyze the spending data below. "
        "Write exactly two short Markdown bullets and nothing else. Bullet 1 must identify the clearest spending pattern "
        "using the supplied category and amount. Bullet 2 must give one specific, realistic action tied to that category "
        "and the supplied weekly cap. Use Indian rupees. Use only the supplied facts and numbers; do not invent causes, "
        "income, goals, or transactions. Do not repeat these instructions, describe your role, list constraints, or restate all data. "
        "Keep the advice non-judgmental and easy for a student to follow. Data: "
        + json.dumps(summary, ensure_ascii=False)
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 120},
    }).encode("utf-8")
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    if truststore is not None:
        tls_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        tls_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=120, context=tls_context) as response:
        result = json.loads(response.read().decode("utf-8"))
    parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    response_text = "\n".join(part.get("text", "").strip() for part in parts if part.get("text", "")).strip()
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    prompt_echo_markers = ("persona:", "constraint", "input data:", "output format:", "return exactly")
    if (
        len(lines) != 2
        or any(marker in response_text.lower() for marker in prompt_echo_markers)
        or len(response_text.split()) > 90
    ):
        return fallback_spending_advice(summary)
    cleaned_lines = [line.lstrip("-•* ").strip() for line in lines]
    if not all(cleaned_lines):
        return fallback_spending_advice(summary)
    return "\n".join(f"- {line}" for line in cleaned_lines)

if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame([
        {"Date": date.today().replace(day=max(1, date.today().day-i)), "Description": name,
         "Category": cat, "Amount": amount, "Payment": payment,
         "_tx_id": uuid.uuid4().hex, "_batch_id": None}
        for i, (name, cat, amount, payment) in enumerate(SEED)
    ])
else:
    # Add internal identifiers to sessions created by older versions of the app.
    if "_tx_id" not in st.session_state.transactions.columns:
        st.session_state.transactions["_tx_id"] = [
            uuid.uuid4().hex for _ in range(len(st.session_state.transactions))
        ]
    if "_batch_id" not in st.session_state.transactions.columns:
        st.session_state.transactions["_batch_id"] = None
st.session_state.setdefault("csv_upload_generation", 0)
st.session_state.setdefault("last_uploaded_batch", None)
st.session_state.setdefault("last_uploaded_filename", None)
st.session_state.setdefault("import_notice", None)
st.session_state.setdefault("delete_notice", None)
if st.session_state.get("gemma_insight_version") != 2:
    # Discard the older prompt dump currently cached in some open app sessions.
    st.session_state.pop("gemma_insight", None)
    st.session_state.pop("gemma_insight_context", None)
    st.session_state["gemma_insight_version"] = 2
if "budget" not in st.session_state:
    st.session_state.budget = 12000

with st.sidebar:
    st.markdown("# ₹ PaisaTrack")
    st.caption("STUDENT MONEY, MADE SIMPLE")
    page = st.radio("Navigate", ["Overview", "Transactions", "Upload CSV"], label_visibility="collapsed", key="page_navigation")
    st.divider()
    st.markdown("### Monthly budget")
    st.session_state.budget = st.number_input("Set your monthly limit", min_value=0, step=500, value=int(st.session_state.budget), help="Used to calculate your remaining balance.")

tx = st.session_state.transactions.copy()
if not tx.empty:
    tx["Date"] = pd.to_datetime(tx["Date"]).dt.date
month = date.today().month
year = date.today().year
current = tx[tx["Date"].apply(lambda d: d.month == month and d.year == year)] if not tx.empty else tx
spent = float(current["Amount"].sum()) if not current.empty else 0
remaining = float(st.session_state.budget) - spent

st.caption("YOUR MONEY, YOUR CONTROL")
st.title("Good money habits start here")
st.write("A clear view of your spending this month.")

metrics = st.columns(4)
for col, label, value, note in [
    (metrics[0], "Spent this month", money(spent), f"{len(current)} transactions recorded"),
    (metrics[1], "Budget remaining", money(remaining), "You're on track" if remaining >= 0 else "Over your monthly limit"),
    (metrics[2], "Average per day", money(spent / max(date.today().day, 1)), "Based on this month so far"),
    (metrics[3], "Top category", current.groupby("Category")["Amount"].sum().idxmax() if not current.empty else "—", "Where most money goes"),
]:
    col.metric(label, value, help=note, border=True)

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.markdown("### Spending by category")
    if current.empty:
        st.info("Add an expense or import a statement to see your breakdown.")
    else:
        breakdown = current.groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
        chart = px.bar(breakdown, x="Amount", y="Category", orientation="h", color_discrete_sequence=["#258253"], text_auto=".2s")
        dark_mode = getattr(st.context.theme, "type", "light") == "dark"
        chart.update_layout(height=300, margin=dict(l=0,r=10,t=10,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, xaxis_title=None, yaxis_title=None, template="plotly_dark" if dark_mode else "plotly_white")
        chart.update_traces(marker_line_width=0, texttemplate="₹%{x:,.0f}", textposition="outside", cliponaxis=False)
        chart.update_xaxes(showgrid=True, gridcolor="#35443b" if dark_mode else "#e8ece7", zeroline=False)
        chart.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
with right:
    st.markdown("### Your money snapshot")
    pct = min(max(spent / st.session_state.budget, 0), 1) if st.session_state.budget else 0
    st.progress(pct, text=f"{pct:.0%} of monthly budget used")
    if not current.empty:
        biggest = current.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        top, top_amount = biggest.index[0], biggest.iloc[0]
        share = top_amount / spent if spent else 0
        st.info(f"**A small insight**\n\n{top} is your biggest category at {money(top_amount)} ({share:.0%} of your spending). A weekly cap of about {money(top_amount / max(date.today().day / 7, 1))} may help you stay mindful.")
    else:
        st.info("Add a few expenses and PaisaTrack will highlight your biggest spending category.", title="Your insights will show up here")
    if not current.empty:
        category_totals = current.groupby("Category")["Amount"].sum().to_dict()
        recent_start = date.today() - timedelta(days=6)
        recent = current[current["Date"].between(recent_start, date.today())]
        last_7_day_totals = recent.groupby("Category")["Amount"].sum().to_dict()
        insight_context = json.dumps({
            "budget": round(float(st.session_state.budget), 2),
            "spent": round(spent, 2),
            "categories": sorted((str(k), round(float(v), 2)) for k, v in category_totals.items()),
            "last_7_days": sorted((str(k), round(float(v), 2)) for k, v in last_7_day_totals.items()),
        }, sort_keys=True)
        if st.button("Ask Gemma for an insight", type="primary"):
            st.session_state.pop("gemma_insight", None)
            st.session_state.pop("gemma_insight_context", None)
            with st.spinner("Asking Google Gemma…"):
                try:
                    ai_text = ask_gemma(
                        category_totals,
                        float(st.session_state.budget),
                        spent,
                        GEMMA_MODEL,
                        last_7_day_totals,
                    )
                    if ai_text:
                        st.session_state["gemma_insight"] = ai_text
                        st.session_state["gemma_insight_context"] = insight_context
                    else:
                        st.warning("Gemma returned an empty response. Try again.")
                except RuntimeError as exc:
                    if str(exc) == "missing_google_api_key":
                        st.error("Add GEMINI_API_KEY to .streamlit/secrets.toml, restart Streamlit, then try again.")
                    else:
                        st.error("Gemma request could not be prepared.")
                except HTTPError as exc:
                    if exc.code in (401, 403):
                        st.error("Google rejected the API key. Check that GEMINI_API_KEY is a valid Google AI Studio key and that the Gemini API is available to your project.")
                    elif exc.code == 404:
                        st.error(f"Google could not find model '{GEMMA_MODEL}'. Check the supported Gemma model ID.")
                    elif exc.code == 429:
                        st.error("Google AI Studio rate limit reached. Check your current quota and try again later.")
                    else:
                        st.error(f"Google's Gemini API returned HTTP {exc.code}. Try again in a moment.")
                except URLError as exc:
                    reason = exc.reason
                    if isinstance(reason, socket.gaierror):
                        st.error("This computer couldn't find Google's API host (DNS lookup failed). Check your internet, VPN, proxy, or DNS settings, then try again.")
                    elif isinstance(reason, TimeoutError):
                        st.error("The connection to Google's API timed out. Check your internet or firewall and try again.")
                    else:
                        st.error(f"Couldn't connect to Google's API. Network detail: {reason}")
                except (TimeoutError, OSError) as exc:
                    st.error(f"Couldn't reach Google's API. Check your internet connection and try again. Detail: {exc}")
                except (ValueError, KeyError, json.JSONDecodeError):
                    st.error("Google replied in an unexpected format. Check the selected model and try again.")
        if st.session_state.get("gemma_insight") and st.session_state.get("gemma_insight_context") == insight_context:
            with st.container(border=True):
                st.markdown("**Gemma’s spending insight**")
                st.markdown(st.session_state["gemma_insight"])
        st.caption("Gemma insights send your budget and category totals to Google. CSV category suggestions send unique, redacted merchant descriptions only after you consent; transaction dates and amounts are not sent. Check Google's current privacy terms.")

st.markdown("### Recent transactions")
if current.empty:
    st.caption("No expenses yet. Add your first one below.")
else:
    display = current.sort_values("Date", ascending=False).head(6).copy()
    display["Date"] = display["Date"].apply(lambda d: d.strftime("%d %b"))
    display["Amount"] = display["Amount"].apply(money)
    st.dataframe(display[TRANSACTION_COLUMNS], width="stretch", hide_index=True, alt="Recent expense transactions")

with st.expander("＋ Add an expense", expanded=(page == "Transactions")):
    with st.form("add_expense", clear_on_submit=True):
        a,b,c = st.columns([1.2,1.2,1])
        description = a.text_input("What did you spend on?", placeholder="e.g. Lunch at canteen")
        amount = b.number_input("Amount (₹)", min_value=1, step=10)
        tx_date = c.date_input("Date", value=date.today())
        d,e = st.columns(2)
        category = d.selectbox("Category", CATEGORIES, index=CATEGORIES.index(category_for(description)) if description and category_for(description) in CATEGORIES else 0)
        payment = e.selectbox("Payment method", ["UPI", "Cash", "Card", "Bank transfer", "Other"])
        submitted = st.form_submit_button("Save expense", type="primary", width="stretch")
        if submitted:
            if not description.strip():
                st.error("Add a short description first.")
            else:
                new_row = pd.DataFrame([{"Date": tx_date, "Description": description.strip(), "Category": category, "Amount": amount, "Payment": payment, "_tx_id": uuid.uuid4().hex, "_batch_id": None}])
                st.session_state.transactions = pd.concat([st.session_state.transactions, new_row], ignore_index=True)
                st.success("Expense added to your dashboard.")
                st.rerun()

if page == "Upload CSV":
    st.markdown("### Upload a bank statement (CSV)")
    st.write("Upload a CSV, then either let Gemma categorize it or add the expenses using the file's existing categories.")
    if st.session_state.import_notice:
        st.success(st.session_state.import_notice)
        st.session_state.import_notice = None
    uploaded = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        help="CSV processing happens in this app session.",
        key=f"csv_upload_{st.session_state.csv_upload_generation}",
    )
    if uploaded:
        st.button("Clear selected file and choose another", on_click=clear_selected_upload)
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            if raw.empty or len(raw.columns) == 0:
                st.warning("This CSV has no transaction rows. Choose a different file.")
                st.stop()
            st.caption("Map the transaction columns. Dates and amounts are required; category and payment columns are optional.")
            cols = list(raw.columns)
            x,y,z = st.columns(3)
            date_default = next((i for i, name in enumerate(cols) if str(name).strip().lower() in {"date", "transaction date", "posted date"}), 0)
            desc_default = next((i for i, name in enumerate(cols) if str(name).strip().lower() in {"description", "merchant", "details", "narration"}), min(1, len(cols)-1))
            amount_default = next((i for i, name in enumerate(cols) if str(name).strip().lower() in {"amount", "debit", "withdrawal", "transaction amount"}), min(2, len(cols)-1))
            widget_suffix = st.session_state.csv_upload_generation
            date_col = x.selectbox("Date column", cols, index=date_default, key=f"csv_date_col_{widget_suffix}")
            desc_col = y.selectbox("Description / merchant", cols, index=desc_default, key=f"csv_desc_col_{widget_suffix}")
            amount_col = z.selectbox("Amount column", cols, index=amount_default, key=f"csv_amount_col_{widget_suffix}")
            category_default = next((name for name in cols if str(name).strip().lower() in {"category", "type", "expense category"}), "No category column")
            payment_default = next((name for name in cols if str(name).strip().lower() in {"payment", "payment method", "mode", "channel"}), "No payment column")
            category_col, payment_col = st.columns(2)
            category_source = category_col.selectbox(
                "Category column (optional)", ["No category column"] + cols,
                index=(cols.index(category_default) + 1) if category_default != "No category column" else 0,
                key=f"csv_category_col_{widget_suffix}",
            )
            payment_source = payment_col.selectbox(
                "Payment column (optional)", ["No payment column"] + cols,
                index=(cols.index(payment_default) + 1) if payment_default != "No payment column" else 0,
                key=f"csv_payment_col_{widget_suffix}",
            )
            st.caption("Amounts are treated as expenses. Exclude credits or refunds in the preview if present.")
            imported = pd.DataFrame({"Date": pd.to_datetime(raw[date_col], errors="coerce").dt.date, "Description": raw[desc_col].astype(str), "Amount": pd.to_numeric(raw[amount_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")})
            imported["Description"] = imported["Description"].str.strip()
            imported = imported.dropna(subset=["Date", "Amount"])
            imported = imported[imported["Description"].ne("")].copy()
            imported["Category"] = imported["Description"].map(category_for)
            if category_source != "No category column":
                category_lookup = {category.casefold(): category for category in CATEGORIES}
                source_categories = raw.loc[imported.index, category_source].fillna("").astype(str).str.strip()
                imported["Category"] = [
                    category_lookup.get(value.casefold(), category_for(description))
                    for value, description in zip(source_categories, imported["Description"])
                ]
            imported["Payment"] = "Bank statement"
            if payment_source != "No payment column":
                imported["Payment"] = raw.loc[imported.index, payment_source].fillna("Other").astype(str).str.strip().replace("", "Other")
            as_uploaded = imported[TRANSACTION_COLUMNS].copy()
            suggestion_key = f"gemma_category_suggestions_{widget_suffix}"
            suggestion_notice_key = f"gemma_category_notice_{widget_suffix}"
            cloud_consent = st.checkbox(
                "I agree to send unique, redacted merchant descriptions to Google Gemma for category suggestions. Dates and amounts will not be sent.",
                key=f"gemma_category_consent_{widget_suffix}",
            )
            if st.button("Auto-categorize with Gemma", disabled=not cloud_consent or imported.empty, key=f"gemma_categorize_{widget_suffix}"):
                try:
                    suggestions = categorize_descriptions_with_gemma(imported["Description"].tolist())
                    uncertain_descriptions = [description for description, suggestion in suggestions.items() if suggestion["needs_review"]]
                    if uncertain_descriptions:
                        st.session_state[suggestion_key] = suggestions
                        st.session_state[suggestion_notice_key] = f"Gemma categorized the clear matches. {len(uncertain_descriptions)} merchant(s) need your review."
                    else:
                        gemma_import = imported.copy()
                        gemma_import["Category"] = gemma_import["Description"].map(lambda description: suggestions[description]["category"])
                        add_csv_transactions(gemma_import[TRANSACTION_COLUMNS], uploaded.name)
                        st.session_state.import_notice = f"Gemma categorized and added all {len(gemma_import)} transactions from {uploaded.name}."
                    st.rerun()
                except RuntimeError as exc:
                    if str(exc) == "missing_google_api_key":
                        st.error("Add GEMINI_API_KEY to .streamlit/secrets.toml and restart Streamlit before using Gemma categorization.")
                    else:
                        st.error("Gemma could not prepare the category request.")
                except HTTPError as exc:
                    if exc.code in (401, 403):
                        st.error("Google rejected the API key. Check GEMINI_API_KEY and Gemini API access.")
                    elif exc.code == 429:
                        st.error("Google AI Studio rate limit reached. Try again later or review your quota.")
                    else:
                        st.error(f"Google's Gemini API returned HTTP {exc.code}. Try again in a moment.")
                except (URLError, TimeoutError, OSError) as exc:
                    st.error(f"Couldn't connect to Google Gemma. Check your internet connection and try again. Detail: {exc}")
                except (ValueError, KeyError, json.JSONDecodeError):
                    st.error("Gemma returned an unreadable category response. You can use the local suggestions and review them manually.")

            suggestions = st.session_state.get(suggestion_key, {})
            if st.session_state.get(suggestion_notice_key):
                st.success(st.session_state[suggestion_notice_key])
            imported["Review category"] = imported["Category"].eq("Other")
            if suggestions:
                imported["Category"] = imported["Description"].map(
                    lambda description: suggestions.get(description, {}).get("category", category_for(description))
                )
                imported["Review category"] = imported["Description"].map(
                    lambda description: suggestions.get(description, {}).get("needs_review", True)
                )
            else:
                st.caption("Gemma can assign clear categories automatically. Or add the statement as uploaded, using its category column when available and local matching otherwise.")
            st.caption(f"Preview: {len(imported)} valid transaction(s) from {len(raw)} CSV row(s).")
            preview = st.data_editor(
                imported[TRANSACTION_COLUMNS + ["Review category"]],
                width="stretch",
                hide_index=True,
                alt="Review and edit suggested categories before importing transactions",
                disabled=["Date", "Description", "Payment", "Amount"],
                column_config={
                    "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
                    "Review category": st.column_config.CheckboxColumn(
                        "Review category",
                        help="Gemma or local matching was unsure. Choose the right category, then uncheck this before importing.",
                    ),
                },
                key=f"csv_preview_editor_{widget_suffix}",
            )
            needs_review = bool(preview["Review category"].any()) if not preview.empty else False
            if needs_review:
                st.info("Please check each flagged category, choose the right category if needed, then uncheck Review category before importing.")
            import_buttons = st.columns(2)
            if import_buttons[0].button("Add reviewed categories", type="primary", disabled=preview.empty or needs_review, key="add_csv_transactions"):
                add_csv_transactions(preview[TRANSACTION_COLUMNS], uploaded.name)
                st.session_state[suggestion_notice_key] = None
                st.session_state[suggestion_key] = {}
                st.rerun()
            if import_buttons[1].button("Add expenses as uploaded", type="secondary", disabled=imported.empty, help="Skips Gemma. Uses the CSV category column when available, otherwise local merchant matching.", key=f"add_csv_as_uploaded_{widget_suffix}"):
                add_csv_transactions(as_uploaded, uploaded.name)
                st.session_state[suggestion_notice_key] = None
                st.session_state[suggestion_key] = {}
                st.rerun()
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    if st.session_state.last_uploaded_batch:
        st.button(
            f"Remove last uploaded file ({st.session_state.last_uploaded_filename})",
            on_click=remove_last_csv_import,
            type="secondary",
        )

if page == "Transactions":
    st.markdown("### All transactions")
    if st.session_state.delete_notice:
        st.success(st.session_state.delete_notice)
        st.session_state.delete_notice = None
    if st.session_state.import_notice:
        st.success(st.session_state.import_notice)
        st.session_state.import_notice = None
    if not tx.empty:
        category_filter = st.multiselect("Filter categories", CATEGORIES)
        shown = tx[tx["Category"].isin(category_filter)] if category_filter else tx
        visible_columns = st.multiselect(
            "Columns to show",
            TRANSACTION_COLUMNS,
            default=TRANSACTION_COLUMNS,
            key="transaction_visible_columns",
            help="Hide columns from this table; the underlying transaction data is kept.",
        )
        if visible_columns:
            st.dataframe(
                shown.sort_values("Date", ascending=False)[visible_columns],
                width="stretch",
                hide_index=True,
                alt="All saved expense transactions",
            )
        else:
            st.info("Select at least one column to display.")
        id_to_label = {
            str(row["_tx_id"]): f"{row['Date']} · {row['Description']} · {money(row['Amount'])}"
            for _, row in shown.iterrows()
        }
        st.multiselect(
            "Select transaction rows to delete",
            options=list(id_to_label),
            format_func=lambda tx_id: id_to_label[tx_id],
            key="delete_transaction_ids",
        )
        st.button(
            "Delete selected rows",
            type="secondary",
            disabled=not st.session_state.get("delete_transaction_ids"),
            on_click=delete_selected_transactions,
        )
        if st.session_state.last_uploaded_batch:
            st.button(
                f"Remove last uploaded file ({st.session_state.last_uploaded_filename})",
                on_click=remove_last_csv_import,
                type="secondary",
            )
    else:
        st.info("No transactions are in the list yet. Add an expense or upload a CSV.")

st.caption("PaisaTrack · Built for student life · Demo data is illustrative")
