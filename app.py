import os, re, requests, numpy as np, pandas as pd, streamlit as st
import plotly.express as px
from urllib.parse import urlencode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error

st.set_page_config(page_title="Instagram BI + Forecast (No Images)", layout="wide")
st.title("📊 Instagram BI + Forecast (No Images)")

# -----------------------------
# Read token/ID from secrets or sidebar
# -----------------------------
default_token = st.secrets.get("instagram", {}).get("IG_TOKEN") if "instagram" in st.secrets else None
default_igid  = st.secrets.get("instagram", {}).get("IG_USER_ID") if "instagram" in st.secrets else None

st.sidebar.header("Instagram Graph API")
token = st.sidebar.text_input("Long-Lived Access Token", value=default_token or "", type="password")
ig_user_id = st.sidebar.text_input("IG User ID", value=default_igid or "")

# -----------------------------
# API helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_media(ig_user_id: str, token: str, limit: int = 150) -> pd.DataFrame:
    """Fetch last N media + per-media insights (reach, impressions, saves, engagement)."""
    base = "https://graph.facebook.com/v19.0"
    params = {
        "fields": "id,caption,timestamp,media_type,like_count,comments_count,permalink",
        "limit": limit,
        "access_token": token
    }
    url = f"{base}/{ig_user_id}/media?{urlencode(params)}"
    rows = []
    seen = 0
    while url and seen < limit:
        resp = requests.get(url, timeout=30)
        data = resp.json()
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            ins = requests.get(
                f"{base}/{mid}/insights?metric=impressions,reach,saves,engagement&access_token={token}",
                timeout=30
            ).json()
            metrics = {d["name"]: d["values"][0]["value"] for d in ins.get("data", [])} if ins.get("data") else {}
            rows.append({**m, **metrics})
            seen += 1
            if seen >= limit:
                break
        url = data.get("paging", {}).get("next")
    return pd.DataFrame(rows)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning + feature engineering (no images)."""
    if df.empty:
        return df

    # time
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["dow"]  = df["timestamp"].dt.dayofweek  # 0=Mon

    # caption + hashtags
    df["caption"] = df["caption"].fillna("")
    df["hashtags"] = df["caption"].str.findall(r"#\w+").apply(lambda lst: [h.lower() for h in lst])
    df["hashtag_count"] = df["hashtags"].str.len()

    # very simple sentiment proxy (you can swap for VADER later)
    pos = ["love","great","best","wow","amazing","new","sale","win","happy","cool"]
    neg = ["bad","hate","worst","angry","late","issue","broken","sad","fail"]
    lowcap = df["caption"].str.lower()
    df["sent_score"] = lowcap.str.count("|".join(pos)) - lowcap.str.count("|".join(neg))

    # numeric metrics
    for c in ["like_count","comments_count","impressions","reach","saves","engagement"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # choose reach if available else impressions
    df["reach_used"] = np.where(df.get("reach").notna(), df.get("reach"), df.get("impressions"))
    # ER normalized by reach/impressions
    df["engagement_rate"] = (df.get("like_count", 0).fillna(0)
                             + df.get("comments_count", 0).fillna(0)
                             + df.get("saves", 0).fillna(0)) / df["reach_used"].replace(0, np.nan)

    # account momentum: rolling median ER of last 10 posts
    df = df.sort_values("timestamp")
    df["roll_er_med_10"] = df["engagement_rate"].rolling(10, min_periods=1).median()

    return df

# -----------------------------
# Guard + fetch
# -----------------------------
if not token or not ig_user_id:
    st.info("Enter your **Long-Lived Token** and **IG User ID** in the sidebar (or set them in `.streamlit/secrets.toml`).")
    st.stop()

with st.spinner("Fetching recent posts from Instagram Graph API..."):
    raw = fetch_media(ig_user_id, token, limit=150)

if raw.empty:
    st.error("No media returned. Check that the IG account is Business/Creator, linked to a Facebook Page, and token has permissions.")
    st.stop()

df = build_features(raw)

# -----------------------------
# BI: Overview
# -----------------------------
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts", int(len(df)))
c2.metric("Avg ER", f"{(df['engagement_rate'].mean() * 100):.2f}%")
c3.metric("Median ER (10-post roll)", f"{(df['roll_er_med_10'].tail(1).iloc[0] * 100):.2f}%")
c4.metric("Avg Hashtag Count", f"{df['hashtag_count'].mean():.1f}")

# Trends
st.markdown("**Trends**")
daily = df.groupby(df["timestamp"].dt.date).agg(
    ER=("engagement_rate", "mean"),
    Likes=("like_count", "sum"),
    Comments=("comments_count", "sum"),
    Saves=("saves", "sum")
).reset_index(names="date")
if len(daily):
    st.plotly_chart(px.line(daily, x="date", y="ER", markers=True, title="Avg Engagement Rate over time"), use_container_width=True)

# Best hour / day
colA, colB = st.columns(2)
by_hour = df.groupby("hour")["engagement_rate"].mean().reset_index()
by_dow  = df.groupby("dow")["engagement_rate"].mean().reset_index()
colA.plotly_chart(px.bar(by_hour, x="hour", y="engagement_rate", title="Avg ER by hour"), use_container_width=True)
colB.plotly_chart(px.bar(by_dow, x="dow",  y="engagement_rate", title="Avg ER by weekday (0=Mon)"), use_container_width=True)

# Top hashtags (by ER with minimal volume filter)
st.markdown("**Top hashtags (by avg ER, with volume)**")
tag_df = df.explode("hashtags")
if "hashtags" in tag_df.columns and tag_df["hashtags"].notna().any():
    agg = (tag_df.groupby("hashtags")
            .agg(ER=("engagement_rate","mean"), Posts=("hashtags","count"))
            .reset_index())
    agg = agg[agg["Posts"] >= 3].sort_values(["ER","Posts"], ascending=[False, False]).head(20)
    st.dataframe(agg, use_container_width=True)
else:
    st.caption("No hashtags found in your captions.")

# -----------------------------
# Modeling: train simple models on your history
# -----------------------------
st.subheader("Forecasts (trained on your own history)")
feat = df[["media_type","hour","dow","hashtag_count","sent_score","roll_er_med_10","engagement_rate"]].dropna()
if len(feat) < 60:
    st.warning("Not enough posts to train reliable models (need ~60+). You can still explore BI above.")
else:
    # Label: tertiles of ER => High/Medium/Low
    q = feat["engagement_rate"].quantile([0.33, 0.66]).values
    def lab(x): return "Low" if x <= q[0] else ("Medium" if x <= q[1] else "High")
    feat["eng_level"] = feat["engagement_rate"].apply(lab)

    X = feat[["media_type","hour","dow","hashtag_count","sent_score","roll_er_med_10"]]
    y_cls = feat["eng_level"]
    y_reg = feat["engagement_rate"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["media_type"]),
        ("num", "passthrough", ["hour","dow","hashtag_count","sent_score","roll_er_med_10"])
    ])

    clf = Pipeline([
        ("prep", pre),
        ("rf", RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", random_state=42))
    ])
    reg = Pipeline([
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg.fit(Xtr2, ytr2)
    mae = mean_absolute_error(yte2, reg.predict(Xte2))

    st.code("Classification report (Engagement Level)\n" + classification_report(yte, yhat, zero_division=0), language="text")
    st.caption(f"Regression validation MAE (Engagement Rate): {mae:.4f} rate points")

    # -----------------------------
    # Planner: predict BEFORE posting
    # -----------------------------
    st.markdown("---")
    st.subheader("Plan a Post (predict before posting)")
    col1, col2, col3 = st.columns(3)
    media_type = col1.selectbox("Content type", sorted(df["media_type"].dropna().unique()))
    hour       = col2.number_input("Planned hour (0–23)", 0, 23, value=int(df["hour"].median() if df["hour"].notna().any() else 12))
    dow        = col3.selectbox("Day of week (0=Mon)", options=list(range(7)), index=3)

    cap = st.text_area("Caption / campaign text")
    hashtags = re.findall(r"#\w+", cap.lower())
    hashtag_count = len(hashtags)

    pos = ["love","great","best","wow","amazing","new","sale","win","happy","cool"]
    neg = ["bad","hate","worst","angry","late","issue","broken","sad","fail"]
    sent_score = sum(w in cap.lower() for w in pos) - sum(w in cap.lower() for w in neg)

    roll_er_med_10 = float(df["roll_er_med_10"].tail(1).iloc[0])

    sample = pd.DataFrame([{
        "media_type": media_type,
        "hour": hour,
        "dow": dow,
        "hashtag_count": hashtag_count,
        "sent_score": sent_score,
        "roll_er_med_10": roll_er_med_10
    }])

    if st.button("Predict for planned post"):
        pred_level = clf.predict(sample)[0]
        pred_er = float(reg.predict(sample)[0]) * 100
        st.success(f"Predicted Engagement Level: **{pred_level}**")
        st.info(f"Predicted Engagement Rate: **{pred_er:.2f}%**")

        # quick tip based on history
        best_hour = int(by_hour.sort_values("engagement_rate", ascending=False).iloc[0]["hour"])
        best_dow  = int(by_dow.sort_values("engagement_rate",  ascending=False).iloc[0]["dow"])
        st.caption(f"Tip: Historically best hour = {best_hour}, best weekday (0=Mon) = {best_dow}")
