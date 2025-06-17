from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
from datetime import datetime
from collections import Counter
from geopy.geocoders import Nominatim
import time
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
logging.basicConfig(level=logging.INFO)

# — Load & Prepare Data —
df_berita = pd.read_excel('data/berita.xlsx')
df_sosmed = pd.read_excel('data/sosmed.xlsx')

# Pastikan jadi datetime64[ns]
df_berita['news_date'] = pd.to_datetime(df_berita['news_date'], errors='coerce')
df_sosmed['date']      = pd.to_datetime(df_sosmed['date'],      errors='coerce')

geolocator = Nominatim(user_agent="digisight_app")

def parse_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except:
        return None

def get_date_range():
    today = datetime.now().date()
    f = parse_date(request.args.get('from', '')) or today
    t = parse_date(request.args.get('to',   '')) or today
    if f > t:
        f, t = t, f
    return f, t

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL_CONTENT = find_column(df_berita, ['content','konten','isi'])
COL_SENT_B  = find_column(df_berita, ['sentiment_overall','sentimen','sentiment'])
COL_TOPIC_B = find_column(df_berita, ['topic_title','topic','judul'])

COL_TEXT_S  = find_column(df_sosmed,  ['text','konten','isi'])
COL_SENT_S  = find_column(df_sosmed,  ['sentimen','sentiment','tone'])
COL_TOPIC_S = find_column(df_sosmed,  ['topic_title','topic','judul'])

@app.errorhandler(Exception)
@cross_origin()
def handle_exception(e):
    app.logger.error("Unhandled Exception", exc_info=e)
    return jsonify({ "error": str(e) }), 500

# — Date Range Endpoint —
@app.route('/api/date_range')
@cross_origin()
def date_range():
    min_b_ts = df_berita['news_date'].dropna().min()
    min_s_ts = df_sosmed['date'].dropna().min()

    min_b = min_b_ts.date() if not pd.isna(min_b_ts) else datetime.now().date()
    min_s = min_s_ts.date() if not pd.isna(min_s_ts) else datetime.now().date()
    earliest = min(min_b, min_s)
    today = datetime.now().date()

    return jsonify({
        "from": earliest.isoformat(),
        "to":   today.isoformat()
    })

# — Total Posts —
@app.route('/api/total_posts_today')
@cross_origin()
def total_posts_today():
    d_from, d_to = get_date_range()
    # Convert python.date → pandas.Timestamp
    start = pd.Timestamp(d_from)
    end   = pd.Timestamp(d_to)

    mb = df_berita['news_date'].between(start, end)
    ms = df_sosmed['date'].between(start, end)

    cnt_b = int(df_berita[mb].shape[0])
    cnt_s = int(df_sosmed[ms].shape[0])
    return jsonify({
        "overall": cnt_b + cnt_s,
        "online":  cnt_b,
        "sosmed":  cnt_s
    })

# — Sentiment —
@app.route('/api/sentiment_today')
@cross_origin()
def sentiment_today():
    d_from, d_to = get_date_range()
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to)

    mb = df_berita['news_date'].between(start, end)
    ms = df_sosmed['date'].between(start, end)

    b = df_berita[mb][COL_SENT_B].dropna() if COL_SENT_B else pd.Series(dtype=str)
    s = df_sosmed[ms][COL_SENT_S].dropna() if COL_SENT_S else pd.Series(dtype=str)
    all_sent = pd.concat([b, s])
    vc = all_sent.value_counts()

    return jsonify({
        "positive": int(vc.get('Positif', vc.get('positive', 0))),
        "neutral":  int(vc.get('Netral',  vc.get('neutral',  0))),
        "negative": int(vc.get('Negatif', vc.get('negative', 0)))
    })

# — Top Issues —
@app.route('/api/top_issues')
@cross_origin()
def top_issues():
    d_from, d_to = get_date_range()
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to)

    mb = df_berita['news_date'].between(start, end)
    ms = df_sosmed['date'].between(start, end)

    t1 = df_berita[mb][COL_TOPIC_B].dropna() if COL_TOPIC_B else pd.Series(dtype=str)
    t2 = df_sosmed[ms][COL_TOPIC_S].dropna() if COL_TOPIC_S else pd.Series(dtype=str)
    topics = pd.concat([t1, t2])

    if topics.empty:
        return jsonify([])

    topn = topics.value_counts().head(10)
    return jsonify([{"text": k, "value": int(v)} for k, v in topn.items()])

# — Locations —
@app.route('/api/locations')
@cross_origin()
def locations():
    d_from, d_to = get_date_range()
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to)

    mb = df_berita['news_date'].between(start, end)
    ms = df_sosmed['date'].between(start, end)

    teks_b = df_berita[mb][COL_CONTENT].dropna().tolist() if COL_CONTENT else []
    teks_s = df_sosmed[ms][COL_TEXT_S].dropna().tolist()   if COL_TEXT_S   else []
    texts  = teks_b + teks_s

    freq = Counter(
        w.strip('.,?!;:"()')
        for txt in texts for w in txt.split()
    )
    candidates = [w for w,c in freq.items() if w and w[0].isupper() and c>=3]
    top5 = sorted(candidates, key=lambda w: freq[w], reverse=True)[:5]

    result = []
    for name in top5:
        try:
            loc = geolocator.geocode(f"{name}, Indonesia", exactly_one=True, timeout=10)
            if loc:
                result.append({
                    "name":  name,
                    "value": freq[name],
                    "lat":   loc.latitude,
                    "lng":   loc.longitude
                })
        except Exception as e:
            app.logger.warning(f"Geocode fail {name}: {e}")
        time.sleep(1)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
