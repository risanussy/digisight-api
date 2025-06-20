from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from geopy.geocoders import Nominatim
import time
import logging
import re

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

name_pattern = re.compile(
    r"""
    ^
    (?:Laporan\s+Reporter|Reporter|Laporan|Penulis)?  
    [^\w]*                                           
    (?P<name>[A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)        
    """,
    re.VERBOSE
)

def extract_reporter(row):
    """
    Kembalikan nama reporter jika bisa dideteksi di kolom content/title.
    """
    for col in [COL_CONTENT, COL_TITLE_B]:
        text = row.get(col)
        if pd.isna(text):
            continue
        m = name_pattern.match(str(text))
        if m:
            return m.group('name')
    return None

def parse_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except:
        return None

# def get_date_range():
#     today = datetime.now().date()
#     f = parse_date(request.args.get('from', '')) or today
#     t = parse_date(request.args.get('to',   '')) or today
#     if f > t:
#         f, t = t, f
#     return f, t

def get_date_range():
    f = parse_date(request.args.get('from'))
    t = parse_date(request.args.get('to'))
    if not f or not t:
        min_b = df_berita['news_date'].min()
        min_s = df_sosmed['date'].min()
        f = min(filter(pd.notna, [min_b, min_s])).date()
        t = datetime.now().date()
    if f > t:
        f, t = t, f
    return f, t


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -----------------------------------------------
# utils – letakkan di atas semua route
# -----------------------------------------------
def safe(val):
    """Convert pandas NaN/NaT/None -> None, else return as-is."""
    return None if pd.isna(val) else val
    
COL_MEDIA_B  = find_column(df_berita, ['media', 'source', 'outlet'])

COL_TITLE_B  = find_column(df_berita, ['title', 'judul', 'topic_title', 'topic'])
COL_LINK_B   = find_column(df_berita, ['url', 'link'])

COL_BYLINE_B = find_column(df_berita,
    ['byline','author','penulis','reporter','nama_reporter'])


COL_CONTENT  = find_column(df_berita, ['content', 'konten', 'isi'])
COL_SENT_B   = find_column(df_berita, ['sentiment_overall', 'sentimen', 'sentiment'])

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

# — Media Dominan —
@app.route('/api/dominant_media')
@cross_origin()
def dominant_media():
    d_from, d_to = get_date_range()
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to)

    if not COL_MEDIA_B:
        return jsonify([])

    mb = df_berita['news_date'].between(start, end)
    vc = df_berita[mb][COL_MEDIA_B].dropna().value_counts().head(10)
    return jsonify([{"media": k, "count": int(v)} for k, v in vc.items()])



# — Daftar Berita (list + filter) —
@app.route('/api/news')
@cross_origin()
def list_news():
    d_from, d_to = get_date_range()
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to)
    df = df_berita[df_berita['news_date'].between(start, end)]

    # filter media outlet
    media = request.args.get('media')
    if media and COL_MEDIA_B:
        df = df[df[COL_MEDIA_B].str.contains(media, case=False, na=False)]

    # filter sentiment/tone
    tone = request.args.get('tone')
    if tone and COL_SENT_B:
        df = df[df[COL_SENT_B].str.lower() == tone.lower()]

    # hasil
    result = []
    for _, row in df.iterrows():
        result.append({
            "media":     safe(row.get(COL_MEDIA_B)),
            "date":      safe(row['news_date']).date().isoformat() if pd.notna(row['news_date']) else None,
            "title":     safe(row.get(COL_TITLE_B)),
            "link":      safe(row.get(COL_LINK_B)),
            "content":   safe(row.get(COL_CONTENT)),
            "sentiment": safe(row.get(COL_SENT_B))
        })
    return jsonify(result)



# — Detail Berita by index —
@app.route('/api/news/<int:idx>')
@cross_origin()
def news_detail(idx):
    if idx < 0 or idx >= len(df_berita):
        return jsonify({"error": "Index out of range"}), 404
    row = df_berita.iloc[idx]
    return jsonify({
        "media":     row.get(COL_MEDIA_B),
        "date":      row['news_date'].date().isoformat(),
        "title":     row.get(COL_TOPIC_B),
        "link":      row.get('link'),
        "content":   row.get(COL_CONTENT),
        "sentiment": row.get(COL_SENT_B)
    })

# — Ad Value (nilai iklan) —
@app.route('/api/ad_value')
@cross_origin()
def ad_value():
    # Pendekatan 1: area (cm2) × advertising_rate (IDR/cm2)
    area = request.args.get('area', type=float)
    advertising_rate = request.args.get('advertising_rate', type=float)

    # Pendekatan 2: (word_count/30) × columns × rate_per_column
    word_count      = request.args.get('word_count', type=int)
    columns         = request.args.get('columns',    type=int)
    rate_per_column = request.args.get('rate_per_column', type=float)

    if area and advertising_rate:
        value = area * advertising_rate
    elif word_count and columns and rate_per_column:
        value = (word_count / 30) * columns * rate_per_column
    else:
        return jsonify({"error": "Missing parameters"}), 400

    return jsonify({"ad_value": value})

# — PR Value (publisitas) —
@app.route('/api/pr_value')
@cross_origin()
def pr_value():
    ad_val     = request.args.get('ad_value',  type=float)
    multiplier = request.args.get('multiplier', default=3, type=float)
    if ad_val is None:
        return jsonify({"error": "Missing ad_value"}), 400
    return jsonify({"pr_value": ad_val * multiplier})

# — Circulation & Rate Media —
@app.route('/api/media_rate')
@cross_origin()
def media_rate():
    media_data = [
        {"media": "Kompas", "circulation": 500000, "rate_per_cm": 80000,  "rate_online_CPM": 150000},
        {"media": "Detik",  "circulation": 800000, "rate_per_cm": 50000,  "rate_online_CPM": 120000},
        # … tambahkan sesuai kebutuhan …
    ]
    return jsonify(media_data)

# — Jurnalis Teraktif —
@app.route('/api/top_journalists')
@cross_origin()
def top_journalists():
    d_from, d_to = get_date_range()
    start, end  = pd.Timestamp(d_from), pd.Timestamp(d_to)
    mb = df_berita['news_date'].between(start, end)

    # 1) Gunakan kolom byline kalau ada
    if COL_BYLINE_B:
        series = df_berita[mb][COL_BYLINE_B].dropna()
    else:
        # 2) Jika tidak ada, ekstrak nama reporter dari konten/title
        series = df_berita[mb].apply(extract_reporter, axis=1).dropna()

    vc = series.value_counts().head(10)
    return jsonify([{"journalist": k, "count": int(v)} for k, v in vc.items()])

# ================================================================
# 🔧 Tambahan DETEKSI KOLOM Sosmed (letakkan di bawah blok find_column yg sudah ada)
# ================================================================
COL_CHANNEL_S  = find_column(df_sosmed, ['channel', 'platform'])
COL_POSTID_S   = find_column(df_sosmed, ['post_id', 'id', 'postid'])
COL_LIKES_S    = find_column(df_sosmed, ['likes', 'like'])
COL_COMMENTS_S = find_column(df_sosmed, ['comments', 'comment'])
COL_SHARES_S   = find_column(df_sosmed, ['shares', 'share', 'retweet'])
COL_VIEWS_S    = find_column(df_sosmed, ['views', 'view', 'impression'])
# jika perlu track keyword spesifik via kolom sendiri
COL_KEYWORD_S  = find_column(df_sosmed, ['keyword', 'key', 'kata_kunci'])

# ================================================================
# 🔧 Utilitas kecil
# ================================================================
def engagement_rate(row: pd.Series) -> float:
    """ER = (likes + comments + shares) / views × 100%"""
    likes    = row.get(COL_LIKES_S,    0) or 0
    comments = row.get(COL_COMMENTS_S, 0) or 0
    shares   = row.get(COL_SHARES_S,   0) or 0
    views    = row.get(COL_VIEWS_S,    0) or 0
    return round(((likes + comments + shares) / views) * 100, 4) if views else 0.0

# ================================================================
# 1) KEYWORD TRACKER  – total kemunculan keyword (top-N)
#    GET /api/keyword_tracker?keyword=<k>&top=10
# ================================================================
@app.route('/api/keyword_tracker')
@cross_origin()
def keyword_tracker():
    kw  = request.args.get('keyword', '').strip()
    top = request.args.get('top', default=20, type=int)
    if not kw:
        return jsonify({"error": "parameter ?keyword= wajib"}), 400

    # gunakan rentang tanggal global
    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    text_col = COL_TEXT_S or ''
    tmp = df_sosmed[mask][text_col].dropna().str.contains(fr'\b{re.escape(kw)}\b', case=False, na=False)
    df_kw = df_sosmed[mask & tmp]

    daily = (df_kw.groupby(df_kw['date'].dt.date)
                   .size()
                   .sort_index()
                   .tail(top))
    return jsonify([{"date": str(d), "count": int(c)} for d, c in daily.items()])

# ================================================================
# 2) TIMELINE KEYWORD  – detail harian per keyword
#    GET /api/timeline_keyword?keyword=...
# ================================================================
@app.route('/api/timeline_keyword')
@cross_origin()
def timeline_keyword():
    kw = request.args.get('keyword', '').strip()
    if not kw:
        return jsonify({"error": "parameter ?keyword= wajib"}), 400

    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    hit = df_sosmed[mask][COL_TEXT_S].str.contains(fr'\b{re.escape(kw)}\b', case=False, na=False)
    ts  = df_sosmed[mask & hit]

    series = ts.groupby(ts['date'].dt.date).size().sort_index()
    return jsonify([{"date": str(k), "count": int(v)} for k, v in series.items()])

# ================================================================
# 3) VIRAL KEYWORDS  – deteksi spike > threshold%
#    GET /api/viral_keywords?threshold=200
# ================================================================
@app.route('/api/viral_keywords')
@cross_origin()
def viral_keywords():
    thr   = request.args.get('threshold', type=float, default=200)
    limit = request.args.get('limit',     type=int,   default=10)

    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))

    # Kolom teks keyword / topic
    kw_col = COL_KEYWORD_S or COL_TOPIC_S
    if not kw_col:
        return jsonify([])

    # 1️⃣  Hitung frekuensi harian per-keyword
    grp = (df_sosmed.loc[mask]
           .groupby([kw_col, df_sosmed.loc[mask, 'date'].dt.date])
           .size()
           .rename('cnt')
           .reset_index())

    # 2️⃣  Cari baris “hari ini” + “kemarin”
    grp['prev'] = grp.groupby(kw_col)['cnt'].shift(1).fillna(0)
    grp = grp[grp['prev'] > 0]        # buang keyword yg baru muncul 1 hari

    grp['growth_pct'] = (grp['cnt'] - grp['prev']) / grp['prev'] * 100
    grp['status']     = grp['growth_pct'].ge(thr).map({True:'Viral', False:'Stabil'})

    top = (grp.sort_values('growth_pct', ascending=False)
              .head(limit))

    return jsonify([
        {
            "keyword":   r[kw_col],
            "date":      str(r['date']),
            "today":     int(r['cnt']),
            "prev":      int(r['prev']),
            "growth_pct":round(r['growth_pct'],2),
            "status":    r['status']
        } for _, r in top.iterrows()
    ])
    
# ================================================================
# 4) POST METRICS VIEW  – likes, comments, shares, views
#    GET /api/post_metrics?channel=Instagram
# ================================================================
@app.route('/api/post_metrics')
@cross_origin()
def post_metrics():
    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))

    ch = request.args.get('channel')
    if ch and COL_CHANNEL_S:
        mask &= df_sosmed[COL_CHANNEL_S].str.lower() == ch.lower()

    cols_req = [COL_POSTID_S, COL_CHANNEL_S,
                COL_LIKES_S, COL_COMMENTS_S, COL_SHARES_S, COL_VIEWS_S]
    cols = [c for c in cols_req if c]

    df = df_sosmed[mask][cols].dropna(subset=[COL_POSTID_S])

    result = []
    for _, r in df.iterrows():
        result.append({
            "post_id":  r[COL_POSTID_S],
            "channel":  r.get(COL_CHANNEL_S),
            "likes":    int(r.get(COL_LIKES_S,    0) or 0),
            "comments": int(r.get(COL_COMMENTS_S, 0) or 0),
            "shares":   int(r.get(COL_SHARES_S,   0) or 0),
            "views":    int(r.get(COL_VIEWS_S,    0) or 0)
        })
    return jsonify(result)

# ================================================================
# 5) ENGAGEMENT RATE PER POST
#    GET /api/engagement_rates?channel=Twitter
# ================================================================
@app.route('/api/engagement_rates')
@cross_origin()
def engagement_rates():
    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))

    ch = request.args.get('channel')
    if ch and COL_CHANNEL_S:
        mask &= df_sosmed[COL_CHANNEL_S].str.lower() == ch.lower()

    df = df_sosmed[mask].copy()
    df['er'] = df.apply(engagement_rate, axis=1)

    result = []
    for _, r in df.iterrows():
        result.append({
            "post_id": r.get(COL_POSTID_S),
            "channel": r.get(COL_CHANNEL_S),
            "er_pct":  r['er']
        })
    # urutkan terbesar
    result.sort(key=lambda x: x['er_pct'], reverse=True)
    return jsonify(result)

# ================================================================
# 6) CHANNEL-BASED SENTIMENT
#    GET /api/channel_sentiment
# ================================================================
@app.route('/api/channel_sentiment')
@cross_origin()
def channel_sentiment():
    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))

    if not (COL_CHANNEL_S and COL_SENT_S):
        return jsonify([])

    df = df_sosmed[mask][[COL_CHANNEL_S, COL_SENT_S]].dropna()
    if df.empty:
        return jsonify([])

    out = {}
    for (chan, sent), grp in df.groupby([COL_CHANNEL_S, COL_SENT_S]):
        out.setdefault(chan, {"positive": 0, "neutral": 0, "negative": 0})
        s = sent.lower()
        if 'pos' in s:
            out[chan]['positive'] += len(grp)
        elif 'neg' in s:
            out[chan]['negative'] += len(grp)
        else:
            out[chan]['neutral']  += len(grp)
    return jsonify(out)

# ================================================================
# 7) ENTITY-BASED SENTIMENT  (sederhana: nama title-case ≥2 kata)
#    GET /api/entity_sentiment
# ================================================================
@app.route('/api/entity_sentiment')
@cross_origin()
def entity_sentiment():
    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    if not COL_TEXT_S or not COL_SENT_S:
        return jsonify([])

    pat = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')  # ex: "Anies Baswedan"
    records = []
    for _, row in df_sosmed[mask][[COL_TEXT_S, COL_SENT_S]].dropna().iterrows():
        sent = row[COL_SENT_S].lower()
        label = 'positive' if 'pos' in sent else 'negative' if 'neg' in sent else 'neutral'
        for m in pat.findall(str(row[COL_TEXT_S])):
            records.append((m, label))

    if not records:
        return jsonify([])

    df_rec = pd.DataFrame(records, columns=['entity', 'sent'])
    pivot  = df_rec.pivot_table(index='entity', columns='sent', aggfunc='size', fill_value=0)
    pivot  = pivot[['positive','neutral','negative']] if set(['positive','neutral','negative']).issubset(pivot.columns) else pivot
    pivot  = pivot.sort_values(by='positive', ascending=False).head(20)

    return jsonify([
        {"entity": idx,
         "positive": int(r.get('positive', 0)),
         "neutral":  int(r.get('neutral',  0)),
         "negative": int(r.get('negative', 0))}
        for idx, r in pivot.iterrows()
    ])

# ================================================================
# 8) DAILY SENTIMENT HEATMAP
#    GET /api/daily_sentiment
# ================================================================
@app.route('/api/daily_sentiment')
@cross_origin()
def daily_sentiment():
    d_from, d_to = get_date_range()
    mask = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    if not COL_SENT_S:
        return jsonify([])

    df = df_sosmed[mask][['date', COL_SENT_S]].dropna()
    if df.empty:
        return jsonify([])

    df['day'] = df['date'].dt.date
    pivot = (df.pivot_table(index='day', columns=COL_SENT_S, aggfunc='size', fill_value=0)
               .rename(columns=str.lower)  # pos/neg/netral
               .sort_index())
    out = []
    for day, row in pivot.iterrows():
        out.append({
            "date":      str(day),
            "positive":  int(row.get('positif',  row.get('positive', 0))),
            "neutral":   int(row.get('netral',   row.get('neutral',  0))),
            "negative":  int(row.get('negatif',  row.get('negative', 0))),
        })
    return jsonify(out)

# ================================================================
#  ⚡  UTILITAS ENTITY EXTRACTION  (letakkan di bawah blok utilitas)
# ================================================================
# Pola sederhana “dua kata atau lebih berawalan huruf besar”
ENTITY_PAT = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')

def extract_entities(text: str) -> set[str]:
    """Kembalikan SET entity (tokoh/organisasi) dari satu teks."""
    if not text:
        return set()
    return set(ENTITY_PAT.findall(str(text)))

# ================================================================
#  1) TOP ENTITIES  – daftar tokoh/org + jumlah mention
#     GET /api/top_entities?top=20
# ================================================================
@app.route('/api/top_entities')
@cross_origin()
def top_entities():
    top = request.args.get('top', default=20, type=int)
    d_from, d_to = get_date_range()
    mask_b = df_berita['news_date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    mask_s = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))

    cnt = Counter()

    for txt in df_berita.loc[mask_b, COL_CONTENT].dropna():
        cnt.update(extract_entities(txt))
    for txt in df_sosmed.loc[mask_s, COL_TEXT_S].dropna():
        cnt.update(extract_entities(txt))

    return jsonify([
        {"entity": ent, "mentions": int(n)}
        for ent, n in cnt.most_common(top)
    ])

# ================================================================
#  2) ENTITY TREND  – time-series jumlah mention per hari
#     GET /api/entity_trend?entity=<nama>
# ================================================================
@app.route('/api/entity_trend')
@cross_origin()
def entity_trend():
    ent = request.args.get('entity', '').strip()
    if not ent:
        return jsonify({"error": "parameter ?entity= wajib"}), 400

    d_from, d_to = get_date_range()
    # Media
    mb = df_berita['news_date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    mask_b = mb & df_berita[COL_CONTENT].str.contains(fr'\b{re.escape(ent)}\b',
                                                      case=False, na=False)
    ts_b = (df_berita.loc[mask_b]
                     .groupby(df_berita.loc[mask_b, 'news_date'].dt.date)
                     .size())
    # Sosmed
    ms = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    mask_s = ms & df_sosmed[COL_TEXT_S].str.contains(fr'\b{re.escape(ent)}\b',
                                                     case=False, na=False)
    ts_s = (df_sosmed.loc[mask_s]
                      .groupby(df_sosmed.loc[mask_s, 'date'].dt.date)
                      .size())

    # Gabungkan; isi 0 jika tidak ada
    all_days = sorted(set(ts_b.index) | set(ts_s.index))
    out = []
    for day in all_days:
        out.append({
            "date": str(day),
            "media":  int(ts_b.get(day, 0)),
            "sosmed": int(ts_s.get(day, 0)),
            "total":  int(ts_b.get(day, 0) + ts_s.get(day, 0))
        })
    return jsonify(out)

# ================================================================
#  3) ENTITY SENTIMENT BY SOURCE – perbandingan sentimen media vs sosmed
#     GET /api/entity_sentiment_source?entity=<nama>
# ================================================================
@app.route('/api/entity_sentiment_source')
@cross_origin()
def entity_sentiment_source():
    ent = request.args.get('entity', '').strip()
    if not ent:
        return jsonify({"error": "parameter ?entity= wajib"}), 400

    d_from, d_to = get_date_range()
    res = {"media": {"positive": 0, "neutral": 0, "negative": 0},
           "sosmed": {"positive": 0, "neutral": 0, "negative": 0}}

    # ------ MEDIA ------
    if COL_SENT_B:
        mb = df_berita['news_date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
        mask_b = mb & df_berita[COL_CONTENT].str.contains(fr'\b{re.escape(ent)}\b',
                                                          case=False, na=False)
        for sent, grp in df_berita.loc[mask_b].groupby(COL_SENT_B):
            s = sent.lower()
            if 'pos' in s:      res["media"]["positive"] += len(grp)
            elif 'neg' in s:    res["media"]["negative"] += len(grp)
            else:               res["media"]["neutral"]  += len(grp)

    # ------ SOSMED ------
    if COL_SENT_S:
        ms = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
        mask_s = ms & df_sosmed[COL_TEXT_S].str.contains(fr'\b{re.escape(ent)}\b',
                                                         case=False, na=False)
        for sent, grp in df_sosmed.loc[mask_s].groupby(COL_SENT_S):
            s = sent.lower()
            if 'pos' in s:      res["sosmed"]["positive"] += len(grp)
            elif 'neg' in s:    res["sosmed"]["negative"] += len(grp)
            else:               res["sosmed"]["neutral"]  += len(grp)

    return jsonify(res)

# ================================================================
#  4) MENTION NETWORK  – nodes & weighted edges co-occurrence
#     GET /api/mention_network?min_edge=3&top_nodes=30
# ================================================================
@app.route('/api/mention_network')
@cross_origin()
def mention_network():
    from itertools import combinations

    min_edge  = request.args.get('min_edge',  default=3,  type=int)
    top_nodes = request.args.get('top_nodes', default=30, type=int)

    d_from, d_to = get_date_range()
    mask_b = df_berita['news_date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))
    mask_s = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to))

    # ------- Kumpulkan entity per dokumen -------
    docs = []
    for txt in df_berita.loc[mask_b, COL_CONTENT].dropna():
        docs.append(extract_entities(txt))
    for txt in df_sosmed.loc[mask_s, COL_TEXT_S].dropna():
        docs.append(extract_entities(txt))

    edge_cnt = Counter()
    node_cnt = Counter()

    for ents in docs:
        ents = [e for e in ents if e]               # buang kosong
        if len(ents) < 2:
            continue
        node_cnt.update(ents)
        for a, b in combinations(sorted(ents), 2):  # pastikan (A,B)==(B,A)
            edge_cnt[(a, b)] += 1

    # Filter edge di bawah threshold
    edge_out = [{"source": a, "target": b, "weight": w}
                for (a, b), w in edge_cnt.items() if w >= min_edge]

    # Ambil TOP node berdasarkan frekuensi
    top = {n for n, _ in node_cnt.most_common(top_nodes)}
    edge_out = [e for e in edge_out if e["source"] in top and e["target"] in top]
    node_out = [{"id": n, "count": int(c)} for n, c in node_cnt.items() if n in top]

    return jsonify({"nodes": node_out, "edges": edge_out})

# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 🔖  KATEGORI ISU sederhana (keyword → kategori)
# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
CATEGORY_MAP = {
    "Politik":  ["pemilu", "parpol", "pilpres", "politik", "kampanye"],
    "Bencana":  ["banjir", "gempa", "longsor", "kebakaran", "tsunami"],
    "Sosial":   ["demo", "pendidikan", "kesehatan", "kemiskinan", "buruh"],
    "Kriminal": ["korupsi", "pencurian", "perampokan", "pembunuhan",
                 "narkoba", "penipuan"],
    "Ekonomi":  ["inflasi", "bbm", "ekonomi", "investasi", "rupiah"],
    "Transportasi": ["lalu lintas", "kecelakaan", "jalan tol", "kereta"],
    # … tambah sendiri bila perlu …
}

# ================================================================
# 9) VOLUME ANOMALY DETECTION  – Early Warning
#    GET /api/volume_anomaly?baseline_days=7&z=2
# ================================================================
@app.route('/api/volume_anomaly')
@cross_origin()
def volume_anomaly():
    """
    Deteksi lonjakan volume harian (berita+sosmed) berdasarkan Z-Score.
    • baseline_days : panjang window rata-rata harian (default 7 hari)
    • z             : ambang Z-Score (default 2)
    """
    base_n = request.args.get('baseline_days', default=7, type=int)
    z_thr  = request.args.get('z',              default=2.0, type=float)

    # gunakan range global agar konsisten dengan filter <from,to>
    d_from, d_to = get_date_range()
    today = d_to
 
    # jika tanggal 'today' tidak ada di Series, fallback ke tanggal data terakhir
    mask_b = df_berita['news_date'].dt.date <= today
    mask_s = df_sosmed['date'].dt.date      <= today
    daily_b = df_berita.loc[mask_b].groupby(df_berita['news_date'].dt.date).size()
    daily_s = df_sosmed.loc[mask_s].groupby(df_sosmed['date'].dt.date).size()
    daily   = (daily_b.add(daily_s, fill_value=0)).sort_index()
 

    # ------ hitung volume per hari ------
    mask_b = df_berita['news_date'].dt.date <= today
    mask_s = df_sosmed['date'].dt.date      <= today
    daily_b = df_berita.loc[mask_b].groupby(df_berita['news_date'].dt.date).size()
    daily_s = df_sosmed.loc[mask_s].groupby(df_sosmed['date'].dt.date).size()
    daily   = (daily_b.add(daily_s, fill_value=0)).sort_index()      # Series

    if today not in daily.index:
        today = daily.index[-1]

    # ------ baseline 7 hari sebelumnya ------
    baseline_range = [today - timedelta(days=i) for i in range(1, base_n+1)]
    baseline_vals  = [daily.get(d, 0) for d in baseline_range]

    mean = float(np.mean(baseline_vals)) if baseline_vals else 0.0
    std  = float(np.std (baseline_vals, ddof=1)) if baseline_vals else 0.0
    z    = (daily[today] - mean) / std if std else 0.0

    return jsonify({
        "date":          str(today),
        "volume_today":  int(daily[today]),
        "baseline_mean": round(mean, 2),
        "baseline_std":  round(std,  2),
        "z_score":       round(z,   2),
        "status":        "Anomaly ⚠️" if z >= z_thr else "Normal ✅"
    })

# ================================================================
# 10) TRENDING KEYWORDS 24 Jam
#     GET /api/trending_keywords?threshold=300&limit=10
# ================================================================
@app.route('/api/trending_keywords')
@cross_origin()
def trending_keywords():
    thr   = request.args.get('threshold', default=300.0, type=float)
    limit = request.args.get('limit',     default=10,   type=int)

    now   = pd.Timestamp(datetime.now())
    last24  = now - pd.Timedelta(hours=24)
    prev24  = now - pd.Timedelta(hours=48)

    kw_col = COL_KEYWORD_S or COL_TOPIC_S
    if not kw_col:
        return jsonify([])

    # Frekuensi keyword 24 jam terakhir vs 24 jam sebelumnya
    recent = (df_sosmed[df_sosmed['date'] >= last24]
                .groupby(kw_col).size())
    prev   = (df_sosmed[df_sosmed['date'].between(prev24, last24)]
                .groupby(kw_col).size())

    out = []
    for kw, cnt_now in recent.items():
        cnt_prev = prev.get(kw, 0)
        if cnt_prev == 0:       # hindari div/0; abaikan keyword baru muncul
            continue
        growth = (cnt_now - cnt_prev) / cnt_prev * 100
        if growth >= thr:
            out.append({
                "keyword": kw,
                "count_24h": int(cnt_now),
                "count_prev": int(cnt_prev),
                "growth_pct": round(growth, 2)
            })

    # urutkan terbesar & batasi
    out.sort(key=lambda x: x["growth_pct"], reverse=True)
    return jsonify(out[:limit])

# ================================================================
# 11) TIMELINE ISU (+fase Muncul/Viral/Redam)
#     GET /api/issue_timeline?issue=<keyword>
# ================================================================
@app.route('/api/issue_timeline')
@cross_origin()
def issue_timeline():
    issue = request.args.get('issue', '').strip()
    if not issue:
        return jsonify({"error": "parameter ?issue= wajib"}), 400

    d_from, d_to = get_date_range()
    mask_b = df_berita['news_date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to)) \
            & df_berita[COL_CONTENT].str.contains(issue, case=False, na=False)
    mask_s = df_sosmed['date'].between(pd.Timestamp(d_from), pd.Timestamp(d_to)) \
            & df_sosmed[COL_TEXT_S].str.contains(issue, case=False, na=False)

    # volume harian
    ts_b = df_berita.loc[mask_b].groupby(df_berita.loc[mask_b,'news_date'].dt.date).size()
    ts_s = df_sosmed.loc[mask_s].groupby(df_sosmed.loc[mask_s,'date'].dt.date).size()
    series = (ts_b.add(ts_s, fill_value=0)).sort_index()

    if series.empty:
        return jsonify([])

    mean = float(series.mean())
    std  = float(series.std(ddof=1)) or 1.0  # hindari std=0

    out = []
    for day, cnt in series.items():
        z = (cnt - mean) / std
        if z >= 3:
            phase = "Viral 🔥"
        elif z > 0:
            phase = "Muncul ↗️"
        else:
            phase = "Redam ↘️"
        out.append({
            "date": str(day),
            "count": int(cnt),
            "z_score": round(z,2),
            "phase": phase
        })
    return jsonify(out)

# ================================================================
# 12) ISSUE CATEGORY MAP
#     GET /api/issue_categories?top=50
# ================================================================
@app.route('/api/issue_categories')
@cross_origin()
def issue_categories():
    top = request.args.get('top', default=50, type=int)

    d_from, d_to = get_date_range()
    start, end = pd.Timestamp(d_from), pd.Timestamp(d_to)

    # Kumpulkan seluruh teks topik/keyword
    topics = []
    if COL_TOPIC_B:
        topics += df_berita[df_berita['news_date'].between(start, end)][COL_TOPIC_B].dropna().tolist()
    if COL_TOPIC_S:
        topics += df_sosmed[df_sosmed['date'].between(start, end)][COL_TOPIC_S].dropna().tolist()

    cnt_cat = Counter()
    for t in topics:
        t_lc = str(t).lower()
        matched = False
        for cat, kws in CATEGORY_MAP.items():
            if any(kw in t_lc for kw in kws):
                cnt_cat[cat] += 1
                matched = True
                break
        if not matched:
            cnt_cat["Lain-lain"] += 1

    # urutkan & batasi
    out = [{"category": c, "count": int(n)} for c, n in cnt_cat.most_common(top)]
    return jsonify(out)

# ================================================================
# 🔧 HELPER: deteksi kategori berdasar kata
# ================================================================
def detect_category(keyword: str) -> str:
    kw_lc = keyword.lower()
    for cat, kws in CATEGORY_MAP.items():
        if any(k in kw_lc for k in kws):
            return cat
    return "Lain-lain"

# ================================================================
# 13) ISSUE SUMMARY TABLE
#     GET /api/issue_summary?limit=10&threshold=300
# ================================================================
@app.route('/api/issue_summary')
@cross_origin()
def issue_summary():
    """
    Ringkasan isu harian (untuk tabel 'Ringkasan Isu'):
        • Ambil keyword yang trending ≥ threshold% (24 jam).
        • Tentukan fase (Muncul / Viral / Redam) berdasar Z-Score terkini.
        • Mapping kategori via CATEGORY_MAP.
    """
    limit = request.args.get('limit',     default=10,   type=int)
    thr   = request.args.get('threshold', default=300., type=float)

    now    = pd.Timestamp(datetime.now())
    last24 = now - pd.Timedelta(hours=24)
    prev24 = now - pd.Timedelta(hours=48)

    kw_col = COL_KEYWORD_S or COL_TOPIC_S
    if not kw_col:
        return jsonify([])

    # — 1. frekuensi keyword 24 jam terakhir vs sebelumnya —
    recent = df_sosmed[df_sosmed['date'] >= last24].groupby(kw_col).size()
    prev   = (df_sosmed[df_sosmed['date'].between(prev24, last24)]
                .groupby(kw_col).size())

    # — 2. hitung growth % dan pilih yang ≥ thr —
    rows = []
    for kw, cnt_now in recent.items():
        cnt_prev = prev.get(kw, 0)
        if cnt_prev == 0:
            continue
        growth = (cnt_now - cnt_prev) / cnt_prev * 100
        if growth < thr:
            continue   # belum dianggap trending
        rows.append((kw, cnt_now, cnt_prev, growth))

    # — 3. urutkan terbesar & batasi —
    rows.sort(key=lambda t: t[3], reverse=True)
    rows = rows[:limit]

    # — 4. tentukan fase per-keyword (pakai Z-Score timeline harian) —
    out = []
    for kw, cnt_now, cnt_prev, growth in rows:
        mask = df_sosmed[kw_col].str.contains(fr'\b{re.escape(kw)}\b',
                                              case=False, na=False) \
               & (df_sosmed['date'] >= last24)
        # timeline 7 hari ke belakang
        ts = (df_sosmed[mask]
                .groupby(df_sosmed[mask]['date'].dt.date)
                .size()
                .sort_index())
        if ts.empty:
            phase = "Muncul"
        else:
            mean = ts.mean()
            std  = ts.std(ddof=1) or 1.0
            z    = (ts.iloc[-1] - mean) / std
            phase = "Viral"   if z >= 3 else "Muncul" if z > 0 else "Redam"

        out.append({
            "time":    now.strftime('%H:%M'),
            "keyword": kw,
            "phase":   phase,
            "category": detect_category(kw),
            "trend":  "Warning" if growth >= thr * 1.5 else "Monitor",
        })

    return jsonify(out)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
