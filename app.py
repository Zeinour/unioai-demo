import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# -----------------------------
# UnioAI Demo (Streamlit)
# -----------------------------
st.set_page_config(page_title="UnioAI Demo", layout="wide")

def load_logo():
    for name in ["assets/unioai_logo.png", "assets/logo.png", "unioai_logo.png", "logo.png"]:
        try:
            return Image.open(name)
        except Exception:
            continue
    return None

with st.sidebar:
    logo = load_logo()
    if logo is not None:
        st.image(logo, use_column_width=True)
    st.markdown("<h2 style='text-align:center; letter-spacing:0.5px;'>UnioAI</h2>", unsafe_allow_html=True)
    st.caption("Bankacılıkta Akıllı Borç Yönetimi ve Tahsilat Analitiği")
    st.divider()

st.title("UnioAI — Tahsilat Simülasyonu & Analitik Panel")

st.info(
    "Bu demo, sunumunuzdaki UnioAI yaklaşımını göstermek için hazırlanmış **simülasyon** uygulamasıdır. "
    "Gerçek banka verisi içermez; sentetik veri veya yüklediğiniz CSV ile çalışır."
)

@st.cache_data
def generate_synthetic(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "musteri_id": range(1, n+1),
        "yas": rng.integers(18, 75, n),
        "gelir_duzeyi": rng.choice(["Dusuk", "Orta", "Yuksek"], n, p=[0.4, 0.45, 0.15]),
        "gecikme_sayisi": rng.integers(0, 8, n),
        "toplam_borc": np.round(rng.uniform(500, 30000, n), 2),
        "gecikme_gunu": rng.integers(0, 240, n),
        "son_odeme_uzerinden_gun": rng.integers(0, 365, n),
        "kanal_tercihi": rng.choice(["SMS", "Arama", "E-posta"], n, p=[0.45, 0.4, 0.15]),
    })

    # Ödeme olasılığı (basit sentetik ilişki)
    p = (
        0.58
        - 0.035*df["gecikme_sayisi"]
        - 0.0020*(df["gecikme_gunu"])
        - 0.000017*(df["toplam_borc"])
        - 0.0010*(df["son_odeme_uzerinden_gun"])
        + df["gelir_duzeyi"].map({"Dusuk": -0.08, "Orta": 0.0, "Yuksek": 0.08})
    )
    p = np.clip(p, 0.02, 0.98)
    df["geri_odeme"] = (rng.random(n) < p).astype(int)

    # DPD (Days Past Due) demo alanı
    df["dpd"] = df["gecikme_gunu"]
    return df

def encode_df(df):
    out = df.copy()
    cat_cols = [c for c in ["gelir_duzeyi", "kanal_tercihi"] if c in out.columns]
    if cat_cols:
        out = pd.get_dummies(out, columns=cat_cols, drop_first=True)
    return out

def suggest_channel_time(score, tercih):
    # Basit strateji: skora göre kanal/zaman penceresi
    if score >= 80:
        kanal = tercih if tercih in ["SMS", "Arama", "E-posta"] else "Arama"
        zaman = "09:00–11:00"
    elif score >= 60:
        kanal = "Arama"
        zaman = "11:00–13:00"
    elif score >= 40:
        kanal = "SMS"
        zaman = "13:00–15:00"
    else:
        kanal = "E-posta"
        zaman = "16:00–18:00"
    return kanal, zaman

def followup_template(name_or_id, channel, window, score):
    if channel == "SMS":
        msg = (f"Merhaba {name_or_id}, borç/prim ödemeleriniz için sizi bilgilendirmek isteriz. "
               f"Size uygun bir zamanda yardımcı olabiliriz. İsterseniz bu kanaldan dönüş yapabilirsiniz.")
    elif channel == "Arama":
        msg = (f"Merhaba {name_or_id}, ödemelerinizle ilgili sizi {window} saatleri arasında aramak istiyoruz. "
               "Uygun olduğunuz zaman dilimini iletirseniz memnun oluruz.")
    else:
        msg = (f"Merhaba {name_or_id}, ödemeleriniz hakkında bilgilendirme ve destek sunmak için iletişime geçiyoruz. "
               "Size uygun bir zamanda dönüş yapabiliriz.")
    return msg + f" (Öncelik skoru: {score:.1f})"

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Veri & Simülasyon Ayarları")

source = st.sidebar.radio("Veri Kaynağı", ["Sentetik veri (hazır)", "CSV yükle"])
if source == "CSV yükle":
    st.sidebar.write("Beklenen sütunlar:")
    st.sidebar.code(
        "musteri_id,yas,gelir_duzeyi,gecikme_sayisi,toplam_borc,gecikme_gunu,son_odeme_uzerinden_gun,kanal_tercihi,geri_odeme"
    )
    f = st.sidebar.file_uploader("CSV yükle", type=["csv"])
    if f:
        df = pd.read_csv(f)
    else:
        st.warning("Dosya yüklenmedi, sentetik veri kullanılıyor.")
        df = generate_synthetic()
else:
    seed = st.sidebar.number_input("Rastgele tohum (seed)", min_value=0, max_value=9999, value=42, step=1)
    nrows = st.sidebar.slider("Sentetik kayıt adedi", 200, 2000, 500, 50)
    df = generate_synthetic(n=nrows, seed=int(seed))

model_type = st.sidebar.selectbox("Model", ["RandomForest", "LogisticRegression"])
test_size = st.sidebar.slider("Test oranı", 0.1, 0.4, 0.25, 0.05)
threshold = st.sidebar.slider("Skor eşiği (0–100)", 0, 100, 70, 1)
capacity = st.sidebar.number_input("Günlük kapasite (çağrı/e-posta adedi)", min_value=5, max_value=2000, value=50, step=5)

st.sidebar.divider()
st.sidebar.header("Finansal Etki Varsayımları")
lost_portfolio = st.sidebar.number_input("Kaybedilen portföy (₺)", min_value=0.0, value=5_900_000_000.0, step=100_000_000.0, format="%.0f")
improvement = st.sidebar.slider("İyileşme varsayımı (%)", 0.0, 20.0, 5.0, 0.5) / 100.0

# -----------------------------
# Model prep
# -----------------------------
target_col = "geri_odeme"
df_enc = encode_df(df)
X = df_enc.drop(columns=[target_col], errors="ignore")
y = df_enc[target_col] if target_col in df_enc else None
if y is None:
    st.error("Hedef kolon (geri_odeme) bulunamadı. CSV'nize bu sütunu ekleyin ya da sentetik veri kullanın.")
    st.stop()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42, stratify=y
)

if model_type == "RandomForest":
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    )
else:
    model = LogisticRegression(max_iter=400, class_weight="balanced")

model.fit(X_train, y_train)
proba_all = model.predict_proba(X_scaled)[:, 1]
df["tahsilat_skoru"] = np.round(proba_all * 100, 1)

# Basit NPL proxy: düşük skor + yüksek gecikme = riskli
df["risk_segmenti"] = pd.cut(
    df["tahsilat_skoru"],
    bins=[-0.1, 40, 60, 80, 100.1],
    labels=["Yüksek Risk", "Orta Risk", "Düşük Risk", "Çok Düşük Risk"]
)
npl_proxy = (df["tahsilat_skoru"] < 40).mean()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 Simülasyon", "💬 Analitik Asistan", "📤 Dışa Aktarım"])

with tab1:
    st.subheader("📊 Genel Performans Göstergeleri")
    toplam_alacak = float(df["toplam_borc"].sum()) if "toplam_borc" in df.columns else 0.0
    borc_odendi = float(df.loc[df["geri_odeme"] == 1, "toplam_borc"].sum()) if "geri_odeme" in df.columns else 0.0
    geri_kazanim_orani = (borc_odendi / toplam_alacak) if toplam_alacak > 0 else 0.0
    yuksek_pay = float((df["tahsilat_skoru"] >= threshold).mean()) if "tahsilat_skoru" in df else 0.0

    # Sunumdaki "operasyonel verimlilik" gibi bir demo metriği
    operasyonel_verimlilik = yuksek_pay * 0.35 + 0.1

    ek_nakit = lost_portfolio * improvement

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Alacak", f"{toplam_alacak:,.0f} ₺")
    k2.metric("Geri Kazanım Oranı", f"{geri_kazanim_orani*100:.1f} %")
    k3.metric("NPL (proxy)", f"{npl_proxy*100:.1f} %")
    k4.metric("Tahmini Ek Nakit", f"{ek_nakit/1_000_000:,.0f} Mn ₺")

    st.caption("Not: NPL (proxy) ve ek nakit, demo/simülasyon varsayımlarıdır.")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Skor Dağılımı")
        fig = px.histogram(df, x="tahsilat_skoru", nbins=25, title="Tahsilat Skoru Histogramı")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Segmenti Dağılımı")
        seg = df["risk_segmenti"].value_counts().reset_index()
        seg.columns = ["segment", "adet"]
        fig2 = px.bar(seg, x="adet", y="segment", orientation="h", title="Müşteri Risk Segmentleri")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Önceliklendirme ve Günlük Aksiyon Listesi")
    df["oncelik"] = np.where(df["tahsilat_skoru"] >= threshold, "YÜKSEK", "NORMAL")

    day_list = df.sort_values("tahsilat_skoru", ascending=False).head(int(capacity)).copy()
    recs = day_list.apply(lambda r: suggest_channel_time(r["tahsilat_skoru"], r.get("kanal_tercihi", "SMS")), axis=1)
    day_list["onerilen_kanal"] = [k for k, _ in recs]
    day_list["onerilen_zaman"] = [z for _, z in recs]
    day_list["mesaj_sablonu"] = day_list.apply(
        lambda r: followup_template(r.get("musteri_id", "Müşteri"), r["onerilen_kanal"], r["onerilen_zaman"], r["tahsilat_skoru"]),
        axis=1
    )

    cols_show = [
        "musteri_id","tahsilat_skoru","oncelik","risk_segmenti","kanal_tercihi","onerilen_kanal","onerilen_zaman",
        "gecikme_sayisi","toplam_borc","gecikme_gunu","son_odeme_uzerinden_gun","gelir_duzeyi","mesaj_sablonu"
    ]
    present_cols = [c for c in cols_show if c in day_list.columns]
    st.dataframe(day_list[present_cols], use_container_width=True, height=420)

with tab2:
    st.subheader("🧪 Simülasyon: Eşik & Kapasite Etkisi")
    st.write(
        "Bu bölüm, **skor eşiği** ve **günlük kapasite** değiştiğinde aksiyon listesi ve beklenen geri kazanımın "
        "nasıl değişebileceğini hızlıca göstermek içindir."
    )

    # Basit beklenen kazanım: seçilen kişilerin skor ortalaması * borç
    sim = df.copy()
    sim["beklenen_kazanım"] = (sim["tahsilat_skoru"] / 100.0) * sim["toplam_borc"]
    picked = sim.sort_values("tahsilat_skoru", ascending=False).head(int(capacity))

    s1, s2, s3 = st.columns(3)
    s1.metric("Seçilen Kayıt", f"{len(picked):,}")
    s2.metric("Seçilenlerin Ortalama Skoru", f"{picked['tahsilat_skoru'].mean():.1f}")
    s3.metric("Beklenen Tahsilat (proxy)", f"{picked['beklenen_kazanım'].sum():,.0f} ₺")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            picked,
            x="gecikme_gunu",
            y="toplam_borc",
            color="risk_segmenti",
            size="tahsilat_skoru",
            hover_data=["musteri_id", "tahsilat_skoru", "onerilen_kanal"] if "onerilen_kanal" in picked.columns else ["musteri_id", "tahsilat_skoru"],
            title="Seçilen Aksiyon Listesi: Gecikme vs Borç"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(sim, x="risk_segmenti", y="toplam_borc", title="Risk Segmentine Göre Borç Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# "LLM benzeri" yerel asistan (dış API yok)
# -----------------------------
def answer_question(q, df, threshold):
    ql = q.lower()
    try:
        if "toplam alacak" in ql:
            val = df["toplam_borc"].sum()
            return f"Toplam alacak: {val:,.0f} ₺"
        if "kaç müşteri" in ql or "kac musteri" in ql:
            return f"Kayıtlı müşteri sayısı: {df['musteri_id'].nunique()}"
        if "ortalama bor" in ql:
            return f"Ortalama borç: {df['toplam_borc'].mean():,.2f} ₺"
        if "yüksek öncelik" in ql or "yuksek oncelik" in ql:
            pay = (df['tahsilat_skoru'] >= threshold).mean()*100
            return f"Eşik ({threshold}) üzerinde olanların payı: %{pay:.1f}"
        if "risk" in ql and "dağılım" in ql:
            c = df["risk_segmenti"].value_counts()
            return "Risk segmenti dağılımı:\n" + c.to_string()
        if "en yüksek skor" in ql or "top 10" in ql or "ilk 10" in ql:
            topn = df.sort_values("tahsilat_skoru", ascending=False)[["musteri_id","tahsilat_skoru"]].head(10)
            return "En yüksek skorlu ilk 10 müşteri:\n" + topn.to_string(index=False)
        toplam = df["toplam_borc"].sum()
        ort = df["toplam_borc"].mean()
        pay = (df['tahsilat_skoru'] >= threshold).mean()*100
        return (f"Kısa özet: Toplam alacak {toplam:,.0f} ₺, ortalama borç {ort:,.2f} ₺. "
                f"Eşik {threshold} üzerindeki pay %{pay:.1f}.")
    except Exception as e:
        return f"Soruyu işlerken bir hata oluştu: {e}"

with tab3:
    st.subheader("💬 Analitik Asistan (Demo)")
    st.caption("Dış LLM / internet yok. Veriler üzerinde hızlı içgörü üreten yerel kurallı asistan (demo).")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.write(content)

    q = st.chat_input("Veriler hakkında soru sor... (örn: Toplam alacak nedir? Risk dağılımı?)")
    if q:
        st.session_state.chat.append(("user", q))
        a = answer_question(q, df, threshold)
        st.session_state.chat.append(("assistant", a))
        with st.chat_message("assistant"):
            st.write(a)

with tab4:
    st.subheader("📤 Power BI / Excel'e Dışa Aktarım")
    st.caption("Tek tıkla analiz için uygun CSV üretin.")

    df_export = df.copy()
    if "oncelik" not in df_export.columns:
        df_export["oncelik"] = np.where(df_export["tahsilat_skoru"] >= threshold, "YÜKSEK", "NORMAL")

    needed = [
        "musteri_id","tahsilat_skoru","oncelik","risk_segmenti","kanal_tercihi",
        "gecikme_sayisi","toplam_borc","gecikme_gunu","son_odeme_uzerinden_gun","gelir_duzeyi","geri_odeme"
    ]
    for c in needed:
        if c not in df_export.columns:
            df_export[c] = np.nan
    df_export = df_export[needed]

    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("CSV indir", data=csv_bytes, file_name="unioai_export.csv", mime="text/csv")

    with st.expander("Power BI'de içeri aktarma"):
        st.markdown("""
1. Power BI Desktop → **Home** > **Get Data** > **Text/CSV**  
2. `unioai_export.csv` dosyasını seç → **Load**  
3. **Transform Data** ile veri türlerini kontrol et  
4. Önerilen görseller:
   - Card: `toplam_borc` toplamı  
   - Bar: `risk_segmenti` bazında ortalama `tahsilat_skoru`  
   - Pie: `kanal_tercihi` dağılımı  
""")

st.divider()
with st.expander("Sunumdaki mesajla eşleştirme (demo notu)"):
    st.markdown("""
- **Problem:** reaktif raporlama, operasyonel hız ve karar destek eksikliği  
- **Çözüm:** AI tabanlı tahsilat skoru + dinamik önceliklendirme + chatbot paneli  
- **Finansal etki:** kaybedilen portföyde küçük bir iyileşme bile ciddi ek nakit potansiyeli yaratır  
""")
