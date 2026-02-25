"""
Mapa regionalnich kancelari poslancu PSP CR – Streamlit aplikace
Strana poslance se urcuje pres poslanec.id_kandidatka → organy (spolehlivy pristup)
"""

import streamlit as st
import requests
import zipfile
import io
import pandas as pd
import re
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium

# ─── Konfigurace ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kanceláře poslanců PSP ČR",
    page_icon="🏛️",
    layout="wide",
)

PARTY_COLORS = {
    "ANO":        "#1565C0",   # modrá ANO
    "ODS":        "#009FE3",   # světlá modrá ODS
    "STAN":       "#FF8F00",   # jantarová STAN
    "Piráti":     "#231F20",   # černá Piráti
    "KDU-ČSL":    "#F9A825",   # žlutá KDU
    "SPD":        "#C62828",   # tmavě červená SPD
    "Motoristé":  "#E65100",   # oranžová
    "TOP 09":     "#6A1B9A",   # fialová TOP 09
    "Nezařazení": "#757575",   # šedá
}

DATA_URL = "https://www.psp.cz/eknih/cdrom/opendata/poslanci.zip"
HEADERS  = {"User-Agent": "Mozilla/5.0 (PSP-kancelare-mapa/1.0)"}

# ─── Nacitani dat ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Načítám data z psp.cz …")
def load_data():
    try:
        resp = requests.get(DATA_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Nepodařilo se stáhnout data: {e}")
        return pd.DataFrame(), {}

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    def read_unl(name, cols):
        for entry in zf.namelist():
            if entry.lower().endswith(name.lower()):
                raw = zf.read(entry).decode("windows-1250", errors="replace")
                rows = []
                for line in raw.splitlines():
                    if not line.strip():
                        continue
                    parts = line.rstrip("|").split("|")
                    parts += [""] * (len(cols) - len(parts))
                    rows.append(parts[:len(cols)])
                return pd.DataFrame(rows, columns=cols)
        return pd.DataFrame()

    # ── Načti tabulky
    poslanec = read_unl("poslanec.unl", [
        "id_poslanec", "id_osoba", "id_kraj", "id_kandidatka",
        "id_obdobi", "web", "ulice", "obec", "psc",
        "email", "telefon", "fax", "psp_telefon", "facebook", "foto",
    ])
    osoby = read_unl("osoby.unl", [
        "id_osoba", "pred", "prijmeni", "jmeno", "za",
        "narozeni", "pohlavi", "zmena", "umrti",
    ])
    pkgps = read_unl("pkgps.unl", [
        "id_poslanec", "adresa_gps", "sirka", "delka",
    ])
    # pkancelar.unl – samostatná tabulka s kontakty regionálních kanceláří
    pkancelar = read_unl("pkancelar.unl", [
        "id_poslanec", "nazev", "ulice_pk", "obec_pk", "psc_pk",
        "telefon_pk", "fax_pk", "email_pk", "od_pk", "do_pk",
    ])
    # Ponech jen aktivní kanceláře (do_pk prázdné)
    if not pkancelar.empty and "do_pk" in pkancelar.columns:
        pkancelar = pkancelar[pkancelar["do_pk"].str.strip() == ""]
    organy = read_unl("organy.unl", [
        "id_organ", "organ_id_organ", "id_typ_organu", "zkratka",
        "nazev_organu_cz", "nazev_organu_en",
        "od_organ", "do_organ", "priorita", "cl_organ_base",
    ])

    if poslanec.empty or osoby.empty or pkgps.empty:
        st.warning("Nepodařilo se načíst základní soubory (poslanec/osoby/pkgps).")
        return pd.DataFrame(), {}

    # ── Strana přes id_kandidatka → organy
    # Toto je PŘÍMÁ vazba na stranu/hnutí za kterou byl poslanec zvolen.
    # Nejspolehlivější zdroj – nevyžaduje zarazeni ani kluby.
    organy_idx = organy.set_index("id_organ")[["zkratka", "nazev_organu_cz"]].to_dict("index")

    def get_party_info(id_kand):
        k = str(id_kand).strip()
        if k and k in organy_idx:
            return organy_idx[k]["zkratka"], organy_idx[k]["nazev_organu_cz"]
        return "", ""

    poslanec[["zkratka_kand", "nazev_kand"]] = poslanec["id_kandidatka"].apply(
        lambda x: pd.Series(get_party_info(x))
    )

    def normalize_party(z_raw, n_raw):
        z = str(z_raw).strip().upper().replace("-", "").replace(" ", "").replace("Č", "C").replace("Á", "A").replace("Í", "I").replace("Ě", "E")
        n = str(n_raw).strip().lower()

        MAPA = {
            "ANO2011": "ANO", "ANO": "ANO",
            "ODS": "ODS",
            "STAN": "STAN",
            "PIR": "Piráti", "PIRATI": "Piráti", "PIRATIISTAN": "Piráti",
            "KDUDSL": "KDU-ČSL", "KDUDSL": "KDU-ČSL", "KDU": "KDU-ČSL",
            "SPD": "SPD", "SPDSPD": "SPD", "SPDATRIUMF": "SPD",
            "MS": "Motoristé", "MOTORISTE": "Motoristé", "MOTORISTESOBĚ": "Motoristé",
            "TOP09": "TOP 09", "TOP09STAN": "TOP 09",
        }
        if z in MAPA:
            return MAPA[z]

        # Fallback podle obsahu názvu
        if "ano 2011" in n or ("hnut" in n and "ano" in n):  return "ANO"
        if "občanská demokratická" in n:                      return "ODS"
        if "starostov" in n:                                  return "STAN"
        if "pirát" in n:                                      return "Piráti"
        if "křesťansk" in n or "lidov" in n or "kdu" in n:   return "KDU-ČSL"
        if "svoboda a přímá" in n:                            return "SPD"
        if "motorist" in n:                                   return "Motoristé"
        if "top" in n and "09" in n:                          return "TOP 09"

        raw = str(z_raw).strip()
        return raw if raw else "Nezařazení"

    poslanec["strana"] = poslanec.apply(
        lambda r: normalize_party(r["zkratka_kand"], r["nazev_kand"]), axis=1
    )

    # ── Jméno
    osoby["jmeno_plne"] = (
        osoby["pred"].str.strip() + " " +
        osoby["jmeno"].str.strip() + " " +
        osoby["prijmeni"].str.strip() + " " +
        osoby["za"].str.strip()
    ).str.strip().str.replace(r"\s+", " ", regex=True)

    # PSP email format: prijmeni@psp.cz (bez diakritiky, lowercase)
    def make_psp_email(prijmeni):
        import unicodedata
        p = str(prijmeni).strip().lower()
        # Odstraň diakritiku přes unicodedata (spolehlivé, bez maketrans)
        p = unicodedata.normalize("NFD", p)
        p = "".join(c for c in p if unicodedata.category(c) != "Mn")
        # Odstraň vše kromě písmen a pomlčky
        p = re.sub(r"[^a-z-]", "", p)
        return f"{p}@psp.cz" if p else ""

    osoby["psp_email"] = osoby["prijmeni"].apply(make_psp_email)

    # ── Merge poslanec + osoba
    df = poslanec.merge(osoby[["id_osoba", "jmeno_plne", "psp_email"]], on="id_osoba", how="left")

    # ── GPS dekódování (formát PSP: GG.AABBCCC = stupně.minutyvteřinytisíciny)
    def parse_gps(val):
        val = str(val).strip()
        if not val:
            return float("nan")
        try:
            f = float(val)
            if 48.0 <= f <= 51.5 or 12.0 <= f <= 18.9:
                return f
            if "." in val:
                deg_s, frac_s = val.split(".", 1)
            else:
                deg_s, frac_s = val, "0000000"
            deg  = int(deg_s)
            frac = frac_s.ljust(7, "0")
            mins = int(frac[0:2])
            secs = int(frac[2:4])
            msec = int(frac[4:7])
            return deg + mins / 60.0 + secs / 3600.0 + msec / 3600000.0
        except Exception:
            return float("nan")

    pkgps["lat"] = pkgps["sirka"].apply(parse_gps)
    pkgps["lon"] = pkgps["delka"].apply(parse_gps)
    pkgps = pkgps.dropna(subset=["lat", "lon"])
    pkgps = pkgps[
        (pkgps["lat"] >= 48.5) & (pkgps["lat"] <= 51.2) &
        (pkgps["lon"] >= 12.1) & (pkgps["lon"] <= 18.9)
    ]

    # ── Finální join: GPS + info o poslanci
    result = pkgps.merge(
        df[["id_poslanec", "jmeno_plne", "psp_email", "strana", "zkratka_kand", "nazev_kand",
            "ulice", "obec", "psc", "email", "telefon", "psp_telefon"]],
        on="id_poslanec", how="left"
    )

    # Přidej kontakty z pkancelar.unl (specifické pro kancelář, spolehlivější)
    if not pkancelar.empty:
        pk_kontakt = pkancelar[["id_poslanec", "telefon_pk", "email_pk",
                                 "ulice_pk", "obec_pk", "psc_pk"]].copy()
        # Pokud má poslanec více aktivních kanceláří, vezmi první
        pk_kontakt = pk_kontakt.drop_duplicates("id_poslanec")
        result = result.merge(pk_kontakt, on="id_poslanec", how="left")
    else:
        result["telefon_pk"] = ""
        result["email_pk"]   = ""
        result["ulice_pk"]   = ""
        result["obec_pk"]    = ""
        result["psc_pk"]     = ""

    result["barva"] = result["strana"].map(PARTY_COLORS).fillna("#9E9E9E")

    # Adresa: pkgps.adresa_gps → pkancelar → poslanec.ulice/obec/psc
    def build_adresa(row):
        # 1. GPS adresa (nejspolehlivější – přímo z tabulky GPS kanceláří)
        gps_addr = str(row.get("adresa_gps", "")).strip()
        if gps_addr and gps_addr not in ("", "nan"):
            cleaned = re.sub(r"\s*;\s*", ", ", gps_addr).strip(", ").strip()
            if cleaned:
                return cleaned
        # 2. pkancelar tabulka
        ul = str(row.get("ulice_pk", "")).strip()
        ps = str(row.get("psc_pk",   "")).strip()
        ob = str(row.get("obec_pk",  "")).strip()
        if ul or ob:
            parts = [p for p in [ul, (ps + " " + ob).strip()] if p]
            return ", ".join(parts)
        # 3. poslanec.unl
        ul = str(row.get("ulice", "")).strip()
        ps = str(row.get("psc",   "")).strip()
        ob = str(row.get("obec",  "")).strip()
        parts = [p for p in [ul, (ps + " " + ob).strip()] if p]
        return ", ".join(parts)

    result["adresa"] = result.apply(build_adresa, axis=1)

    # Telefon: pkancelar → poslanec.telefon → poslanec.psp_telefon
    def build_telefon(row):
        t = str(row.get("telefon_pk", "")).strip()
        if t and t not in ("", "nan"):
            return t
        t = str(row.get("telefon", "")).strip()
        if t and t not in ("", "nan"):
            return t
        t = str(row.get("psp_telefon", "")).strip()
        return t if t not in ("nan",) else ""

    # Email: pkancelar → poslanec.email → psp_email (prijmeni@psp.cz)
    def build_email(row):
        e = str(row.get("email_pk", "")).strip()
        if e and e not in ("", "nan", "posta@psp.cz"):
            return e
        e = str(row.get("email", "")).strip()
        if e and e not in ("", "nan", "posta@psp.cz"):
            return e
        # Záloha: konstruovaný PSP email
        e = str(row.get("psp_email", "")).strip()
        return e if e not in ("", "nan") else ""

    result["telefon_k"] = result.apply(build_telefon, axis=1)
    result["email_k"]   = result.apply(build_email, axis=1)

    # Přidej web z poslanec pro případ že telefon chybí
    # Zkontroluj kolik záznamů má kontakty
    n_tel   = (result["telefon_k"].str.len() > 0).sum()
    n_email = (result["email_k"].str.len() > 0).sum()

    debug = {
        "zkratky":  sorted(result["zkratka_kand"].dropna().unique().tolist()),
        "nazvy":    sorted(result["nazev_kand"].dropna().unique().tolist()),
        "strany":   sorted(result["strana"].unique().tolist()),
        "radku":    len(result),
        "s_telefonem": int(n_tel),
        "s_emailem":   int(n_email),
        "ukazka_kontaktu": result[["jmeno_plne","telefon_k","email_k","adresa"]].head(5).to_dict("records"),
    }

    return result.reset_index(drop=True), debug


# ─── Vzdálenosti ──────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))


@st.cache_data(ttl=86400 * 30, show_spinner=False)
def load_psc_db() -> dict:
    """
    Stáhne databázi PSČ → GPS z GeoNames (CZ.zip, ~200 KB, volně dostupné).
    Vrátí slovník {psc_bez_mezery: (lat, lon)}.
    """
    url = "https://download.geonames.org/export/zip/CZ.zip"
    db = {}
    try:
        r = requests.get(url, headers={"User-Agent": "PSP-kancelare-mapa/1.0"}, timeout=15)
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        # Soubor uvnitř se jmenuje CZ.txt, TSV formát
        with zf.open("CZ.txt") as f:
            for line in f:
                parts = line.decode("utf-8").strip().split("\t")
                if len(parts) >= 11:
                    psc  = parts[1].replace(" ", "")
                    try:
                        lat = float(parts[9])
                        lon = float(parts[10])
                        db[psc] = (lat, lon)
                    except ValueError:
                        pass
    except Exception:
        pass
    return db


def psc_to_coords(psc: str):
    psc_clean = re.sub(r"\s", "", psc).strip()
    db = load_psc_db()
    if psc_clean in db:
        return db[psc_clean]
    # Fallback: zkus první 3 cifry (přibližná poloha okresu)
    if len(psc_clean) >= 3:
        prefix = psc_clean[:3]
        matches = [(k, v) for k, v in db.items() if k.startswith(prefix)]
        if matches:
            lats = [v[0] for _, v in matches]
            lons = [v[1] for _, v in matches]
            return sum(lats)/len(lats), sum(lons)/len(lons)
    return None, None



# ─── HLAVNÍ UI ────────────────────────────────────────────────────────────────

st.title("🏛️ Mapa kanceláří poslanců PSP ČR")
st.caption("Data: [psp.cz](https://www.psp.cz/sqw/hp.sqw?k=1300) · aktualizováno každou hodinu")

df_all, debug_info = load_data()

if df_all.empty:
    st.error("Data se nepodařilo načíst. Zkontrolujte připojení k internetu.")
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔍 Filtry")

    st.subheader("📮 Vzdálenost od PSČ")
    psc_input = st.text_input("Vaše PSČ (např. 602 00)", placeholder="PSČ")
    max_km    = st.slider("Maximální vzdálenost (km)", 5, 300, 50, 5)
    st.divider()
    st.subheader("🏳️ Stranická příslušnost")

    strany_dostupne = sorted(df_all["strana"].unique())
    strany_vybrane  = []

    for strana in strany_dostupne:
        barva = PARTY_COLORS.get(strana, "#9E9E9E")
        pocet = int((df_all["strana"] == strana).sum())
        col_cb, col_label = st.columns([1, 6])
        with col_cb:
            checked = st.checkbox("", value=True, key=f"cb_{strana}", label_visibility="collapsed")
        with col_label:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px">'
                f'<span style="width:14px;height:14px;border-radius:50%;background:{barva};'
                f'display:inline-block;flex-shrink:0"></span>'
                f'<span style="font-weight:600;font-size:14px">{strana}</span>'
                f'<span style="color:#888;font-size:12px">({pocet})</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if checked:
            strany_vybrane.append(strana)

    st.divider()
    st.caption(f"Celkem kanceláří: **{len(df_all)}**")

# ─── Filtrování ───────────────────────────────────────────────────────────────

df = df_all[df_all["strana"].isin(strany_vybrane)].copy()

user_lat, user_lon = None, None
if psc_input.strip():
    with st.spinner("Hledám souřadnice PSČ …"):
        user_lat, user_lon = psc_to_coords(psc_input)
    if user_lat:
        df["vzdalenost_km"] = df.apply(
            lambda r: haversine(user_lat, user_lon, r["lat"], r["lon"]), axis=1
        )
        df["vzdal_typ"] = "✈️"
        df = df[df["vzdalenost_km"] <= max_km].copy()
        st.info(f"📍 PSČ **{psc_input.strip()}** · zobrazuji **{len(df)}** kanceláří do **{max_km} km** ✈️")
    else:
        st.warning("PSČ nenalezeno nebo chyba geocódování.")

# ─── MAPA ─────────────────────────────────────────────────────────────────────

if df.empty:
    center_lat, center_lon, zoom = 49.75, 15.5, 7
elif user_lat:
    center_lat, center_lon, zoom = user_lat, user_lon, 9
else:
    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())
    zoom = 7

m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")

# Marker PSČ uživatele
if user_lat:
    folium.Marker(
        location=[user_lat, user_lon],
        popup=folium.Popup(f"<b>Vaše PSČ: {psc_input.strip()}</b>", max_width=160),
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)
    folium.Circle(
        location=[user_lat, user_lon],
        radius=max_km * 1000,
        color="#e53935",
        fill=False,
        weight=1.5,
        dash_array="6 4",
    ).add_to(m)

# Kancelářské tečky
for _, row in df.iterrows():
    barva  = row.get("barva",      "#9E9E9E")
    strana = row.get("strana",     "—")
    jmeno  = row.get("jmeno_plne", "—")
    adresa = row.get("adresa",     "")
    tel    = row.get("telefon_k",  "")
    email  = row.get("email_k",    "")
    vzdal  = row.get("vzdalenost_km", None)

    tel_row   = (f'<tr><td style="padding-right:8px;color:#555;vertical-align:top">📞</td>'
                 f'<td><a href="tel:{tel}" style="color:#1565C0">{tel}</a></td></tr>') if tel else ""
    email_row = (f'<tr><td style="color:#555;vertical-align:top">✉️</td>'
                 f'<td><a href="mailto:{email}" style="color:#1565C0;word-break:break-all">{email}</a></td></tr>') if email else ""
    vzdal_row = (f'<tr><td colspan="2" style="padding-top:8px;color:#777;font-size:11px">'
                 f'✈️ {vzdal:.1f} km od zadaného PSČ</td></tr>') if vzdal is not None else ""
    adresa_row = (f'<tr><td style="padding-right:8px;color:#555;vertical-align:top">📍</td>'
                  f'<td>{adresa}</td></tr>') if adresa else ""

    popup_html = f"""
    <div style="font-family:system-ui,sans-serif;font-size:13px;min-width:240px;max-width:300px">
      <div style="font-size:15px;font-weight:700;margin-bottom:6px;color:#111">{jmeno}</div>
      <div style="margin-bottom:10px">
        <span style="background:{barva};color:white;padding:3px 10px;border-radius:12px;
                     font-size:11px;font-weight:700;letter-spacing:.4px">{strana}</span>
      </div>
      <table style="border-collapse:collapse;width:100%;line-height:1.7;color:#333">
        {adresa_row}
        {tel_row}
        {email_row}
        {vzdal_row}
      </table>
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=9,
        color="white",
        weight=1.5,
        fill=True,
        fill_color=barva,
        fill_opacity=0.92,
        popup=folium.Popup(popup_html, max_width=310),
        tooltip=folium.Tooltip(f"<b>{jmeno}</b><br><i>{strana}</i>", sticky=True),
    ).add_to(m)

# Legenda na mapě
strany_na_mape = sorted(df["strana"].unique()) if not df.empty else []
if strany_na_mape:
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0">'
        f'<div style="width:12px;height:12px;border-radius:50%;'
        f'background:{PARTY_COLORS.get(s,"#9E9E9E")};flex-shrink:0"></div>'
        f'<span style="font-size:12px;color:#111">{s}</span></div>'
        for s in strany_na_mape
    )
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:30px;left:10px;z-index:9999;
         background:rgba(255,255,255,0.95);padding:10px 14px;border-radius:10px;
         box-shadow:0 2px 10px rgba(0,0,0,0.18);font-family:system-ui,sans-serif">
      <div style="font-weight:700;font-size:12px;margin-bottom:6px;color:#111">Stranická příslušnost</div>
      {items}
    </div>
    """))

# ─── Layout: mapa nahoře, tabulka pod ní ─────────────────────────────────────

st_folium(m, width=None, height=630, returned_objects=[])

st.subheader(f"📋 Kanceláře ({len(df)})")

if not df.empty:
    tabulka = df[["jmeno_plne", "strana", "adresa", "email_k"]].copy()
    tabulka.columns = ["Poslanec", "Strana", "Adresa kanceláře", "E-mail"]
    if "vzdalenost_km" in df.columns:
        tabulka.insert(2, "km", df["vzdalenost_km"].round(1))
        tabulka.insert(3, "typ", df.get("vzdal_typ", "✈️"))

    st.dataframe(
        tabulka,
        use_container_width=True,
        height=400,
        hide_index=True,
        column_config={
            "Strana":           st.column_config.TextColumn("Strana",  width="small"),
            "km":               st.column_config.NumberColumn("km",    format="%.1f km", width="small"),
            "Adresa kanceláře": st.column_config.TextColumn("Adresa",  width="large"),
            "E-mail":           st.column_config.TextColumn("E-mail",  width="medium"),
        }
    )
else:
    st.info("Žádné kanceláře neodpovídají filtrům.")
