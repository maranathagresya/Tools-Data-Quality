import streamlit as st
import pandas as pd
import numpy as np
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="General Data Quality Checker",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "Upload & Preview"

# ===============================
# LOAD CSS
# ===============================
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css tidak ditemukan")

load_css()

# ===============================
# SIDEBAR NAVIGATION
# ===============================
PAGES = [
    "Upload & Preview",
    "Data Cleaning",
    "Ringkasan Skor"
]

st.sidebar.markdown("## 🔍 General Data Quality Checker")
st.sidebar.caption("Cek & bersihkan duplikasi dan missing value untuk data apapun")

st.session_state.page = st.sidebar.radio(
    "Navigation", PAGES, index=PAGES.index(st.session_state.page)
)

st.sidebar.divider()
st.sidebar.caption(
    "Baris yang **identik persis di semua kolom** akan dihapus otomatis. "
    "Baris yang hanya sama di **kolom kunci** (mis. ID) akan ditandai, tidak dihapus otomatis. "
    "Nilai kosong dinormalisasi jadi **NaN**, tidak pernah diisi otomatis — verifikasi tetap oleh perusahaan."
)

# ===============================
# STEPPER (reuse .stepper dari style.css)
# ===============================
def render_stepper(pages, current_page):
    current_idx = pages.index(current_page)
    items_html = ""
    for i, p in enumerate(pages):
        state = "done" if i < current_idx else ("active" if i == current_idx else "todo")
        items_html += f'<div class="step {state}"><span class="dot">{i + 1}</span><span class="label">{p}</span></div>'
        if i < len(pages) - 1:
            connector_state = "done" if i < current_idx else "todo"
            items_html += f'<div class="connector {connector_state}"></div>'
    st.markdown(f'<div class="stepper">{items_html}</div>', unsafe_allow_html=True)

render_stepper(PAGES, st.session_state.page)

# ===============================
# HELPER: KARTU STATISTIK & BANNER
# ===============================
def render_stat_cards(cards):
    html = '<div class="stat-card-row">'
    for c in cards:
        tone = c.get("tone", "neutral")
        html += (
            f'<div class="stat-card tone-{tone}">'
            f'<div class="stat-card-value">{c["value"]}</div>'
            f'<div class="stat-card-label">{c["label"]}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_wide_stat_card(segments, tone="neutral"):
    """
    Kartu lebar berisi beberapa segmen berdampingan (dipisah garis vertikal),
    dipakai buat ngegabungin beberapa metrik yang saling berhubungan
    (mis. baris null, baris duplikat, total baris tidak ideal) jadi satu kartu.
    Tiap segmen: {"value": ..., "sub_value": "(opsional, ditampilkan kecil di sebelah value)", "label": ...}
    """
    html = f'<div class="stat-card-row"><div class="stat-card stat-card-wide tone-{tone}">'
    for i, seg in enumerate(segments):
        if i > 0:
            html += '<div class="stat-card-divider"></div>'
        sub = f' <span class="stat-card-subvalue">{seg["sub_value"]}</span>' if seg.get("sub_value") else ""
        html += (
            '<div class="stat-card-segment">'
            f'<div class="stat-card-value">{seg["value"]}{sub}</div>'
            f'<div class="stat-card-label">{seg["label"]}</div>'
            '</div>'
        )
    html += "</div></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_info_banner(text, tone="info"):
    st.markdown(f'<div class="info-banner tone-{tone}">{text}</div>', unsafe_allow_html=True)


# ===============================
# HELPER: NILAI KOSONG "TERSAMAR" (PLACEHOLDER)
# ===============================
PLACEHOLDER_MISSING_TOKENS = {
    "-", "--", "na", "n/a", "n.a", "n.a.", "nan", "none", "null",
    "kosong", "tidak ada", "tdk ada", "tidak diisi", "belum diisi",
    "unknown", "unk", "?", "-,-", ".", "..", "...", "tba", "tbd",
}

# Nilai kosong ASLI (sudah kosong dari sumbernya) tetap ditampilkan sebagai kosong/NaN.
# Sel berisi teks "placeholder" (mis. "-", "N/A", "tidak ada") TIDAK diisi NaN begitu saja,
# tapi ditandai beda supaya perusahaan tahu ini butuh verifikasi, bukan kosong murni.
PLACEHOLDER_MARKER = "🔎 Placeholder (dianggap kosong)"


def is_text_column(series):
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.StringDtype)


def normalize_missing_placeholders(df):
    """
    Mendeteksi sel berisi teks placeholder (mis. '-', 'N/A', 'tidak ada').
    Nilai sel TIDAK diubah jadi NaN di sini — hanya mask booleannya yang dikembalikan,
    supaya nanti bisa ditampilkan dengan tanda berbeda dari kosong asli (NaN murni).
    """
    df = df.copy()
    placeholder_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    total_changed = 0
    for col in df.columns:
        if not is_text_column(df[col]):
            continue
        s = df[col]
        is_str = s.notna() & s.map(lambda v: isinstance(v, str))
        normalized_check = s.where(~is_str, s.astype(str).str.strip().str.lower())
        col_mask = is_str & normalized_check.isin(PLACEHOLDER_MISSING_TOKENS)
        changed = int(col_mask.sum())
        if changed:
            placeholder_mask[col] = col_mask
            total_changed += changed
    return df, total_changed, placeholder_mask


# =========================================================
# STEP 1 — UPLOAD & PREVIEW
# =========================================================
if st.session_state.page == "Upload & Preview":

    st.subheader("Step 1 — Upload & Preview Data")

    if "uploader_key_counter" not in st.session_state:
        st.session_state["uploader_key_counter"] = 0

    has_existing_data = "raw_data" in st.session_state

    if has_existing_data:
        active_name = st.session_state.get("uploaded_filename", "file sebelumnya")
        info_col, btn_col = st.columns([4, 1])
        with info_col:
            st.success(f"Data aktif saat ini: **{active_name}**")
        with btn_col:
            if st.button("Ganti Data", use_container_width=True):
                for key in [
                    "raw_data", "uploaded_filename", "uploaded_file_signature",
                    "clean_data", "total_missing", "total_cells",
                    "removed_dup_count", "dup_key_count", "key_cols",
                    "key_cols_select",
                    "excel_file_id", "excel_sheet_names", "excel_selected_sheet"
                ]:
                    st.session_state.pop(key, None)
                st.session_state["uploader_key_counter"] += 1
                st.rerun()
        uploader_container = st.expander("⬆️ Upload file lain (opsional, akan menggantikan data aktif)")
    else:
        uploader_container = st.container()

    with uploader_container:
        uploaded_file = st.file_uploader(
            "Upload file CSV atau Excel",
            type=["csv", "xlsx"],
            key=f"file_uploader_{st.session_state['uploader_key_counter']}"
        )

    if uploaded_file is not None:
        is_excel = not uploaded_file.name.lower().endswith(".csv")
        selected_sheet = None

        # ---------- PILIH SHEET (KHUSUS FILE EXCEL) ----------
        if is_excel:
            file_id = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("excel_file_id") != file_id:
                try:
                    uploaded_file.seek(0)
                    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
                    st.session_state["excel_sheet_names"] = xls.sheet_names
                    st.session_state["excel_file_id"] = file_id
                    st.session_state["excel_selected_sheet"] = xls.sheet_names[0]
                except Exception as e:
                    st.error("❌ File Excel gagal dibaca")
                    st.code(str(e))
                    st.stop()

            sheet_names = st.session_state["excel_sheet_names"]
            default_sheet = st.session_state.get("excel_selected_sheet", sheet_names[0])
            default_index = sheet_names.index(default_sheet) if default_sheet in sheet_names else 0

            selected_sheet = st.selectbox(
                "📑 Pilih Sheet Excel yang Ingin Digunakan",
                options=sheet_names,
                index=default_index,
                key="excel_sheet_selectbox",
                help="File Excel ini punya lebih dari satu sheet. Pilih sheet mana yang datanya ingin dianalisis."
            )
            st.session_state["excel_selected_sheet"] = selected_sheet

            if len(sheet_names) > 1:
                st.caption(f"File ini punya **{len(sheet_names)} sheet**: {', '.join(sheet_names)}")

        # ---------- BACA DATA ----------
        file_signature = (
            (uploaded_file.name, uploaded_file.size, selected_sheet)
            if is_excel else
            (uploaded_file.name, uploaded_file.size)
        )

        if st.session_state.get("uploaded_file_signature") != file_signature:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    try:
                        df_new = pd.read_csv(uploaded_file)
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df_new = pd.read_csv(uploaded_file, encoding="latin1")
                else:
                    uploaded_file.seek(0)
                    df_new = pd.read_excel(uploaded_file, sheet_name=selected_sheet, engine="openpyxl")

                st.session_state["raw_data"] = df_new
                st.session_state["uploaded_filename"] = (
                    f"{uploaded_file.name} (Sheet: {selected_sheet})" if is_excel else uploaded_file.name
                )
                st.session_state["uploaded_file_signature"] = file_signature
                for key in ["clean_data", "total_missing", "total_cells",
                            "removed_dup_count", "dup_key_count", "key_cols",
                            "key_cols_select"]:
                    st.session_state.pop(key, None)
                st.rerun()

            except Exception as e:
                st.error("❌ File gagal dibaca")
                st.code(str(e))
                st.stop()

    if "raw_data" not in st.session_state:
        st.info("Silakan upload file untuk memulai pengecekan data.")
        st.stop()

    df = st.session_state["raw_data"]

    st.markdown("### 👀 Preview Data")
    st.dataframe(df.head(20), use_container_width=True)

    _, placeholder_count, placeholder_mask_preview = normalize_missing_placeholders(df)
    total_missing_preview = int(df.isnull().sum().sum()) + placeholder_count
    full_row_dup_preview = int(df.duplicated().sum())
    rows_with_missing_mask = df.isnull().any(axis=1) | placeholder_mask_preview.any(axis=1)
    rows_with_missing_preview = int(rows_with_missing_mask.sum())
    full_row_dup_mask = df.duplicated(keep="first")
    # Baris "tidak ideal" = gabungan (union) baris yang punya nilai kosong/placeholder ATAU
    # baris yang jadi bagian dari duplikat persis — dihitung sekali aja kalau kena keduanya,
    # supaya gak dobel-hitung.
    rows_not_ideal_preview = int((rows_with_missing_mask | full_row_dup_mask).sum())

    render_stat_cards([
        {"value": df.shape[0], "label": "Jumlah Baris", "tone": "neutral"},
        {"value": df.shape[1], "label": "Jumlah Kolom", "tone": "neutral"},
    ])

    render_wide_stat_card(
        [
            {
                "value": rows_with_missing_preview,
                "sub_value": f"({total_missing_preview} sel)" if total_missing_preview else None,
                "label": "Baris Mengandung Nilai Kosong",
            },
            {
                "value": full_row_dup_preview,
                "label": "Baris Duplikat Persis",
            },
            {
                "value": rows_not_ideal_preview,
                "label": "Total Baris Tidak Ideal",
            },
        ],
        tone="danger" if rows_not_ideal_preview else "success"
    )

    if placeholder_count > 0:
        render_info_banner(
            f"🔎 Ditemukan <b>{placeholder_count} sel</b> berisi teks placeholder "
            f"(mis. \"-\", \"N/A\", \"tidak ada\") yang kemungkinan berarti kosong. "
            f"Sel ini <b>TIDAK</b> akan diisi NaN begitu saja — akan ditandai terpisah di Step 2 "
            f"(Data Cleaning) supaya beda dengan sel yang memang kosong sejak awal.",
            tone="info"
        )

    st.markdown("### Struktur Kolom")
    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str).values,
        "Kosong Asli (NaN)": df.isnull().sum().values,
        "Placeholder (mis. '-', 'N/A')": placeholder_mask_preview.sum().values,
        "Unique": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.divider()
    if st.button("➡️ Lanjut ke Data Cleaning"):
        st.session_state.page = "Data Cleaning"
        st.rerun()


# =========================================================
# STEP 2 — DATA CLEANING
# =========================================================
elif st.session_state.page == "Data Cleaning":
    st.subheader("Step 2 — Data Cleaning")

    if "raw_data" not in st.session_state:
        st.warning("Upload data terlebih dahulu (Step 1)")
        st.stop()

    df = st.session_state["raw_data"].copy()
    df_clean = df.copy()
    cleaning_log = []

    # ------------------------------------------------------
    # 1) DETEKSI PLACEHOLDER KOSONG (TIDAK DIUBAH JADI NaN)
    # ------------------------------------------------------
    df_clean, placeholder_changed, placeholder_mask = normalize_missing_placeholders(df_clean)
    if placeholder_changed > 0:
        cleaning_log.append(
            f"{placeholder_changed} sel berisi teks placeholder (mis. '-', 'N/A', 'tidak ada') ditandai "
            "terpisah sebagai placeholder — nilainya TIDAK diubah jadi NaN, supaya beda dengan sel yang "
            "memang kosong sejak awal"
        )

    # ------------------------------------------------------
    # 2) HAPUS DUPLIKAT PERSIS (SEMUA KOLOM SAMA) + CATAT BARIS YANG DIHAPUS
    # ------------------------------------------------------
    before_rows = df_clean.shape[0]
    df_before_exact_dedup = df_clean.copy() 
    exact_dup_mask = df_clean.duplicated(keep="first")
    removed_rows_df = df_clean[exact_dup_mask].copy()
    df_clean = df_clean[~exact_dup_mask].copy()
    placeholder_mask = placeholder_mask.loc[df_clean.index]
    after_rows = df_clean.shape[0]
    removed_dup_count = before_rows - after_rows
    if removed_dup_count > 0:
        cleaning_log.append(
            f"{removed_dup_count} baris duplikat (identik di semua kolom) dihapus otomatis — "
            "daftar baris yang dihapus bisa dilihat di bawah"
        )

    # ------------------------------------------------------
    # 3) TANDAI DUPLIKAT BERDASARKAN KOLOM KUNCI (OPSIONAL, TIDAK DIHAPUS)
    # ------------------------------------------------------
    st.markdown("### 🔑 Pilih Kolom Kunci (opsional)")
    st.caption(
        "Kolom kunci = kolom yang seharusnya unik per baris (mis. ID, no. invoice, email). "
        "Baris yang sama pada kolom ini akan DITANDAI, bukan dihapus otomatis, karena datanya bisa saja "
        "sama ID tapi berbeda kolom lain (perlu ditinjau manual)."
    )
    key_cols = st.multiselect(
        "Pilih satu atau beberapa kolom kunci",
        options=df_clean.columns.tolist(),
        key="key_cols_select"
    )

    dup_key_count = 0
    dup_key_mask = pd.Series(False, index=df_clean.index)
    if key_cols:
        dup_key_mask = df_clean.duplicated(subset=key_cols, keep=False)
        dup_key_count = int(dup_key_mask.sum())
        if dup_key_count > 0:
            cleaning_log.append(
                f"Ditemukan {dup_key_count} baris duplikat berdasarkan kolom kunci "
                f"({', '.join(key_cols)}) — perlu ditinjau manual, tidak dihapus otomatis"
            )
        df_clean.insert(
            0, "catatan_duplikat",
            np.where(dup_key_mask, "🔁 Duplikat Kolom Kunci", "")
        )

    # ------------------------------------------------------
    # RINGKASAN HASIL CLEANING
    # ------------------------------------------------------
    st.success("✅ Data dibersihkan: duplikat persis dihapus, placeholder kosong ditandai terpisah — tanpa mengisi nilai kosong secara otomatis")

    render_stat_cards([
        {"value": before_rows, "label": "Total Baris", "tone": "neutral"},
        {"value": after_rows, "label": "Baris Setelah Duplikat", "tone": "neutral"},
        {"value": removed_dup_count, "label": "Duplikat Persis Dihapus",
         "tone": "success" if removed_dup_count else "neutral"},
        {"value": dup_key_count, "label": "Duplikat Kolom Kunci Ditandai",
         "tone": "warning" if dup_key_count else "success"},
    ])

    st.markdown("### Data Cleaning Log")
    if cleaning_log:
        for log in cleaning_log:
            st.write("•", log)
    else:
        st.write("• Tidak ada perubahan signifikan yang diperlukan")

    # ------------------------------------------------------
    # DETAIL BARIS DUPLIKAT (PERSIS + KOLOM KUNCI) — DIKELOMPOKKAN PER GRUP,
    # LENGKAP DENGAN "DUPLIKAT DENGAN BARIS BERAPA AJA"
    # ------------------------------------------------------
    st.markdown("### 🔁 Detail Baris Duplikat")
    st.caption(
        "Menunjukkan baris mana saja yang terdeteksi duplikat, dikelompokkan per grup — tiap baris yang "
        "punya kembaran ditandai dengan grup yang sama, dan kolom \"Duplikat Dengan Baris\" langsung "
        "nyebutin nomor baris pasangannya. \"No. Baris\" mengacu ke urutan baris data di file yang "
        "diupload (baris 1 = baris data pertama setelah header)."
    )

    def _row_key(row):
        # NaN diganti sentinel biar baris yang sama-sama NaN di kolom itu tetap dianggap 1 grup
        return tuple("__NaN__" if pd.isna(v) else v for v in row)

    dup_detail_parts = []

    # ---- Grup Duplikat Persis (ambil dari data SEBELUM dihapus, biar 2 sisinya kelihatan) ----
    if removed_dup_count > 0:
        full_dup_mask_all = df_before_exact_dedup.duplicated(keep=False)
        dup_rows_all = df_before_exact_dedup[full_dup_mask_all]

        group_map = {}
        for idx, row in dup_rows_all.iterrows():
            key = _row_key(row)
            group_map.setdefault(key, []).append(idx)

        sorted_keys = sorted(group_map.keys(), key=lambda k: min(group_map[k]))
        removed_idx_set = set(removed_rows_df.index)

        exact_detail = dup_rows_all.copy()
        exact_detail.insert(0, "No. Baris", exact_detail.index + 1)
        exact_detail.insert(1, "Jenis Duplikat", "🗑️ Duplikat Persis")
        exact_detail.insert(2, "Grup Duplikat", "")
        exact_detail.insert(3, "Peran", "")
        exact_detail.insert(4, "Duplikat Dengan Baris", "")

        for g_num, key in enumerate(sorted_keys, start=1):
            idxs = group_map[key]
            grup_label = f"Persis #{g_num}"
            for idx in idxs:
                partner_rows = sorted(i + 1 for i in idxs if i != idx)
                exact_detail.loc[idx, "Grup Duplikat"] = grup_label
                exact_detail.loc[idx, "Peran"] = "🗑️ Dihapus" if idx in removed_idx_set else "🟢 Dipertahankan"
                exact_detail.loc[idx, "Duplikat Dengan Baris"] = ", ".join(map(str, partner_rows))

        dup_detail_parts.append(exact_detail)

    # ---- Grup Duplikat Kolom Kunci (semua baris masih ada di df_clean, gak ada yang dihapus) ----
    if dup_key_count > 0:
        dup_rows_key = df_clean.loc[dup_key_mask].drop(columns=["catatan_duplikat"], errors="ignore")

        key_group_map = {}
        for idx, row in dup_rows_key[key_cols].iterrows():
            key = _row_key(row)
            key_group_map.setdefault(key, []).append(idx)

        sorted_key_keys = sorted(key_group_map.keys(), key=lambda k: min(key_group_map[k]))

        key_detail = dup_rows_key.copy()
        key_detail.insert(0, "No. Baris", key_detail.index + 1)
        key_detail.insert(1, "Jenis Duplikat", "🔁 Duplikat Kolom Kunci")
        key_detail.insert(2, "Grup Duplikat", "")
        key_detail.insert(3, "Peran", "🔁 Ditandai (Tidak Dihapus)")
        key_detail.insert(4, "Duplikat Dengan Baris", "")

        for g_num, key in enumerate(sorted_key_keys, start=1):
            idxs = key_group_map[key]
            grup_label = f"Kunci #{g_num}"
            for idx in idxs:
                partner_rows = sorted(i + 1 for i in idxs if i != idx)
                key_detail.loc[idx, "Grup Duplikat"] = grup_label
                key_detail.loc[idx, "Duplikat Dengan Baris"] = ", ".join(map(str, partner_rows))

        dup_detail_parts.append(key_detail)

    if dup_detail_parts:
        dup_detail_df = pd.concat(dup_detail_parts, ignore_index=True).sort_values(
            ["Jenis Duplikat", "Grup Duplikat", "No. Baris"]
        ).reset_index(drop=True)
        st.dataframe(dup_detail_df, use_container_width=True)
        dup_detail_csv = dup_detail_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Detail Baris Duplikat",
            data=dup_detail_csv,
            file_name="detail_baris_duplikat.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ Tidak ditemukan baris duplikat (persis maupun kolom kunci) di data ini.")

    # ------------------------------------------------------
    # SIAPKAN TAMPILAN: PISAHKAN TANDA PLACEHOLDER VS KOSONG ASLI
    # ------------------------------------------------------
    display_df = df_clean.astype(object)
    placeholder_style_mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)
    missing_style_mask = pd.DataFrame(False, index=df_clean.index, columns=df_clean.columns)

    for col in df_clean.columns:
        if col == "catatan_duplikat":
            continue
        is_missing = df_clean[col].isna()
        is_placeholder = placeholder_mask[col] if col in placeholder_mask.columns else pd.Series(False, index=df_clean.index)
        is_pure_missing = is_missing & ~is_placeholder
        if is_placeholder.any():
            display_df.loc[is_placeholder, col] = PLACEHOLDER_MARKER
            placeholder_style_mask[col] = is_placeholder
        if is_pure_missing.any():
            display_df.loc[is_pure_missing, col] = "None"   
            missing_style_mask[col] = is_pure_missing

    # Baris "bersih" = baris yang nggak punya sel kosong asli ATAUPUN placeholder di kolom manapun.
    # Duplikat persis (semua kolom sama) sudah dihapus di langkah sebelumnya, jadi otomatis tidak
    # dihitung lagi di sini. Duplikat kolom kunci sengaja TIDAK ikut memengaruhi skor — cuma ditandai
    # di kolom "catatan_duplikat" untuk ditinjau manual, karena bisa saja itu wajar (bukan salah input).
    clean_rows_mask = ~(missing_style_mask.any(axis=1) | placeholder_style_mask.any(axis=1))
    clean_rows_count = int(clean_rows_mask.sum())

    # ------------------------------------------------------
    # DETAIL BARIS DENGAN NILAI KOSONG — LENGKAP DENGAN NOMOR BARIS
    # (diletakkan tepat di bawah Detail Baris Duplikat biar gak kepisah)
    # ------------------------------------------------------
    st.markdown("### 🕳️ Detail Baris dengan Nilai Kosong")
    st.caption(
        "Menunjukkan baris mana saja (nomor barisnya) yang punya sel kosong asli atau placeholder, "
        "beserta di kolom apa aja kekosongannya — supaya bisa langsung ditelusuri ke baris aslinya."
    )

    data_cols_only = [c for c in df_clean.columns if c != "catatan_duplikat"]
    rows_with_issue_mask = missing_style_mask.any(axis=1) | placeholder_style_mask.any(axis=1)

    if rows_with_issue_mask.any():
        missing_rows_records = []
        for idx in df_clean.index[rows_with_issue_mask]:
            empty_bits = []
            for col in data_cols_only:
                if col in missing_style_mask.columns and missing_style_mask.at[idx, col]:
                    empty_bits.append(f"{col} (kosong asli)")
                elif col in placeholder_style_mask.columns and placeholder_style_mask.at[idx, col]:
                    empty_bits.append(f"{col} (placeholder)")
            missing_rows_records.append({
                "No. Baris": idx + 1,
                "Jumlah Kolom Kosong": len(empty_bits),
                "Kolom yang Kosong": ", ".join(empty_bits),
            })

        missing_rows_df = pd.DataFrame(missing_rows_records).sort_values(
            "Jumlah Kolom Kosong", ascending=False
        ).reset_index(drop=True)

        st.dataframe(missing_rows_df, use_container_width=True)
        missing_rows_csv = missing_rows_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Download Detail Baris Kosong",
            data=missing_rows_csv,
            file_name="detail_baris_kosong.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ Tidak ditemukan baris dengan nilai kosong asli atau placeholder di data ini.")

    st.divider()

    # ------------------------------------------------------
    # PREVIEW + DOWNLOAD
    # ------------------------------------------------------
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown("### Preview Data Setelah Cleaning")
    with header_right:
        csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download Clean Data",
            data=csv_bytes,
            file_name="clean_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    n_cells = display_df.shape[0] * display_df.shape[1]
    MAX_STYLE_CELLS = 50_000  # batas aman biar tabel berwarna tetap ringan & cepat di browser

    if n_cells <= MAX_STYLE_CELLS:
        # naikkan batas render Styler pandas secukupnya sesuai ukuran data ini
        pd.set_option("styler.render.max_elements", n_cells + 5000)

        styler = display_df.style
        has_key_dup = "catatan_duplikat" in df_clean.columns and (df_clean["catatan_duplikat"] != "").any()
        if has_key_dup:
            def highlight_dup_rows(row):
                if df_clean.loc[row.name, "catatan_duplikat"] != "":
                    return ["background-color: #fef9c3"] * len(row)
                return [""] * len(row)
            styler = styler.apply(highlight_dup_rows, axis=1)

        def highlight_missing_cells(data):
            styles = pd.DataFrame("", index=data.index, columns=data.columns)
            for col in placeholder_style_mask.columns:
                if col in styles.columns:
                    styles.loc[placeholder_style_mask[col], col] = "background-color: #dbeafe"
            for col in missing_style_mask.columns:
                if col in styles.columns:
                    styles.loc[missing_style_mask[col], col] = "background-color: #fee2e2"
            return styles

        styler = styler.apply(highlight_missing_cells, axis=None)
        st.dataframe(styler, use_container_width=True)

        legend_bits = []
        if has_key_dup:
            legend_bits.append("🟡 kuning = duplikat kolom kunci")
        legend_bits.append('🔴 merah + tulisan "None" = kosong asli sejak sumber data')
        legend_bits.append(f'🔵 biru + tanda "{PLACEHOLDER_MARKER}" = teks placeholder yang dianggap kosong')
        st.caption(" · ".join(legend_bits))
    else:
        st.info(
            f"ℹ️ Data ini punya **{n_cells:,} sel** ({display_df.shape[0]:,} baris × {display_df.shape[1]} kolom) — "
            "terlalu besar untuk diwarnai otomatis tanpa membuat browser berat, jadi tabel di bawah ditampilkan "
            "**tanpa highlight warna**. Perhitungan jumlah kosong, placeholder, duplikat, dan skor akurasi di "
            "Step 3 tetap dihitung dari SELURUH data seperti biasa, tidak terpengaruh oleh ini."
        )
        st.dataframe(display_df, use_container_width=True)

    st.divider()

    # ------------------------------------------------------
    # MISSING VALUE PER KOLOM (berdasarkan data hasil cleaning)
    # ------------------------------------------------------
    st.markdown("### 🧩 Missing Value per Kolom (Setelah Cleaning)")

    missing_df = pd.DataFrame({
        "Kolom": [c for c in df_clean.columns if c != "catatan_duplikat"],
    })
    missing_df["Tipe Data"] = [str(df_clean[c].dtype) for c in missing_df["Kolom"]]
    missing_df["Kosong Asli (NaN)"] = [int(missing_style_mask[c].sum()) for c in missing_df["Kolom"]]
    missing_df["Placeholder"] = [int(placeholder_style_mask[c].sum()) for c in missing_df["Kolom"]]
    missing_df["Total Perlu Ditinjau"] = missing_df["Kosong Asli (NaN)"] + missing_df["Placeholder"]
    missing_df["Persentase (%)"] = (missing_df["Total Perlu Ditinjau"] / max(after_rows, 1) * 100).round(2)
    missing_df = missing_df.sort_values("Total Perlu Ditinjau", ascending=False).reset_index(drop=True)

    def highlight_missing(row):
        pct = row["Persentase (%)"]
        if pct == 0:
            color = "background-color: #dcfce7"
        elif pct <= 30:
            color = "background-color: #fef3c7"
        else:
            color = "background-color: #fee2e2"
        return [color] * len(row)

    st.dataframe(missing_df.style.apply(highlight_missing, axis=1), use_container_width=True)

    total_missing = int(missing_df["Total Perlu Ditinjau"].sum())
    non_data_cols = 1 if "catatan_duplikat" in df_clean.columns else 0
    total_cells = df_clean.shape[0] * (df_clean.shape[1] - non_data_cols)

    # ------------------------------------------------------
    # NILAI BERULANG PER KOLOM (SEMUA KOLOM, BUKAN CUMA KOLOM KUNCI)
    # ------------------------------------------------------
    st.divider()
    st.markdown("### 🧾 Ringkasan Nilai Berulang per Kolom")
    st.caption(
        "Cek otomatis untuk SEMUA kolom (mis. Nama, Email, No. HP) — bukan cuma kolom kunci yang dipilih di atas — "
        "supaya perusahaan tahu kolom mana saja yang punya nilai sama berulang dan perlu ditinjau."
    )

    dup_value_rows = []
    check_cols = [c for c in df_clean.columns if c != "catatan_duplikat"]
    for col in check_cols:
        vc = df_clean[col].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        vc = vc.value_counts()
        dup_values = vc[vc > 1]
        rows_involved = int(dup_values.sum()) if not dup_values.empty else 0
        contoh = ", ".join(str(v) for v in dup_values.index[:3]) if not dup_values.empty else "-"
        dup_value_rows.append({
            "Kolom": col,
            "Jumlah Nilai Unik": int(df_clean[col].nunique()),
            "Baris dengan Nilai Berulang": rows_involved,
            "Contoh Nilai yang Berulang": contoh,
        })

    dup_value_df = pd.DataFrame(dup_value_rows).sort_values(
        "Baris dengan Nilai Berulang", ascending=False
    ).reset_index(drop=True)

    def highlight_dup_value(row):
        if row["Baris dengan Nilai Berulang"] > 0:
            return ["background-color: #fef3c7"] * len(row)
        return [""] * len(row)

    st.dataframe(dup_value_df.style.apply(highlight_dup_value, axis=1), use_container_width=True)

    # simpan untuk step berikutnya
    st.session_state["clean_data"] = df_clean
    st.session_state["total_missing"] = total_missing
    st.session_state["total_cells"] = total_cells
    st.session_state["clean_rows_count"] = clean_rows_count
    st.session_state["clean_rows_mask"] = clean_rows_mask
    st.session_state["removed_dup_count"] = removed_dup_count
    st.session_state["dup_key_count"] = dup_key_count
    st.session_state["key_cols"] = key_cols
    st.session_state["removed_rows_df"] = removed_rows_df
    st.session_state["missing_df"] = missing_df
    st.session_state["dup_value_df"] = dup_value_df
    st.session_state["display_df"] = display_df

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Kembali ke Upload & Preview"):
            st.session_state.page = "Upload & Preview"
            st.rerun()
    with col2:
        if st.button("➡️ Lanjut ke Ringkasan Skor"):
            st.session_state.page = "Ringkasan Skor"
            st.rerun()


# =========================================================
# STEP 3 — RINGKASAN SKOR
# =========================================================
elif st.session_state.page == "Ringkasan Skor":
    st.subheader("Step 3 — Ringkasan Skor Akurasi Data")

    if "clean_data" not in st.session_state:
        st.warning("Jalankan Step 2 (Data Cleaning) terlebih dahulu")
        st.stop()

    total_missing = st.session_state["total_missing"]
    total_cells = st.session_state["total_cells"]
    removed_dup_count = st.session_state.get("removed_dup_count", 0)
    dup_key_count = st.session_state.get("dup_key_count", 0)
    key_cols = st.session_state.get("key_cols", [])
    df_clean = st.session_state["clean_data"]
    total_rows = df_clean.shape[0]
    clean_rows_count = st.session_state.get("clean_rows_count", 0)

    # ---------- SKOR ----------
    # Skor Akurasi Data = persentase baris yang "bersih": sudah bebas duplikat
    # persis (dihapus di Step 2) dan tidak punya sel kosong/placeholder sama sekali.
    overall_score = (clean_rows_count / total_rows * 100) if total_rows else 0
    overall_score = round(max(0, overall_score), 1)
    dirty_rows_count = total_rows - clean_rows_count

    render_stat_cards([
        {"value": f"{total_rows:,}".replace(",", "."), "label": "Total Baris Data", "tone": "neutral"},
        {"value": f"{clean_rows_count:,}".replace(",", "."), "label": "Baris Data Bersih", "tone": "success"},
        {"value": f"{dirty_rows_count:,}".replace(",", "."), "label": "Baris Data Tidak Ideal",
         "tone": "danger" if dirty_rows_count else "success"},
        {"value": f"{overall_score}%", "label": "Akurasi Data",
         "tone": "success" if overall_score >= 85 else ("warning" if overall_score >= 70 else "danger")},
    ])

    if overall_score >= 85:
        status = "🟢 Data Sangat Layak Digunakan"
        tone = st.success
    elif overall_score >= 70:
        status = "🟡 Data Cukup Layak, Perlu Perbaikan Ringan"
        tone = st.warning
    else:
        status = "🔴 Data Belum Layak, Perlu Perbaikan Signifikan"
        tone = st.error

    tone(f"Status kesiapan data saat ini: **{status}**")

    st.caption(
        f"Skor dihitung dari **{clean_rows_count} baris bersih** dari total **{total_rows} baris** "
        f"({clean_rows_count} ÷ {total_rows} × 100 = {overall_score}%). "
        "Baris 'bersih' = tidak punya sel kosong asli maupun placeholder di kolom manapun "
        "(duplikat persis sudah dihapus di Step 2, jadi tidak dihitung lagi di sini). "
        "Duplikat berdasarkan kolom kunci sengaja tidak mengurangi skor — cuma ditandai untuk ditinjau manual."
    )

    st.divider()

    # ---------- DATA BERSIH YANG TERMASUK DALAM SKOR ----------
    st.markdown("### 📋 Data Bersih yang Termasuk dalam Skor Akurasi")

    display_df = st.session_state.get("display_df")
    clean_rows_mask = st.session_state.get("clean_rows_mask")

    if display_df is not None and clean_rows_mask is not None:
        clean_only_df = display_df[clean_rows_mask]

        st.caption(
            f"Berikut **{clean_rows_count} baris** yang dihitung sebagai data bersih (tidak ada nilai kosong "
            f"maupun placeholder, dan duplikat persis sudah dihapus) — inilah baris-baris yang membentuk skor "
            f"**{overall_score}%** di atas."
        )

        if clean_rows_count > 0:
            info_left, info_right = st.columns([4, 1])
            with info_right:
                clean_csv_bytes = clean_only_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ Download Data Bersih",
                    data=clean_csv_bytes,
                    file_name="data_bersih.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            st.dataframe(clean_only_df, use_container_width=True)
        else:
            st.info("Tidak ada baris yang sepenuhnya bersih (semua baris punya minimal satu sel kosong/placeholder).")
    else:
        st.info("Jalankan ulang Step 2 (Data Cleaning) untuk menampilkan rincian data bersih ini.")

    st.divider()

    # ---------- TEMUAN ----------
    st.markdown("### Temuan Utama")

    missing_df = st.session_state.get("missing_df")
    dup_value_df = st.session_state.get("dup_value_df")
    removed_rows_df = st.session_state.get("removed_rows_df")

    true_missing_total = int(missing_df["Kosong Asli (NaN)"].sum()) if missing_df is not None else total_missing
    placeholder_total = int(missing_df["Placeholder"].sum()) if missing_df is not None else 0
    cols_with_dup_values = int((dup_value_df["Baris dengan Nilai Berulang"] > 0).sum()) if dup_value_df is not None else 0

    findings = []
    if removed_dup_count > 0:
        findings.append(f"**{removed_dup_count} baris duplikat persis** sudah dihapus otomatis saat Data Cleaning (daftar lengkapnya ada di file laporan).")
    if total_missing > 0:
        findings.append(
            f"Masih ada **{total_missing} sel perlu ditinjau** dari total {total_cells} sel: "
            f"**{true_missing_total} kosong asli** (memang kosong sejak sumber data) dan "
            f"**{placeholder_total} placeholder** (mis. '-', 'N/A', 'tidak ada') yang dianggap kosong."
        )
    if key_cols and dup_key_count > 0:
        findings.append(
            f"Terdapat **{dup_key_count} baris duplikat berdasarkan kolom kunci** ({', '.join(key_cols)}) "
            "yang sudah ditandai tapi BELUM dihapus — perlu ditinjau manual."
        )
    if cols_with_dup_values > 0:
        findings.append(
            f"Ditemukan **{cols_with_dup_values} kolom** dengan nilai yang berulang di beberapa baris "
            "(mis. nama, email, atau kolom lain yang sama persis) — lihat 'Ringkasan Nilai Berulang per Kolom' di Step 2."
        )
    if not findings:
        findings.append("Tidak ditemukan masalah missing value maupun duplikat pada data ini.")

    for f in findings:
        st.write("•", f)

    # ---------- REKOMENDASI ----------
    st.markdown("### 💡 Rekomendasi Tindakan")
    recommendations = []
    if overall_score < 70:
        recommendations.append("Lakukan audit data secara menyeluruh sebelum digunakan untuk pelaporan atau pengambilan keputusan.")
    if total_missing > 0:
        recommendations.append("Lengkapi nilai kosong melalui verifikasi ke sumber data asli (bukan diisi otomatis).")
    if key_cols and dup_key_count > 0:
        recommendations.append(f"Tinjau {dup_key_count} baris yang ditandai duplikat kolom kunci bersama pemilik data terkait.")
    if cols_with_dup_values > 0:
        recommendations.append(f"Periksa {cols_with_dup_values} kolom yang punya nilai berulang — pastikan itu memang wajar (mis. nama umum) bukan salah input/duplikasi data.")
    if not recommendations:
        recommendations.append("Data sudah dalam kondisi baik — lanjutkan monitoring kualitas data secara berkala.")

    for i, r in enumerate(recommendations, 1):
        st.write(f"{i}. {r}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Kembali ke Data Cleaning"):
            st.session_state.page = "Data Cleaning"
            st.rerun()
    with col2:
        if st.button("🔄 Mulai Analisis Baru"):
            st.session_state.clear()
            st.rerun()
