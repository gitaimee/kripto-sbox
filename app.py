import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import html

# ------------------ KONFIGURASI HALAMAN ------------------
st.set_page_config(
    page_title="Enkripsi & Dekripsi Citra AES-128",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ KUSTOMISASI TAMPILAN & TERMINAL ------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    
    /* Header Utama */
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.65rem;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Kotak Terminal */
    .terminal-wrapper {
        background-color: #0b0f19;
        border: 1px solid #263045;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
        margin-top: 0.75rem;
        margin-bottom: 1.5rem;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    }
    .terminal-header {
        background-color: #161d2f;
        padding: 0.65rem 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #263045;
    }
    .terminal-dots {
        display: flex;
        gap: 7px;
    }
    .dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-red { background-color: #ff5f56; }
    .dot-yellow { background-color: #ffbd2e; }
    .dot-green { background-color: #27c93f; }
    .terminal-title {
        color: #94a3b8;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
    }
    .terminal-body {
        padding: 1rem 1.25rem;
        color: #e2e8f0;
        font-size: 0.83rem;
        line-height: 1.65;
        max-height: 320px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    .log-time { color: #64748b; }
    .log-tag-info { color: #38bdf8; font-weight: 600; }
    .log-tag-enc { color: #c084fc; font-weight: 600; }
    .log-tag-dec { color: #34d399; font-weight: 600; }
    .log-hex { color: #93c5fd; }
    .log-dim { color: #94a3b8; }

    /* Kartu Statistik */
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ MUAT FILE S-BOX ------------------
BASE = "/content/drive/MyDrive/sbox_project (1)"
if not os.path.exists(BASE):
    BASE = "/content/drive/MyDrive/sbox_project"
if not os.path.exists(BASE):
    BASE = "./sbox_project"

sbox_path = os.path.join(BASE, "sbox_ml.npy")

if os.path.exists(sbox_path):
    SBOX_ML = np.load(sbox_path).astype(np.uint8)
    sbox_source = f"sbox_ml.npy ({sbox_path})"
else:
    SBOX_ML = np.arange(256, dtype=np.uint8)
    sbox_source = "S-Box bawaan (0-255)"

INV_SBOX_ML = np.argsort(SBOX_ML).astype(np.uint8)

# ------------------ FUNGSI KRIPTOGRAFI AES-128 ------------------
def gf_mul(a: int, b: int) -> int:
    r = 0
    for _ in range(8):
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11B
        b >>= 1
    return r

def mix_columns_col(col):
    c0 = gf_mul(2, col[0]) ^ gf_mul(3, col[1]) ^ col[2] ^ col[3]
    c1 = col[0] ^ gf_mul(2, col[1]) ^ gf_mul(3, col[2]) ^ col[3]
    c2 = col[0] ^ col[1] ^ gf_mul(2, col[2]) ^ gf_mul(3, col[3])
    c3 = gf_mul(3, col[0]) ^ col[1] ^ col[2] ^ gf_mul(2, col[3])
    return [c0, c1, c2, c3]

def inv_mix_columns_col(col):
    c0 = gf_mul(14, col[0]) ^ gf_mul(11, col[1]) ^ gf_mul(13, col[2]) ^ gf_mul(9, col[3])
    c1 = gf_mul(9, col[0]) ^ gf_mul(14, col[1]) ^ gf_mul(11, col[2]) ^ gf_mul(13, col[3])
    c2 = gf_mul(13, col[0]) ^ gf_mul(9, col[1]) ^ gf_mul(14, col[2]) ^ gf_mul(11, col[3])
    c3 = gf_mul(11, col[0]) ^ gf_mul(13, col[1]) ^ gf_mul(9, col[2]) ^ gf_mul(14, col[3])
    return [c0, c1, c2, c3]

def key_expansion(key, sbox):
    w = [list(key[4*i : 4*i+4]) for i in range(4)]
    rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]
            t = [int(sbox[b]) for b in t]
            t[0] ^= rcon[i // 4 - 1]
        w.append([w[i-4][j] ^ t[j] for j in range(4)])
    return [sum(w[4*r : 4*r+4], []) for r in range(11)]

def aes_encrypt_block_fast(block, rk, sbox):
    s = [b ^ k for b, k in zip(block, rk[0])]
    for r in range(1, 11):
        s = [int(sbox[b]) for b in s]
        s = [s[0], s[5], s[10], s[15], s[4], s[9], s[14], s[3], s[8], s[13], s[2], s[7], s[12], s[1], s[6], s[11]]
        if r < 10:
            for c in range(4):
                s[4*c : 4*c+4] = mix_columns_col(s[4*c : 4*c+4])
        s = [b ^ k for b, k in zip(s, rk[r])]
    return s

def aes_decrypt_block_fast(block, rk, inv_sbox):
    s = [b ^ k for b, k in zip(block, rk[10])]
    for r in range(9, -1, -1):
        s = [s[0], s[13], s[10], s[7], s[4], s[1], s[14], s[11], s[8], s[5], s[2], s[15], s[12], s[9], s[6], s[3]]
        s = [int(inv_sbox[b]) for b in s]
        s = [b ^ k for b, k in zip(s, rk[r])]
        if r > 0:
            for c in range(4):
                s[4*c : 4*c+4] = inv_mix_columns_col(s[4*c : 4*c+4])
    return s

def encrypt_image_cbc(img_arr, key_list):
    rk = key_expansion(key_list, SBOX_ML)
    flat = img_arr.flatten()
    cipher = np.zeros_like(flat)
    prev = [0] * 16
    for idx in range(0, len(flat), 16):
        block = flat[idx : idx+16].tolist()
        if len(block) < 16:
            block += [0] * (16 - len(block))
        block = [b ^ p for b, p in zip(block, prev)]
        enc = aes_encrypt_block_fast(block, rk, SBOX_ML)
        cipher[idx : idx+16] = enc[:len(flat)-idx]
        prev = enc
    return cipher.reshape(img_arr.shape)

def decrypt_image_cbc(cipher_arr, key_list):
    rk = key_expansion(key_list, SBOX_ML)
    flat_c = cipher_arr.flatten()
    decrypted = np.zeros_like(flat_c)
    prev = [0] * 16
    for idx in range(0, len(flat_c), 16):
        c_block = flat_c[idx : idx+16].tolist()
        dec = aes_decrypt_block_fast(c_block, rk, INV_SBOX_ML)
        dec = [d ^ p for d, p in zip(dec, prev)]
        decrypted[idx : idx+16] = dec[:len(flat_c)-idx]
        prev = c_block
    return decrypted.reshape(cipher_arr.shape)

# ------------------ FUNGSI UJI STATISTIK, DIFERENSIAL & HISTOGRAM ------------------
def kanal(img):
    return [("Gray", img)] if img.ndim == 2 else [(k, img[..., i]) for i, k in enumerate("RGB")]

def entropy(a):
    p = np.bincount(a.reshape(-1), minlength=256) / a.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def chi_square(a):
    h = np.bincount(a.reshape(-1), minlength=256)
    e = a.size / 256
    return float(((h - e) ** 2 / e).sum())

def npcr_uaci(c1, c2):
    a, b = c1.astype(np.int16), c2.astype(np.int16)
    return float((a != b).mean() * 100), float((np.abs(a - b) / 255).mean() * 100)

def plot_histogram(im, is_cipher=False):
    warna = {"R": "#ef4444", "G": "#22c55e", "B": "#3b82f6"}
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for kn, a in kanal(im):
        h = np.bincount(a.ravel(), minlength=256)
        if im.ndim == 2:
            ax.bar(range(256), h, width=1.0, color="#3b82f6" if is_cipher else "#64748b")
        else:
            ax.plot(range(256), h, color=warna[kn], lw=1.1, label=kn)
    if im.ndim == 3:
        ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Nilai Piksel (0-255)", fontsize=8.5)
    ax.set_ylabel("Frekuensi", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def render_terminal(logs, title_dim):
    body_html = "\n".join(logs)
    return f"""
    <div class="terminal-wrapper">
        <div class="terminal-header">
            <div class="terminal-dots">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
            </div>
            <div class="terminal-title">aes128-cbc-process — {title_dim}</div>
            <div style="width:45px;"></div>
        </div>
        <div class="terminal-body">{body_html}</div>
    </div>
    """

def fmt_hex_preview(arr, n=8):
    return " ".join(f"{int(x):02X}" for x in arr.flatten()[:n])

# ------------------ TAMPILAN UTAMA ------------------
st.markdown("""
<div class="hero-box">
    <div class="hero-badge">AES-128 CBC | S-Box Hasil Optimasi</div>
    <div class="hero-title">Enkripsi dan Dekripsi Citra Digital</div>
    <p class="hero-subtitle">Simulasi AES-128 mode CBC menggunakan S-Box 8-bit kustom beserta uji statistik, diferensial (NPCR/UACI), dan sensitivitas kunci.</p>
</div>
""", unsafe_allow_html=True)

# ------------------ INPUT GAMBAR & KUNCI ------------------
ctrl_col1, ctrl_col2 = st.columns([2.2, 1])

with ctrl_col1:
    uploaded = st.file_uploader(
        "Pilih Gambar (PNG, JPG, JPEG, TIF, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

with ctrl_col2:
    key_hex_input = st.text_input(
        "Kunci Enkripsi (16-Byte Hex)",
        value="01010101010101010101010101010101",
        max_chars=32
    )

clean_hex = "".join(c for c in key_hex_input if c in "0123456789abcdefABCDEF").ljust(32, "0")[:32]
key = [int(clean_hex[i:i+2], 16) for i in range(0, 32, 2)]

if uploaded is None:
    st.info("Unggah gambar untuk memulai proses enkripsi, dekripsi, dan pengujian.")
else:
    raw_img = Image.open(uploaded)
    is_grayscale = raw_img.mode in ("L", "1", "I", "F")

    if is_grayscale:
        img = raw_img.convert("L").resize((128, 128))
        mode_label = "128x128 Grayscale (1 Kanal)"
        short_dim = "128x128"
    else:
        img = raw_img.convert("RGB").resize((128, 128))
        mode_label = "128x128x3 RGB (3 Kanal)"
        short_dim = "128x128x3"

    pixels = np.array(img, dtype=np.uint8)
    total_bytes = pixels.size
    total_blocks = total_bytes // 16

    # Kotak Terminal Proses
    st.subheader("Log Proses")
    terminal_slot = st.empty()
    logs = []

    def push_log(tag_cls, tag_text, msg):
        ts = time.strftime("%H:%M:%S")
        logs.append(f'<span class="log-time">[{ts}]</span> <span class="{tag_cls}">[{tag_text}]</span> {msg}')
        terminal_slot.markdown(render_terminal(logs, short_dim), unsafe_allow_html=True)

    push_log("log-tag-info", "INIT", f"Sumber S-Box: <span class='log-dim'>{html.escape(sbox_source)}</span>")
    push_log("log-tag-info", "CITRA", f"Format: <b>{mode_label}</b> ({total_bytes:,} byte / {total_blocks:,} blok).")
    push_log("log-tag-info", "KUNCI", f"Key schedule AES-128 siap | Hex: <span class='log-hex'>{clean_hex.upper()}</span>")

    # Proses Enkripsi
    push_log("log-tag-enc", "ENKRIPSI", "Menjalankan AES-128 CBC (SubBytes -> ShiftRows -> MixColumns -> AddRoundKey)...")
    t_enc_start = time.perf_counter()
    cipher_img = encrypt_image_cbc(pixels, key)
    t_enc_ms = (time.perf_counter() - t_enc_start) * 1000

    push_log("log-tag-enc", "ENKRIPSI", f"Blok #0 Asli      : <span class='log-hex'>{fmt_hex_preview(pixels)} ...</span>")
    push_log("log-tag-enc", "ENKRIPSI", f"Blok #0 Terenkripsi: <span class='log-hex'>{fmt_hex_preview(cipher_img)} ...</span>")
    push_log("log-tag-enc", "SELESAI ", f"Enkripsi {total_blocks:,} blok selesai dalam <b>{t_enc_ms:.1f} ms</b>.")

    # Proses Dekripsi
    push_log("log-tag-dec", "DEKRIPSI", "Menjalankan dekripsi AES-128 CBC menggunakan INV_SBOX_ML...")
    t_dec_start = time.perf_counter()
    decrypted_img = decrypt_image_cbc(cipher_img, key)
    t_dec_ms = (time.perf_counter() - t_dec_start) * 1000
    byte_diff = int(np.sum(pixels != decrypted_img))

    push_log("log-tag-dec", "DEKRIPSI", f"Blok #0 Dekripsi   : <span class='log-hex'>{fmt_hex_preview(decrypted_img)} ...</span>")
    push_log("log-tag-dec", "SELESAI ", f"Dekripsi {total_blocks:,} blok selesai dalam <b>{t_dec_ms:.1f} ms</b> | Selisih: <b>{byte_diff}</b>.")

    # Ringkasan Metrik
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Ukuran & Blok</div><div class="stat-value">{short_dim} ({total_blocks:,} blok)</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Durasi Enkripsi</div><div class="stat-value">{t_enc_ms:.1f} ms</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Durasi Dekripsi</div><div class="stat-value">{t_dec_ms:.1f} ms</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Selisih Nilai</div><div class="stat-value">{byte_diff} / {total_bytes:,}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Perbandingan Visual Gambar
    st.subheader("Perbandingan Visual Citra")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("**1. Citra Asli**")
            st.image(pixels, use_container_width=True)
    with col2:
        with st.container(border=True):
            st.markdown("**2. Citra Terenkripsi**")
            st.image(cipher_img, use_container_width=True)
    with col3:
        with st.container(border=True):
            st.markdown("**3. Citra Hasil Dekripsi**")
            st.image(decrypted_img, use_container_width=True)

    # Uji Statistik & Histogram
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Uji Statistik & Histogram")

    stat_rows = []
    for (k, a), (_, c) in zip(kanal(pixels), kanal(cipher_img)):
        chi_c = chi_square(c)
        stat_rows.append({
            "Kanal": k,
            "Entropi Asli": f"{entropy(a):.5f}",
            "Entropi Cipher": f"{entropy(c):.5f}",
            "Chi-Square Asli": f"{chi_square(a):.2f}",
            "Chi-Square Cipher": f"{chi_c:.2f}",
            "Seragam (Chi-Square < 293,25)": "Ya" if chi_c < 293.25 else "Tidak",
        })
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        with st.container(border=True):
            st.markdown("**Histogram Citra Asli**")
            fig_asli = plot_histogram(pixels, is_cipher=False)
            st.pyplot(fig_asli, use_container_width=True)
            plt.close(fig_asli)
    with h_col2:
        with st.container(border=True):
            st.markdown("**Histogram Citra Terenkripsi**")
            fig_cipher = plot_histogram(cipher_img, is_cipher=True)
            st.pyplot(fig_cipher, use_container_width=True)
            plt.close(fig_cipher)

    # Uji NPCR, UACI & Sensitivitas Kunci
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Uji Diferensial (NPCR & UACI) dan Sensitivitas Kunci")

    N_UJI = 5
    rng = np.random.default_rng(123)
    diff_rows = []
    for t in range(N_UJI):
        pos = tuple(int(rng.integers(0, s)) for s in pixels.shape)
        img2 = pixels.copy()
        img2[pos] = (int(img2[pos]) + 1) % 256
        c2 = encrypt_image_cbc(img2, key)
        for (k, a), (_, b) in zip(kanal(cipher_img), kanal(c2)):
            npcr_val, uaci_val = npcr_uaci(a, b)
            diff_rows.append({
                "Uji": t + 1,
                "Posisi": str(pos),
                "Kanal": k,
                "NPCR (%)": round(npcr_val, 4),
                "UACI (%)": round(uaci_val, 4),
                "Lolos NPCR": "Ya" if npcr_val > 99.5893 else "Tidak",
                "Lolos UACI": "Ya" if (33.3730 <= uaci_val <= 33.5541) else "Tidak"
            })
    df_diff = pd.DataFrame(diff_rows)

    df_diff_sum = (
        df_diff.groupby("Kanal", sort=False)[["NPCR (%)", "UACI (%)"]]
        .mean()
        .round(4)
        .reset_index()
    )
    df_diff_sum["Lolos NPCR (> 99,5893%)"] = df_diff_sum["NPCR (%)"].apply(lambda x: "Ya" if x > 99.5893 else "Tidak")
    df_diff_sum["Lolos UACI (33,3730-33,5541%)"] = df_diff_sum["UACI (%)"].apply(lambda x: "Ya" if (33.3730 <= x <= 33.5541) else "Tidak")

    key_salah = [key[0] ^ 1] + key[1:]
    c_salah = encrypt_image_cbc(pixels, key_salah)
    d_salah = decrypt_image_cbc(cipher_img, key_salah)
    sama_key = bool(np.array_equal(d_salah, pixels))

    key_rows = []
    for (k, a), (_, b), (_, d) in zip(kanal(cipher_img), kanal(c_salah), kanal(d_salah)):
        npcr_k, uaci_k = npcr_uaci(a, b)
        key_rows.append({
            "Kanal": k,
            "NPCR Cipher (%)": round(npcr_k, 4),
            "UACI Cipher (%)": round(uaci_k, 4),
            "Dekripsi Kunci Salah = Asli": "Ya" if sama_key else "Tidak",
            "Entropi Dekripsi Salah": round(entropy(d), 5)
        })
    df_key = pd.DataFrame(key_rows)

    tab_sum, tab_detail, tab_key = st.tabs([
        "Rata-rata NPCR & UACI",
        f"Detail {N_UJI}x Uji Piksel (+1)",
        "Sensitivitas Kunci (Flip 1-Bit)"
    ])
    with tab_sum:
        st.dataframe(df_diff_sum, hide_index=True, use_container_width=True)
    with tab_detail:
        st.dataframe(df_diff, hide_index=True, use_container_width=True)
    with tab_key:
        st.dataframe(df_key, hide_index=True, use_container_width=True)

    # Tabel Matriks Piksel
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Matriks Nilai Piksel (0-255)")

    if is_grayscale:
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            view_mode = st.radio(
                "Ukuran Matriks:",
                ["Cuplikan 8x8 Piksel Pertama", "Matriks Penuh (128x128)"],
                horizontal=True
            )
        with opt_col2:
            fmt_mode = st.radio(
                "Format Angka:",
                ["Desimal (0-255)", "Heksadesimal (00-FF)"],
                horizontal=True
            )
        ch_suffix = ""
    else:
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            channel_mode = st.radio(
                "Kanal Warna:",
                ["Red (R)", "Green (G)", "Blue (B)"],
                horizontal=True
            )
        with opt_col2:
            view_mode = st.radio(
                "Ukuran Matriks:",
                ["Cuplikan 8x8 Piksel Pertama", "Matriks Penuh (128x128)"],
                horizontal=True
            )
        with opt_col3:
            fmt_mode = st.radio(
                "Format Angka:",
                ["Desimal (0-255)", "Heksadesimal (00-FF)"],
                horizontal=True
            )
        ch_idx = {"Red (R)": 0, "Green (G)": 1, "Blue (B)": 2}[channel_mode]
        ch_suffix = f" ({channel_mode})"

    slice_n = 8 if "8x8" in view_mode else 128

    def format_matrix_df(mat):
        sub = mat[:slice_n, :slice_n] if is_grayscale else mat[:slice_n, :slice_n, ch_idx]
        df = pd.DataFrame(
            sub,
            index=[f"B{i}" for i in range(slice_n)],
            columns=[f"K{j}" for j in range(slice_n)]
        )
        if "Heksadesimal" in fmt_mode:
            return df.style.format(lambda x: f"{x:02X}")
        return df

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"**Matriks Citra Asli{ch_suffix}**")
        st.dataframe(format_matrix_df(pixels), use_container_width=True)
    with m_col2:
        st.markdown(f"**Matriks Citra Terenkripsi{ch_suffix}**")
        st.dataframe(format_matrix_df(cipher_img), use_container_width=True)
    with m_col3:
        st.markdown(f"**Matriks Citra Hasil Dekripsi{ch_suffix}**")
        st.dataframe(format_matrix_df(decrypted_img), use_container_width=True)