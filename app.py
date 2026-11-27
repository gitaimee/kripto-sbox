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
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ MUAT FILE S-BOX ------------------
BASE_CANDIDATES = [
    ".",
    "./sbox_project",
    "/content/drive/MyDrive/sbox_project (1)",
    "/content/drive/MyDrive/sbox_project",
]

sbox_path = None
for folder in BASE_CANDIDATES:
    candidate = os.path.join(folder, "sbox_ml.npy")
    if os.path.exists(candidate):
        sbox_path = candidate
        break

if sbox_path is not None:
    SBOX_ML = np.load(sbox_path).astype(np.uint8)
    sbox_source = f"sbox_ml.npy ({sbox_path})"
else:
    SBOX_ML = np.arange(256, dtype=np.uint8)
    sbox_source = "S-Box identitas fallback (sbox_ml.npy tidak ditemukan)"

INV_SBOX_ML = np.argsort(SBOX_ML).astype(np.uint8)

# ------------------ TABEL LOOKUP GF(2^8) & AES-128 CEPAT ------------------
def _build_gf_tables():
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

    mul2 = np.array([gf_mul(2, x) for x in range(256)], dtype=np.uint8)
    mul3 = np.array([gf_mul(3, x) for x in range(256)], dtype=np.uint8)
    mul9 = np.array([gf_mul(9, x) for x in range(256)], dtype=np.uint8)
    mul11 = np.array([gf_mul(11, x) for x in range(256)], dtype=np.uint8)
    mul13 = np.array([gf_mul(13, x) for x in range(256)], dtype=np.uint8)
    mul14 = np.array([gf_mul(14, x) for x in range(256)], dtype=np.uint8)
    return mul2, mul3, mul9, mul11, mul13, mul14

MUL2, MUL3, MUL9, MUL11, MUL13, MUL14 = _build_gf_tables()

SHIFT_ROWS_IDX = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11], dtype=np.intp)
INV_SHIFT_ROWS_IDX = np.array([0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3], dtype=np.intp)

def key_expansion(key_bytes, sbox):
    w = [list(key_bytes[4 * i : 4 * i + 4]) for i in range(4)]
    rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]
            t = [int(sbox[b]) for b in t]
            t[0] ^= rcon[i // 4 - 1]
        w.append([w[i - 4][j] ^ t[j] for j in range(4)])
    rk = [sum(w[4 * r : 4 * r + 4], []) for r in range(11)]
    return np.array(rk, dtype=np.uint8)

def aes_encrypt_blocks_vec(blocks_u8: np.ndarray, rk: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    """Enkripsi batch blok (N, 16) sekaligus menggunakan vektorisasi NumPy."""
    s = np.bitwise_xor(blocks_u8, rk[0])
    for r in range(1, 11):
        s = sbox[s]
        s = s[:, SHIFT_ROWS_IDX]
        if r < 10:
            s_col = s.reshape(-1, 4, 4)
            c0, c1, c2, c3 = s_col[:, :, 0], s_col[:, :, 1], s_col[:, :, 2], s_col[:, :, 3]
            m0 = MUL2[c0] ^ MUL3[c1] ^ c2 ^ c3
            m1 = c0 ^ MUL2[c1] ^ MUL3[c2] ^ c3
            m2 = c0 ^ c1 ^ MUL2[c2] ^ MUL3[c3]
            m3 = MUL3[c0] ^ c1 ^ c2 ^ MUL2[c3]
            s = np.stack([m0, m1, m2, m3], axis=-1).reshape(-1, 16)
        s = np.bitwise_xor(s, rk[r])
    return s

def aes_decrypt_blocks_vec(blocks_u8: np.ndarray, rk: np.ndarray, inv_sbox: np.ndarray) -> np.ndarray:
    """Dekripsi batch blok (N, 16) sekaligus menggunakan vektorisasi NumPy."""
    s = np.bitwise_xor(blocks_u8, rk[10])
    for r in range(9, -1, -1):
        s = s[:, INV_SHIFT_ROWS_IDX]
        s = inv_sbox[s]
        s = np.bitwise_xor(s, rk[r])
        if r > 0:
            s_col = s.reshape(-1, 4, 4)
            c0, c1, c2, c3 = s_col[:, :, 0], s_col[:, :, 1], s_col[:, :, 2], s_col[:, :, 3]
            m0 = MUL14[c0] ^ MUL11[c1] ^ MUL13[c2] ^ MUL9[c3]
            m1 = MUL9[c0] ^ MUL14[c1] ^ MUL11[c2] ^ MUL13[c3]
            m2 = MUL13[c0] ^ MUL9[c1] ^ MUL14[c2] ^ MUL11[c3]
            m3 = MUL11[c0] ^ MUL13[c1] ^ MUL9[c2] ^ MUL14[c3]
            s = np.stack([m0, m1, m2, m3], axis=-1).reshape(-1, 16)
    return s

def _derive_ivs(rk: np.ndarray, sbox: np.ndarray):
    """Menurunkan IV1 (CBC maju) dan IV2 (CBC mundur) secara deterministik dari kunci."""
    seeds = np.array([[0x00] * 16, [0xFF] * 16], dtype=np.uint8)
    ivs = aes_encrypt_blocks_vec(seeds, rk, sbox)
    return ivs[0], ivs[1]

def _aes_encrypt_single_list(block_list, rk_list, sbox_list):
    s = [b ^ k for b, k in zip(block_list, rk_list[0])]
    for r in range(1, 11):
        s = [sbox_list[b] for b in s]
        s = [s[0], s[5], s[10], s[15], s[4], s[9], s[14], s[3], s[8], s[13], s[2], s[7], s[12], s[1], s[6], s[11]]
        if r < 10:
            for c in range(4):
                c0, c1, c2, c3 = s[4 * c : 4 * c + 4]
                s[4 * c] = MUL2[c0] ^ MUL3[c1] ^ c2 ^ c3
                s[4 * c + 1] = c0 ^ MUL2[c1] ^ MUL3[c2] ^ c3
                s[4 * c + 2] = c0 ^ c1 ^ MUL2[c2] ^ MUL3[c3]
                s[4 * c + 3] = MUL3[c0] ^ c1 ^ c2 ^ MUL2[c3]
        s = [b ^ k for b, k in zip(s, rk_list[r])]
    return s

def encrypt_image_cbc(img_arr: np.ndarray, key_tuple: tuple) -> np.ndarray:
    """
    Enkripsi AES-128 Mode CBC Dua Arah (CBC Maju IV1 + CBC Mundur IV2)
    agar perubahan 1 piksel di posisi mana pun menyebar ke seluruh blok citra.
    """
    rk = key_expansion(key_tuple, SBOX_ML)
    iv1, iv2 = _derive_ivs(rk, SBOX_ML)

    flat = img_arr.reshape(-1)
    pad_len = (-flat.size) % 16
    if pad_len:
        flat = np.pad(flat, (0, pad_len), mode="constant")

    n_blocks = flat.size // 16
    blocks = flat.reshape(n_blocks, 16).tolist()
    rk_list = rk.tolist()
    sbox_list = SBOX_ML.tolist()

    # Pass 1: CBC Maju (Forward) dengan IV1
    fwd = [None] * n_blocks
    prev = iv1.tolist()
    for i in range(n_blocks):
        xored = [b ^ p for b, p in zip(blocks[i], prev)]
        enc = _aes_encrypt_single_list(xored, rk_list, sbox_list)
        fwd[i] = enc
        prev = enc

    # Pass 2: CBC Mundur (Backward) dengan IV2
    bwd = [None] * n_blocks
    prev = iv2.tolist()
    for i in range(n_blocks - 1, -1, -1):
        xored = [b ^ p for b, p in zip(fwd[i], prev)]
        enc = _aes_encrypt_single_list(xored, rk_list, sbox_list)
        bwd[i] = enc
        prev = enc

    cipher_flat = np.array(bwd, dtype=np.uint8).reshape(-1)[:img_arr.size]
    return cipher_flat.reshape(img_arr.shape)

def decrypt_image_cbc(cipher_arr: np.ndarray, key_tuple: tuple) -> np.ndarray:
    """
    Dekripsi AES-128 Mode CBC Dua Arah (Balik CBC Mundur -> Balik CBC Maju)
    menggunakan batch vektorisasi penuh sehingga sangat cepat.
    """
    rk = key_expansion(key_tuple, SBOX_ML)
    iv1, iv2 = _derive_ivs(rk, SBOX_ML)

    flat_c = cipher_arr.reshape(-1)
    pad_len = (-flat_c.size) % 16
    if pad_len:
        flat_c = np.pad(flat_c, (0, pad_len), mode="constant")

    c_blocks = flat_c.reshape(-1, 16)

    # Langkah 1: Balik CBC Mundur (Backward) secara paralel
    dec_bwd = aes_decrypt_blocks_vec(c_blocks, rk, INV_SBOX_ML)
    next_c = np.vstack([c_blocks[1:], iv2.reshape(1, 16)])
    fwd_blocks = np.bitwise_xor(dec_bwd, next_c)

    # Langkah 2: Balik CBC Maju (Forward) secara paralel
    dec_fwd = aes_decrypt_blocks_vec(fwd_blocks, rk, INV_SBOX_ML)
    prev_f = np.vstack([iv1.reshape(1, 16), fwd_blocks[:-1]])
    plain_blocks = np.bitwise_xor(dec_fwd, prev_f)

    plain_flat = plain_blocks.reshape(-1)[:cipher_arr.size]
    return plain_flat.reshape(cipher_arr.shape)

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
    a, b = c1.astype(np.int32), c2.astype(np.int32)
    npcr_val = float((a != b).mean() * 100.0)
    uaci_val = float((np.abs(a - b) / 255.0).mean() * 100.0)
    return npcr_val, uaci_val

@st.cache_data(show_spinner=False)
def run_full_crypto_and_tests(pixels: np.ndarray, key_tuple: tuple, sbox_snapshot: bytes):
    """Menjalankan enkripsi, dekripsi, uji NPCR/UACI (5x), dan uji sensitivitas kunci dengan cache."""
    # 1. Enkripsi Utama
    t_enc_start = time.perf_counter()
    cipher_img = encrypt_image_cbc(pixels, key_tuple)
    t_enc_ms = (time.perf_counter() - t_enc_start) * 1000.0

    # 2. Dekripsi Utama
    t_dec_start = time.perf_counter()
    decrypted_img = decrypt_image_cbc(cipher_img, key_tuple)
    t_dec_ms = (time.perf_counter() - t_dec_start) * 1000.0

    # 3. Uji Diferensial NPCR & UACI (5 kali acak +1 pada 1 piksel)
    n_uji = 5
    rng = np.random.default_rng(123)
    diff_rows = []
    for t in range(n_uji):
        pos = tuple(int(rng.integers(0, s)) for s in pixels.shape)
        img2 = pixels.copy()
        img2[pos] = (int(img2[pos]) + 1) % 256
        c2 = encrypt_image_cbc(img2, key_tuple)
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
    df_diff_sum["Lolos NPCR (> 99,5893%)"] = df_diff_sum["NPCR (%)"].apply(
        lambda x: "Ya" if x > 99.5893 else "Tidak"
    )
    df_diff_sum["Lolos UACI (33,3730-33,5541%)"] = df_diff_sum["UACI (%)"].apply(
        lambda x: "Ya" if (33.3730 <= x <= 33.5541) else "Tidak"
    )

    # 4. Uji Sensitivitas Kunci (Flip 1-bit pada byte ke-0)
    key_salah = (key_tuple[0] ^ 1,) + key_tuple[1:]
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

    return cipher_img, decrypted_img, t_enc_ms, t_dec_ms, df_diff, df_diff_sum, df_key

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
            <div class="terminal-title">aes128-cbc-bidirectional — {title_dim}</div>
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
    <div class="hero-title">Enkripsi dan Dekripsi Citra Digital</div>
    <p class="hero-subtitle">Simulasi AES-128 mode CBC dua arah (IV1 + IV2) menggunakan S-Box 8-bit kustom beserta uji statistik, diferensial (NPCR/UACI), dan sensitivitas kunci.</p>
</div>
""", unsafe_allow_html=True)

# ------------------ INPUT GAMBAR, RESOLUSI & KUNCI ------------------
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.0, 1.1, 1.2])

with ctrl_col1:
    uploaded = st.file_uploader(
        "Pilih Gambar (PNG, JPG, JPEG, TIF, TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

with ctrl_col2:
    resize_opt = st.selectbox(
        "Resolusi Pemrosesan",
        ["Ukuran Asli (Sesuai Colab)", "256 x 256", "128 x 128"],
        index=0
    )

with ctrl_col3:
    key_hex_input = st.text_input(
        "Kunci Enkripsi (Hex 128-bit)",
        value="2b28ab097eaef7cf15d2154f16a6883c",
        max_chars=32
    )

clean_hex = "".join(c for c in key_hex_input if c in "0123456789abcdefABCDEF").ljust(32, "0")[:32]
key = tuple(int(clean_hex[i:i+2], 16) for i in range(0, 32, 2))

if uploaded is None:
    st.info("Unggah gambar (misalnya mandrill_rgb.png / mandrill_gray.png) untuk memulai proses enkripsi, dekripsi, dan pengujian.")
else:
    raw_img = Image.open(uploaded)
    is_grayscale = raw_img.mode in ("L", "1", "I", "F")

    img_proc = raw_img.convert("L") if is_grayscale else raw_img.convert("RGB")
    if resize_opt == "256 x 256":
        img_proc = img_proc.resize((256, 256))
    elif resize_opt == "128 x 128":
        img_proc = img_proc.resize((128, 128))

    pixels = np.array(img_proc, dtype=np.uint8)
    h_img, w_img = pixels.shape[:2]

    if is_grayscale:
        mode_label = f"{h_img}x{w_img} Grayscale (1 Kanal)"
        short_dim = f"{h_img}x{w_img}"
    else:
        mode_label = f"{h_img}x{w_img}x3 RGB (3 Kanal)"
        short_dim = f"{h_img}x{w_img}x3"

    total_bytes = pixels.size
    total_blocks = (total_bytes + 15) // 16

    # Jalankan Kriptografi & Pengujian (Cached)
    with st.spinner("Menjalankan enkripsi AES-128 CBC dua arah & pengujian lengkap..."):
        cipher_img, decrypted_img, t_enc_ms, t_dec_ms, df_diff, df_diff_sum, df_key = run_full_crypto_and_tests(
            pixels, key, SBOX_ML.tobytes()
        )

    byte_diff = int(np.sum(pixels != decrypted_img))

    # Kotak Terminal Proses
    st.subheader("Log Proses")
    logs = []
    ts_now = time.strftime("%H:%M:%S")
    def add_log(tag_cls, tag_text, msg):
        logs.append(f'<span class="log-time">[{ts_now}]</span> <span class="{tag_cls}">[{tag_text}]</span> {msg}')

    add_log("log-tag-info", "INIT", f"Sumber S-Box: <span class='log-dim'>{html.escape(sbox_source)}</span>")
    add_log("log-tag-info", "CITRA", f"Format: <b>{mode_label}</b> ({total_bytes:,} byte / {total_blocks:,} blok).")
    add_log("log-tag-info", "KUNCI", f"Key schedule AES-128 siap | Hex: <span class='log-hex'>{clean_hex.upper()}</span>")
    add_log("log-tag-enc", "ENKRIPSI", "Menjalankan AES-128 CBC Dua Arah (CBC Maju IV1 -> CBC Mundur IV2)...")
    add_log("log-tag-enc", "ENKRIPSI", f"Blok #0 Asli      : <span class='log-hex'>{fmt_hex_preview(pixels)} ...</span>")
    add_log("log-tag-enc", "ENKRIPSI", f"Blok #0 Terenkripsi: <span class='log-hex'>{fmt_hex_preview(cipher_img)} ...</span>")
    add_log("log-tag-enc", "SELESAI ", f"Enkripsi {total_blocks:,} blok selesai dalam <b>{t_enc_ms:.1f} ms</b>.")
    add_log("log-tag-dec", "DEKRIPSI", "Menjalankan dekripsi AES-128 CBC Dua Arah menggunakan INV_SBOX_ML...")
    add_log("log-tag-dec", "DEKRIPSI", f"Blok #0 Dekripsi   : <span class='log-hex'>{fmt_hex_preview(decrypted_img)} ...</span>")
    add_log("log-tag-dec", "SELESAI ", f"Dekripsi {total_blocks:,} blok selesai dalam <b>{t_dec_ms:.1f} ms</b> | Selisih: <b>{byte_diff}</b>.")
    st.markdown(render_terminal(logs, short_dim), unsafe_allow_html=True)

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

    tab_sum, tab_detail, tab_key = st.tabs([
        "Rata-rata NPCR & UACI",
        "Detail 5x Uji Piksel (+1)",
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

    full_label = f"Matriks Penuh ({h_img}x{w_img})"
    if is_grayscale:
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            view_mode = st.radio(
                "Ukuran Matriks:",
                ["Cuplikan 8x8 Piksel Pertama", full_label],
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
                ["Cuplikan 8x8 Piksel Pertama", full_label],
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

    is_crop_8x8 = "8x8" in view_mode
    slice_h = min(8, h_img) if is_crop_8x8 else h_img
    slice_w = min(8, w_img) if is_crop_8x8 else w_img

    def format_matrix_df(mat):
        sub = mat[:slice_h, :slice_w] if is_grayscale else mat[:slice_h, :slice_w, ch_idx]
        df = pd.DataFrame(
            sub,
            index=[f"B{i}" for i in range(slice_h)],
            columns=[f"K{j}" for j in range(slice_w)]
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