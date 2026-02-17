import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import streamlit as st
from PIL import Image

# --- keyup optional ---
try:
    from streamlit_keyup import st_keyup
    HAS_KEYUP = True
except Exception:
    HAS_KEYUP = False
    st_keyup = None


# =========================
# CONFIG (DEINE PFADE)
# =========================
IMG_DIR = Path(r"C:\Users\NukMed-AI\Desktop\Soft Tissue Diana\nnunet_inference_inputs_ANT_FINAL_ALL_rot180")
SEG_DIR = Path(r"C:\nnunet\predictions\ANT_FINAL_ALL_rot180")

# Output-Ordner (Unterordner werden hier erstellt)
OUTPUT_DIR = Path(r"C:\Users\NukMed-AI\Desktop\Soft Tissue Diana\Annotation")

# Resume-CSV optional (alte)
OLD_CSV = Path(r"C:\nnunet\predictions\manual_side_labels_ANT.csv")

# neue CSV (Session)
OUT_CSV = Path(r"C:\nnunet\predictions\manual_labels_streamlit_ANT.csv")

ACTION = "copy"  # "copy" oder "move"
MAX_N = 21488
ALPHA = 0.45

LABEL_MAP = {
    "1": "left",
    "2": "right",
    "3": "empty",
    "4": "skip",
    "5": "low_quality",
    "6": "wrong_frame",
}

ALL_LABELS = list(set(LABEL_MAP.values()))


# =========================
# Helpers
# =========================
def load_2d(p: Path) -> np.ndarray:
    x = np.asanyarray(nib.load(str(p)).dataobj)
    if x.ndim == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    elif x.ndim == 3 and x.shape[2] > 1:
        x = x[:, :, 0]
    return x

def norm_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= np.nanmin(x)
    m = np.nanmax(x)
    if m > 0:
        x /= m
    return (x * 255).astype(np.uint8)

def make_overlay(img_u8: np.ndarray, seg: np.ndarray, alpha=0.45) -> np.ndarray:
    seg_bin = (seg > 0).astype(np.uint8) * 255
    img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1).astype(np.float32)
    mask_rgb = np.zeros_like(img_rgb)
    mask_rgb[..., 0] = seg_bin
    m = (seg_bin > 0)[..., None].astype(np.float32)
    out = img_rgb * (1 - alpha * m) + mask_rgb * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)

def find_matching_image(seg_name: str) -> Path | None:
    c1 = IMG_DIR / seg_name
    if c1.exists():
        return c1
    if seg_name.endswith(".nii.gz"):
        base = seg_name[:-7]
        c2 = IMG_DIR / f"{base}_0000.nii.gz"
        if c2.exists():
            return c2
        if base.endswith("_0000"):
            c3 = IMG_DIR / f"{base[:-5]}.nii.gz"
            if c3.exists():
                return c3
    return None

def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for lbl in ALL_LABELS:
        (OUTPUT_DIR / lbl.upper()).mkdir(parents=True, exist_ok=True)

def sort_file(seg_path: Path, label: str) -> Path:
    ensure_output_dirs()
    dst = OUTPUT_DIR / label.upper() / seg_path.name
    if ACTION == "move":
        shutil.move(str(seg_path), str(dst))
    else:
        shutil.copy2(str(seg_path), str(dst))
    return dst

def remove_sorted_copy(dst_path: Path):
    try:
        if dst_path.exists():
            dst_path.unlink()
    except Exception:
        pass

def load_existing_labels() -> pd.DataFrame:
    dfs = []
    if OLD_CSV.exists():
        old = pd.read_csv(OLD_CSV)
        if {"filename","label"}.issubset(old.columns):
            dfs.append(old[["filename","label"]])
    if OUT_CSV.exists():
        cur = pd.read_csv(OUT_CSV)
        if {"filename","label"}.issubset(cur.columns):
            dfs.append(cur[["filename","label"]])

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["filename"], keep="last").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["filename","label"])

def save_df(df: pd.DataFrame):
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Segmentation Side Labeler (ANT)", layout="wide")
st.title("Segmentation Labeler (ANT) — Hotkeys 1–6, u=undo, q=quit")

st.markdown(
    """
**Hotkeys (ohne Enter):**  
1=left · 2=right · 3=empty · 4=skip · 5=low_quality · 6=wrong_frame · u=undo · q=save+stop
"""
)

# init state
if "df" not in st.session_state:
    st.session_state.df = load_existing_labels()
if "history" not in st.session_state:
    st.session_state.history = []
if "todo" not in st.session_state:
    seg_files = sorted(list(SEG_DIR.glob("*.nii*")))[:MAX_N]
    done = set(st.session_state.df["filename"].astype(str).tolist())
    st.session_state.todo = [p for p in seg_files if p.name not in done]
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "running" not in st.session_state:
    st.session_state.running = True

# --- IMPORTANT: prevents multi-fire on reruns ---
if "busy" not in st.session_state:
    st.session_state.busy = False

# --- NEW: safe fallback handling (only for the text input) ---
if "_hotkey_value" not in st.session_state:
    st.session_state._hotkey_value = ""
if "_clear_hotkey" not in st.session_state:
    st.session_state._clear_hotkey = False

# controls
colA, colB, colC = st.columns([1,1,2])
with colA:
    st.write(f"**To-do:** {len(st.session_state.todo)}")
with colB:
    st.write(f"**Progress:** {st.session_state.idx}/{len(st.session_state.todo)}")
with colC:
    st.write(f"**CSV:** {OUT_CSV}")
    st.write(f"**Output dir:** {OUTPUT_DIR}")

if not st.session_state.running:
    st.success("Stopped. CSV saved.")
    st.stop()

if st.session_state.idx >= len(st.session_state.todo):
    st.success("Fertig — keine Fälle mehr.")
    save_df(st.session_state.df)
    st.stop()

# =========================
# key capture (UNCHANGED look; only prevents Streamlit error)
# =========================
key = ""
if HAS_KEYUP:
    key = st_keyup("", key="hotkey", debounce=0)
else:
    def _on_hotkey_change():
        # capture once on Enter
        st.session_state._hotkey_value = st.session_state.hotkey_fallback
        st.session_state._clear_hotkey = True

    # clear BEFORE widget is instantiated in this run
    if st.session_state._clear_hotkey:
        st.session_state.hotkey_fallback = ""
        st.session_state._clear_hotkey = False

    st.text_input(
        "Hotkey-Fallback (1–6 / u / q) + Enter",
        key="hotkey_fallback",
        on_change=_on_hotkey_change,
    )
    key = st.session_state._hotkey_value
    st.session_state._hotkey_value = ""


def apply_label(label: str):
    if st.session_state.get("busy", False):
        return
    st.session_state.busy = True

    i = st.session_state.idx
    todo = st.session_state.todo
    if i >= len(todo):
        st.session_state.busy = False
        return

    p_seg = todo[i]

    # update df
    df_local = st.session_state.df
    new_row = pd.DataFrame([{"filename": p_seg.name, "label": label}])
    df2 = pd.concat([df_local, new_row], ignore_index=True)
    df2 = df2.drop_duplicates(subset=["filename"], keep="last").reset_index(drop=True)
    st.session_state.df = df2

    # save csv immediately
    save_df(df2)

    # sort/copy file
    sorted_dst = sort_file(p_seg, label)

    # record for undo
    st.session_state.history.append({
        "filename": p_seg.name,
        "label": label,
        "sorted_dst": str(sorted_dst)
    })

    # advance exactly once
    st.session_state.idx = i + 1

    # clear keyup input (safe)
    if "hotkey" in st.session_state:
        st.session_state.hotkey = ""
    # clear fallback input NEXT run (safe)
    st.session_state._clear_hotkey = True

    st.session_state.busy = False


def undo():
    if st.session_state.get("busy", False):
        return
    if not st.session_state.history:
        return
    st.session_state.busy = True

    last = st.session_state.history.pop()

    # remove from df (only from current df)
    df_local = st.session_state.df
    df_local = df_local[df_local["filename"] != last["filename"]].copy()
    st.session_state.df = df_local
    save_df(df_local)

    # remove sorted copy
    remove_sorted_copy(Path(last["sorted_dst"]))

    # step back
    st.session_state.idx = max(0, st.session_state.idx - 1)

    # clear keyup input (safe)
    if "hotkey" in st.session_state:
        st.session_state.hotkey = ""
    # clear fallback input NEXT run (safe)
    st.session_state._clear_hotkey = True

    st.session_state.busy = False


def save_stop():
    save_df(st.session_state.df)
    st.session_state.running = False


# handle key
if key:
    k = key.strip().lower()
    if k in LABEL_MAP:
        apply_label(LABEL_MAP[k])
        st.rerun()
    elif k == "u":
        undo()
        st.rerun()
    elif k == "q":
        save_stop()
        st.rerun()


# =========================
# show current case (UNCHANGED DISPLAY CODE)
# =========================
p_seg = st.session_state.todo[st.session_state.idx]
p_img = find_matching_image(p_seg.name)

seg = load_2d(p_seg).astype(np.int16)

left, right = st.columns([2, 1])

with left:
    st.subheader(f"{st.session_state.idx+1}/{len(st.session_state.todo)} — {p_seg.name}")
    if p_img is None:
        st.warning("Kein passendes Image gefunden — zeige nur Maske.")
        vis = (seg > 0).astype(np.uint8) * 255
        st.image(vis, clamp=True)
    else:
        img = load_2d(p_img)
        img_u8 = norm_u8(img)
        ov = make_overlay(img_u8, seg, alpha=ALPHA)
        st.image(ov, clamp=True)

with right:
    st.subheader("Quick info")
    st.write("Seg nonzero pixels:", int((seg > 0).sum()))
    st.write("Image match:", str(p_img) if p_img else "None")

    st.markdown("### Fallback Buttons (falls Hotkeys spinnen)")
    c1, c2, c3 = st.columns(3)
    if c1.button("LEFT (1)"):
        apply_label("left")
        st.rerun()
    if c2.button("RIGHT (2)"):
        apply_label("right")
        st.rerun()
    if c3.button("EMPTY (3)"):
        apply_label("empty")
        st.rerun()

    c4, c5, c6 = st.columns(3)
    if c4.button("SKIP (4)"):
        apply_label("skip")
        st.rerun()
    if c5.button("LOW_QUALITY (5)"):
        apply_label("low_quality")
        st.rerun()
    if c6.button("WRONG_FRAME (6)"):
        apply_label("wrong_frame")
        st.rerun()

    c7, c8 = st.columns(2)
    if c7.button("UNDO (u)"):
        undo()
        st.rerun()
    if c8.button("SAVE+STOP (q)"):
        save_stop()
        st.rerun()
