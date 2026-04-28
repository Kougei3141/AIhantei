import streamlit as st
import torch
import open_clip
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI 似てる度スコアゲーム", layout="wide")

device = "cpu"  # CloudはCPU前提


# =========================
# モデル読み込み（キャッシュ）
# =========================
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50",  # 軽量で安定
        pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


model, preprocess = load_model()


# =========================
# 前処理
# =========================
def center_crop(img, size=224):
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    return img.resize((size, size))


# =========================
# スコア関数
# =========================
def clip_score(img1, img2):
    i1 = preprocess(img1).unsqueeze(0).to(device)
    i2 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        f1 = model.encode_image(i1)
        f2 = model.encode_image(i2)

        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)

        return float((f1 @ f2.T).item())


def color_score(img1, img2):
    i1 = np.array(img1.resize((64, 64))).astype(np.float32)
    i2 = np.array(img2.resize((64, 64))).astype(np.float32)

    return 1 - np.mean(np.abs(i1 - i2)) / 255


def structure_score(img1, img2):
    i1 = np.array(img1.resize((64, 64)).convert("L")).astype(np.float32)
    i2 = np.array(img2.resize((64, 64)).convert("L")).astype(np.float32)

    return 1 - np.mean(np.abs(i1 - i2)) / 255


# =========================
# 最終スコア（ゲーム調整）
# =========================
def final_score(c, col, struct, difficulty):

    base = (0.75 * c + 0.15 * col + 0.10 * struct) * 100

    score = base + 5  # 気持ちよくする補正

    # 最低保証（意味が似てるとき）
    if c >= 0.85:
        score = max(score, 50)
    elif c >= 0.80:
        score = max(score, 45)
    elif c >= 0.75:
        score = max(score, 40)

    # 意味だけ似てて他が違う場合
    if c >= 0.80 and col < 0.15 and struct < 0.10:
        score -= 10

    # 色ズレ
    if col < 0.08:
        score -= 5

    # 構図ズレ
    if struct < 0.08:
        score -= 5

    # 意味が弱い
    if c < 0.65:
        score -= 10

    # 難易度
    if difficulty == "やさしい":
        score += 8
    elif difficulty == "きびしい":
        score -= 8

    return max(0, min(100, score))


def rank(score):
    if score >= 90:
        return "SS"
    elif score >= 80:
        return "S"
    elif score >= 70:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 45:
        return "C"
    else:
        return "D"


# =========================
# UI
# =========================
st.title("🎨 AI 似てる度スコアゲーム")

difficulty = st.radio(
    "難易度",
    ["やさしい", "ふつう", "きびしい"],
    horizontal=True,
    index=1
)

ref = st.file_uploader("お題画像", type=["png", "jpg", "jpeg"], key="ref")
gen = st.file_uploader("あなたの画像", type=["png", "jpg", "jpeg"], key="gen")

if ref and gen:
    img1 = Image.open(ref).convert("RGB")
    img2 = Image.open(gen).convert("RGB")

    img1_c = center_crop(img1)
    img2_c = center_crop(img2)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="お題", use_container_width=True)

    with col2:
        st.image(img2, caption="あなたの画像", use_container_width=True)

    with st.spinner("採点中..."):
        c = clip_score(img1_c, img2_c)
        col = color_score(img1_c, img2_c)
        struct = structure_score(img1_c, img2_c)

        score = final_score(c, col, struct, difficulty)

    st.divider()

    st.header(f"🏆 スコア：{score:.1f}点")
    st.subheader(f"ランク：{rank(score)}")

    st.progress(int(score))

    st.divider()

    st.subheader("🔍 内訳")
    st.write(f"雰囲気・意味：{c:.2f}")
    st.write(f"色：{col:.2f}")
    st.write(f"構図：{struct:.2f}")

    st.divider()

    adjust = st.slider("手動補正", -15, 15, 0)
    final = max(0, min(100, score + adjust))

    st.header(f"最終スコア：{final:.1f}点")
