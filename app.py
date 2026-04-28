import streamlit as st
import torch
import open_clip
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="イラスト再現スコアゲーム",
    layout="wide"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# モデル読み込み
# =========================
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    return model, preprocess


model, preprocess = load_model()


# =========================
# 前処理
# =========================
def center_crop(img, size=256):
    w, h = img.size
    m = min(w, h)

    left = (w - m) // 2
    top = (h - m) // 2

    img = img.crop((left, top, left + m, top + m))
    return img.resize((size, size))


def safe_corr(a, b):
    a = a.flatten()
    b = b.flatten()

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    v = np.corrcoef(a, b)[0, 1]

    if np.isnan(v):
        return 0.0

    return float(v)


def blur_array(img, k=21):
    arr = np.array(img)
    return cv2.GaussianBlur(arr, (k, k), 0)


# =========================
# 各スコア
# =========================
def clip_score(img1, img2):
    i1 = preprocess(img1).unsqueeze(0).to(device)
    i2 = preprocess(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        f1 = model.encode_image(i1)
        f2 = model.encode_image(i2)

        f1 = f1 / f1.norm(dim=-1, keepdim=True)
        f2 = f2 / f2.norm(dim=-1, keepdim=True)

        score = (f1 @ f2.T).item()

    return float(score)


def color_score(img1, img2):
    i1 = np.array(img1)
    i2 = np.array(img2)

    hist1 = cv2.calcHist(
        [i1], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    hist2 = cv2.calcHist(
        [i2], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return max(0.0, min(1.0, float(score)))


def structure_score(img1, img2):
    i1 = blur_array(img1)
    i2 = blur_array(img2)

    g1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)

    score = safe_corr(g1, g2)

    return max(0.0, min(1.0, score))


def edge_score(img1, img2):
    i1 = blur_array(img1)
    i2 = blur_array(img2)

    g1 = cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)

    e1 = cv2.Canny(g1, 50, 150)
    e2 = cv2.Canny(g2, 50, 150)

    score = safe_corr(e1, e2)

    return max(0.0, min(1.0, score))


# =========================
# ゲーム用スコア調整
# =========================
def final_score(c, col, edge, struct, difficulty):
    """
    c      : 意味・雰囲気
    col    : 色
    edge   : 輪郭
    struct : 構図
    """

    # 輪郭は不安定なのでほぼ使わない
    raw = (
        0.76 * c +
        0.14 * col +
        0.10 * struct
    ) * 100

    # 遊びやすくするための基礎補正
    score = raw + 5

    # 意味が強く似ている場合、極端に低くなりすぎないようにする
    if c >= 0.85:
        score = max(score, 50)
    elif c >= 0.80:
        score = max(score, 45)
    elif c >= 0.75:
        score = max(score, 38)

    # 意味だけ似ていて、色も構図もかなり違う場合は減点
    if c >= 0.80 and col < 0.15 and struct < 0.12:
        score -= 12

    # 色が極端に違う場合
    if col < 0.05:
        score -= 5

    # 構図がかなり違う場合
    if struct < 0.08:
        score -= 5

    # 意味が弱い場合は厳しめ
    if c < 0.65:
        score -= 12

    # 難易度補正
    if difficulty == "やさしい":
        score += 8
    elif difficulty == "ふつう":
        score += 0
    elif difficulty == "きびしい":
        score -= 8

    return max(0, min(100, score))


def rank_comment(score):
    if score >= 90:
        return "SS", "ほぼ完全再現！これはかなり強い。"
    elif score >= 80:
        return "S", "かなり似ています。ゲーム的には高得点です。"
    elif score >= 70:
        return "A", "いい感じ！雰囲気はかなり近いです。"
    elif score >= 60:
        return "B", "そこそこ似ています。色や構図を寄せると伸びます。"
    elif score >= 45:
        return "C", "方向性は近いけど、再現度はまだ弱めです。"
    else:
        return "D", "別物寄りです。特徴をもっと寄せる必要があります。"


def advice(c, col, edge, struct):
    tips = []

    if c < 0.75:
        tips.append("キャラ・物体・テーマなど、画像の中心要素をもっと近づけると伸びます。")

    if col < 0.30:
        tips.append("色味がかなり違います。背景色やメインカラーを寄せると点が上がります。")

    if struct < 0.25:
        tips.append("構図がズレています。顔や体、物体の位置を近づけると改善します。")

    if edge < 0.10:
        tips.append("輪郭は一致していません。ただしこの項目は不安定なので参考程度でOKです。")

    if not tips:
        tips.append("かなりバランスよく似ています。細部を寄せるとさらに高得点を狙えます。")

    return tips


# =========================
# UI
# =========================
st.title("🎨 イラスト再現スコアゲーム")

st.write(
    "お題画像と再現画像を比べて、雰囲気・色・構図からスコアを出します。"
)

difficulty = st.radio(
    "難易度",
    ["やさしい", "ふつう", "きびしい"],
    horizontal=True,
    index=1
)

ref = st.file_uploader(
    "お題画像をアップロード",
    type=["png", "jpg", "jpeg"],
    key="ref"
)

gen = st.file_uploader(
    "再現画像をアップロード",
    type=["png", "jpg", "jpeg"],
    key="gen"
)

if ref and gen:
    img1 = Image.open(ref).convert("RGB")
    img2 = Image.open(gen).convert("RGB")

    img1_c = center_crop(img1)
    img2_c = center_crop(img2)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="お題画像", use_container_width=True)

    with col2:
        st.image(img2, caption="再現画像", use_container_width=True)

    with st.spinner("採点中..."):
        c = clip_score(img1_c, img2_c)
        col = color_score(img1_c, img2_c)
        edge = edge_score(img1_c, img2_c)
        struct = structure_score(img1_c, img2_c)

        score = final_score(c, col, edge, struct, difficulty)

    st.divider()

    rank, comment = rank_comment(score)

    st.header(f"🏆 スコア：{score:.1f}点")
    st.subheader(f"ランク：{rank}")
    st.write(comment)

    st.progress(int(score))

    st.divider()

    st.subheader("🔍 内訳")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("雰囲気・意味", f"{c:.2f}")

    with m2:
        st.metric("色", f"{col:.2f}")

    with m3:
        st.metric("構図", f"{struct:.2f}")

    with m4:
        st.metric("輪郭", f"{edge:.2f}")

    st.caption("※ 輪郭スコアは画像の位置ズレに弱いため、最終点にはほぼ使っていません。")

    st.divider()

    st.subheader("💬 改善アドバイス")

    for t in advice(c, col, edge, struct):
        st.write("・" + t)

    st.divider()

    st.subheader("🎛 手動補正")

    adjust = st.slider(
        "人間の目で見て違和感がある場合は補正",
        -15,
        15,
        0
    )

    final = max(0, min(100, score + adjust))

    st.header(f"最終スコア：{final:.1f}点")