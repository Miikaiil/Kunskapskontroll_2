import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

# --- 1. LADDA ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load('mnist_voting_model_final.pkl')
    scaler = joblib.load('mnist_scaler_final.pkl')
    return model, scaler

model, scaler = load_assets()

# --- 2. DYNAMISKT TEMA (COPPER & NEON) ---
st.set_page_config(page_title="CICEK INSIGHT LAB", layout="wide")
st.markdown("""
<style>
    :root { --neon: #00D4FF; --copper: #B87333; --bg: #050A18; }
    .stApp { background-color: var(--bg); color: white; font-family: 'JetBrains Mono', monospace; }
    
    .glass-card {
        background: rgba(10, 20, 40, 0.7); border: 1px solid rgba(0,212,255,0.3);
        border-radius: 12px; padding: 25px; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .header-text { color: var(--neon); font-style: italic; font-weight: 800; font-size: 3rem; margin: 0; }
    .copper-text { color: var(--copper); font-weight: 100; font-size: 1.5rem; }
    .digit-btn {
        background: transparent; border: 2px solid var(--copper); color: var(--copper);
        border-radius: 50%; width: 60px; height: 60px; font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. UI LAYOUT ---
st.markdown(f'<p class="header-text">CICEK INSIGHT <span class="copper-text">| FEATURE HEATMAP</span></p>', unsafe_allow_html=True)

# Bred ritruta (Vit yta för stabilitet, matchar din fungerande logik)
st.markdown('<div class="glass-card" style="background:#FFFFFF; padding:5px;">', unsafe_allow_html=True)
canvas_result = st_canvas(
    stroke_width=20, stroke_color="#000000", background_color="#FFFFFF",
    height=250, width=1100, drawing_mode="freedraw", key="copper_canvas"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- 4. MODELLLOGIK ---
if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
    img_raw = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img_np = np.array(img_raw)
    _, thresh = cv2.threshold(255 - img_np, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    if contours:
        # Skapa en horisontell rad av detekterade siffror
        st.markdown('<p style="color:gray; font-size:12px; margin-top:10px;">SEKVENS DETEKTERAD - KLICKA FÖR ATT ANALYSERA:</p>', unsafe_allow_html=True)
        digit_cols = st.columns(len(contours))
        
        if 'active_idx' not in st.session_state: st.session_state.active_idx = 0
        sequence_data = []

        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 5 or h < 5: continue
            
            # ROI Preprocessing
            roi = thresh[y:y+h, x:x+w]
            pad = max(w, h) + 20
            sq = np.zeros((pad, pad), dtype=np.uint8)
            sq[(pad-h)//2 : (pad-h)//2+h, (pad-w)//2 : (pad-w)//2+w] = roi
            mnist = Image.fromarray(sq).resize((20, 20), Image.LANCZOS)
            final = Image.new('L', (28, 28), 0)
            final.paste(mnist, (4, 4))
            
            arr = np.array(final).reshape(1, -1).astype(float)
            scaled = scaler.transform(arr)
            pred = model.predict(scaled)[0]
            probs = model.predict_proba(scaled)[0]
            
            sequence_data.append({"val": pred, "img": arr.reshape(28,28), "prob": np.max(probs)})

            with digit_cols[i]:
                if st.button(f"{pred}", key=f"d_{i}"):
                    st.session_state.active_idx = i

        # --- 5. VISUALISERINGS-SECTION ---
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_info, col_heat, col_arch = st.columns([1, 1, 1.2])
        
        selected = sequence_data[st.session_state.active_idx]
        
        with col_info:
            st.markdown(f"<p class='copper-text' style='margin:0;'>ANALYS: SIFFRA {selected['val']}</p>", unsafe_allow_html=True)
            st.metric("Confidence Score", f"{selected['prob']*100:.1f}%")
            st.markdown(f"""
                <div style="font-size:11px; color:gray; line-height:1.5;">
                    Voter Ensemble har analyserat 784 pixel-ingångar. 
                    Denna heatmap visualiserar 'Pixel Importance' med din Copper-palett.
                </div>
            """, unsafe_allow_html=True)

        with col_heat:
            # COPPER HEATMAP (Plotly Heatmap med anpassad färgskala)
            # Vi skapar en färgskala som går från mörkt blått till din Copper (#B87333)
            fig_heat = px.imshow(selected['img'], 
                                color_continuous_scale=[[0, '#050A18'], [0.5, '#5D3A1A'], [1, '#B87333']])
            fig_heat.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,b=0,t=0), height=200, width=200,
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat, config={'displayModeBar': False})
            st.markdown("<p style='text-align:center; font-size:9px; color:gray;'>NEURAL FEATURE ACTIVATION</p>", unsafe_allow_html=True)

        with col_arch:
            # HÄFTIGT VERKTYG: Ensemble Architecture Visualization
            # Här visar vi hur röstningen fördelas (Top 3 utmanare)
            st.markdown("<p style='font-size:10px; color:var(--neon); letter-spacing:2px;'>VOTING ARCHITECTURE</p>", unsafe_allow_html=True)
            top_3_idx = np.argsort(model.predict_proba(scaler.transform(selected['img'].reshape(1,-1)))[0])[-3:][::-1]
            for idx in top_3_idx:
                v = model.predict_proba(scaler.transform(selected['img'].reshape(1,-1)))[0][idx]
                st.write(f"Class {idx}:")
                st.progress(v)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div style="margin-top:100px; text-align:center; opacity:0.3; letter-spacing:10px;">AWAITING NEURAL SCAN...</div>', unsafe_allow_html=True)