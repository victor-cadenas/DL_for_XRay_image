import streamlit as st
from PIL import Image
import numpy as np
from utils import load_model, generate_prediction, generate_heatmap, preprocess_image
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

path = os.path.join("models", "best_model.pth")

st.set_page_config(
    page_title="IA Radiología",
    layout="wide"
)

## CSS 
st.markdown("""
<style>

.main {
    background-color: #f5f7f9;
}

.block-container {
    padding-top: 2rem;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #007bff;
    color: white;
    font-weight: bold;
}

img {
    border-radius: 12px;
}
            
[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    font-size: 1.2rem !important;
}

[data-testid="stMetricValue"] {
    display: flex;
    justify-content: center;
    color: #007bff;
}

.stImage img {
    object-fit: cover;
}

</style>
""", unsafe_allow_html=True)

# Web

st.title("🏥 Sistema de Análisis de Radiografías")

if not os.path.exists(path):
    st.error("### 🚨 Modelo no encontrado")
    st.markdown(f"""
    Para ejecutar esta aplicación, por favor:
    1. Descarga el modelo desde este **https://upm365-my.sharepoint.com/:u:/g/personal/victor_cadenas_alumnos_upm_es/IQDyG5qZfNwETKkJd44FHfEDAWbWBZdfcLO9hJCph1uum2E?e=0B10yF**.
    2. Colócalo en la carpeta `/models/` del proyecto con el nombre `best_model.pth`.
    3. Refresca esta página.
    """)
    st.stop()

# Barra lateral para subir las fotos

with st.sidebar:

    st.header("Configuración")

    uploaded_file = st.file_uploader(
        "Elige una radiografía...",
        type=["jpg", "jpeg", "png"]
    )

    analizar_btn = st.button("Predecir")

# Main

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    
    img_display = img_pil.resize((500, 500)) 

    izq, col1, col2, der = st.columns([0.5, 2, 2, 0.5])

    with col1:
        st.subheader("Radiografía")
        st.image(img_display, use_container_width=True) 

    with col2:
        st.subheader("Mapa de calor")
        if analizar_btn:
            with st.spinner("Procesando..."):
                preprocessed_img = preprocess_image(img_pil)
                modelo = load_model(path, device)
                
                prediccion, probabilidad = generate_prediction(modelo, preprocessed_img, device)
                heatmap_overlay = generate_heatmap(modelo, preprocessed_img, device)
                
                heatmap_display = heatmap_overlay.resize((500, 500))
                st.image(heatmap_display, use_container_width=True) # 
        else:
            # Placeholder para mantener el equilibrio visual antes de pulsar el botón
            st.info("Pulsa 'Predecir' para ver el mapa de calor.")

    if analizar_btn:
        st.markdown("---") # Línea separadora
        # Usamos columnas laterales vacías para centrar el bloque de métricas
        _, center_col, _ = st.columns([1, 2, 1])
        
        with center_col:
            st.markdown("<h3 style='text-align: center;'>Resultado del Diagnóstico</h3>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Diagnóstico", prediccion)
            c2.metric("Confianza", f"{probabilidad:.2f}%")