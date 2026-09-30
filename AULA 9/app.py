import cv2
import numpy as np
import pyautogui
import streamlit as st
from ultralytics import YOLO

# Desativar pausas padrão do PyAutoGUI para maior fluidez
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True  # Mova o mouse para o canto superior esquerdo para interromper em emergência

# Configuração da página Streamlit
st.set_page_config(page_title="Controle de Mouse por Gestos - YOLO", layout="wide")
st.title("🖱️ Controle do Mouse em Tempo Real via Mão (YOLO Pose)")
st.caption("Visão Computacional + PyAutoGUI executando em CPU local")

# Obter dimensão da tela do monitor
screen_width, screen_height = pyautogui.size()

# Sidebar de Configurações
st.sidebar.header("⚙️ Configurações do Controle")
conf_threshold = st.sidebar.slider("Confiança do YOLO", 0.1, 1.0, 0.4, 0.05)
smoothing = st.sidebar.slider("Fator de Suavização (Smoothing)", 0.1, 0.9, 0.5, 0.05,
                              help="Valores maiores deixam o cursor mais estável, porém com leve atraso.")
frame_margin = st.sidebar.slider("Margem de Borda (Pixels)", 20, 150, 80, 10,
                                help="Área limite da câmera mapeada para as pontas da tela.")

enable_control = st.sidebar.checkbox("Ativar Controle do Mouse", value=True)

# 1. Carregamento do Modelo de Pose leve
@st.cache_resource
def load_pose_model():
    # yolov8n-pose detecta os 17 pontos articulares do corpo humano
    return YOLO("yolov8n-pose.pt")

model = load_pose_model()

# Placeholders do Streamlit para o feed e métricas
col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    st.subheader("📊 Métricas")
    metric_x = st.empty()
    metric_y = st.empty()
    st.info("💡 **Dica:** Levantar o pulso/mão direita controla o cursor no monitor.")

# Botão de Iniciar/Parar
run_app = st.checkbox("Ligar Câmera", value=True)

# Variáveis globais para suavização (Exponential Moving Average)
prev_x, prev_y = 0, 0

if run_app:
    # Captura da webcam local via OpenCV
    cap = cv2.VideoCapture(0)

    # Definir resolução da webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened() and run_app:
        success, frame = cap.read()
        if not success:
            st.error("Não foi possível acessar a câmera.")
            break

        # Espelhar imagem horizontalmente para navegação natural (modo espelho)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Inferência com YOLO Pose na CPU
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            imgsz=320,  # Redução para manter alto FPS na CPU
            device="cpu",
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Verificar se detectou algum corpo/keypoints
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Pegar keypoints da primeira pessoa detectada
            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            # No modelo YOLO Pose COCO:
            # Índice 10 = Pulso Direito (Right Wrist)
            # Índice 9  = Pulso Esquerdo (Left Wrist)
            if len(keypoints) > 10:
                wrist_x, wrist_y = keypoints[10]  # Pulso direito

                # Se o pulso for detectado com coordenadas válidas (>0)
                if wrist_x > 0 and wrist_y > 0:
                    # Desenhar ponto de destaque no pulso
                    cv2.circle(annotated_frame, (int(wrist_x), int(wrist_y)), 12, (0, 255, 0), -1)

                    # Mapeamento com margem de borda para alcance total da tela
                    target_x = np.interp(wrist_x, [frame_margin, w - frame_margin], [0, screen_width])
                    target_y = np.interp(wrist_y, [frame_margin, h - frame_margin], [0, screen_height])

                    # Aplicação do Filtro de Suavização (EMA)
                    curr_x = prev_x + (target_x - prev_x) * (1 - smoothing)
                    curr_y = prev_y + (target_y - prev_y) * (1 - smoothing)

                    # Mover o mouse via PyAutoGUI
                    if enable_control:
                        pyautogui.moveTo(int(curr_x), int(curr_y))

                    # Atualizar variáveis passadas
                    prev_x, prev_y = curr_x, curr_y

                    # Atualizar métricas na interface
                    metric_x.metric("Cursor X", f"{int(curr_x)} px")
                    metric_y.metric("Cursor Y", f"{int(curr_y)} px")

        # Desenhar caixa de margem/área útil na tela
        cv2.rectangle(annotated_frame, (frame_margin, frame_margin),
                      (w - frame_margin, h - frame_margin), (255, 255, 0), 2)

        # Converter BGR (OpenCV) para RGB (Streamlit)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.write("Câmera desligada. Marque 'Ligar Câmera' para iniciar.")