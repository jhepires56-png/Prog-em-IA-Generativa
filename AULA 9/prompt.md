import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import spacy
from ultralytics import YOLO
import pyautogui
import pygame

# -----------------------------------------------------------------------------
# Configuração da Janela Principal do Tkinter
# -----------------------------------------------------------------------------
root = tk.Tk()
root.title("Scanner com Yolo")
root.geometry("800x700")

# Centralização da janela na tela
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f"{width}x{height}+{x}+{y}")

# -----------------------------------------------------------------------------
# Carregamento dos Modelos (YOLO e SpaCy)
# -----------------------------------------------------------------------------
# Carrega o modelo pré-treinado do YOLOv8
yolo_model = YOLO("yolov8n.pt")

# Carrega o modelo de PNL SpaCy para processamento de texto
try:
    nlp_model = spacy.load("pt_core_news_sm")
except OSError:
    nlp_model = spacy.blank("pt")

# Variável global para armazenar o caminho da imagem selecionada
selected_image_path = None

# -----------------------------------------------------------------------------
# Interface Gráfica (Componentes do Tkinter)
# -----------------------------------------------------------------------------
title_label = tk.Label(root, text="Scanner com Yolo", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

input_label = tk.Label(root, text="O que você gostaria de ver na imagem?", font=("Helvetica", 12))
input_label.pack(pady=5)

text_entry = tk.Entry(root, width=50, font=("Helvetica", 11))
text_entry.pack(pady=5)

image_label = tk.Label(root, text="Nenhuma imagem carregada.", bg="lightgray", width=60, height=15)
image_label.pack(pady=10)

# -----------------------------------------------------------------------------
# Funções de Manipulação e Processamento
# -----------------------------------------------------------------------------
def select_image():
    """
    Abre o seletor de arquivos para escolher uma imagem e exibe a prévia no Tkinter.
    """
    global selected_image_path
    path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
    if path:
        selected_image_path = path
        img = Image.open(path)
        img.thumbnail((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk, text="")
        image_label.image = img_tk


def process_and_draw():
    """
    Executa a detecção do YOLO, filtra pelo input com SpaCy,
    gera a captura automatizada via pyautogui e desenha na tela via Pygame.
    """
    if not selected_image_path:
        messagebox.showwarning("Aviso", "Por favor, selecione uma imagem primeiro.")
        return

    # Processamento do texto de busca do usuário usando SpaCy
    user_text = text_entry.get().strip().lower()
    target_labels = []
    if user_text:
        doc = nlp_model(user_text)
        target_labels = [
            token.text for token in doc 
            if token.pos_ in ["NOUN", "PROPN"] or token.text in yolo_model.names.values()
        ]

    # Execução da inferência de detecção com o YOLO
    image = Image.open(selected_image_path).convert("RGB")
    results = yolo_model(image)[0]

    # Anotação e desenho das caixas delimitadoras
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for box, cls_idx in zip(results.boxes.xyxy, results.boxes.cls):
        class_name = results.names[int(cls_idx)]
        if not target_labels or any(label in class_name.lower() for label in target_labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 10)), class_name, fill="red")

    # Atualiza a interface do Tkinter com a imagem anotada
    preview_img = annotated_image.copy()
    preview_img.thumbnail((400, 300))
    img_tk = ImageTk.PhotoImage(preview_img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    root.update()

    # Automação de captura do frame utilizando PyAutoGUI
    temp_path = "temp_detected.png"
    annotated_image.save(temp_path)
    
    # Captura de tela automatizada da área do programa
    screenshot = pyautogui.screenshot(region=(root.winfo_x(), root.winfo_y(), width, height))

    # Renderização da imagem processada na tela através do Pygame
    pygame.init()
    screen = pygame.display.set_mode((annotated_image.width, annotated_image.height))
    pygame.display.set_caption("Detecção do Input - Renderização Pygame")

    pygame_surface = pygame.image.load(temp_path)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(pygame_surface, (0, 0))
        pygame.display.flip()

    pygame.quit()
    
    # Limpeza do arquivo temporário
    if os.path.exists(temp_path):
        os.remove(temp_path)


# -----------------------------------------------------------------------------
# Botões de Ação
# -----------------------------------------------------------------------------
btn_select = tk.Button(root, text="Selecionar Imagem", command=select_image, font=("Helvetica", 10))
btn_select.pack(pady=5)

btn_process = tk.Button(root, text="Detectar e Desenhar (Pygame)", command=process_and_draw, font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white")
btn_process.pack(pady=10)

# Inicialização do loop principal do Tkinter
if __name__ == "__main__":
    root.mainloop()
