import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import spacy
from ultralytics import YOLO


class YoloScannerApp:
    """
    Aplicações Tkinter com suporte a processamento em background (Threading)
    para detecção de objetos via YOLOv8 e análise de NLP com spaCy.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Scanner Inteligente com YOLO & spaCy")
        self.root.geometry("850x680")
        self.root.config(bg="#f0f2f5")

        self.center_window()

        # Variáveis de controle
        self.yolo_model = None
        self.nlp = None
        self.image_path = None

        self.create_widgets()
        
        # Carrega os modelos em background para não congelar a abertura do app
        threading.Thread(target=self.load_models, daemon=True).start()

    def center_window(self):
        """Centraliza a janela principal na tela."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def load_models(self):
        """Carrega os modelos de IA assincronamente."""
        self.update_status("Carregando modelos de IA, aguarde...")
        
        # 1. Modelo YOLO
        try:
            self.yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro YOLO", f"Falha ao carregar YOLO: {e}"))

        # 2. Modelo spaCy
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except Exception:
            try:
                self.nlp = spacy.blank("pt")
            except Exception:
                self.nlp = None

        self.update_status("Modelos carregados com sucesso! Selecione uma imagem.")
        self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))

    def create_widgets(self):
        """Cria os componentes gráficos da aplicação."""
        title_label = tk.Label(
            self.root,
            text="Scanner Inteligente com YOLO & spaCy",
            font=("Arial", 16, "bold"),
            bg="#f0f2f5",
            fg="#333333"
        )
        title_label.pack(pady=(15, 5))

        # Status Bar / Label informativo
        self.status_label = tk.Label(
            self.root,
            text="Iniciando aplicação...",
            font=("Arial", 9, "italic"),
            bg="#f0f2f5",
            fg="#666666"
        )
        self.status_label.pack(pady=(0, 10))

        # Botão para carregar imagem (desabilitado até carregar modelos)
        self.btn_load = tk.Button(
            self.root,
            text="Carregar e Analisar Imagem",
            command=self.on_click_load,
            font=("Arial", 11, "bold"),
            bg="#007bff",
            fg="white",
            activebackground="#0056b3",
            activeforeground="white",
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        self.btn_load.pack(pady=5)

        # Painel de Imagem
        self.image_label = tk.Label(self.root, bg="#e0e0e0", width=500, height=300)
        self.image_label.pack(pady=10)

        # Caixa de Texto
        self.result_text = tk.Text(
            self.root,
            height=9,
            width=90,
            font=("Consolas", 10),
            bg="white",
            fg="#333333"
        )
        self.result_text.pack(pady=10)
        self.result_text.insert(tk.END, "Aguardando carregamento de imagem...")
        self.result_text.config(state=tk.DISABLED)

    def update_status(self, message):
        """Atualiza a mensagem de status da interface."""
        self.root.after(0, lambda: self.status_label.config(text=message))

    def on_click_load(self):
        """Abre o seletor de arquivo e dispara o processamento em thread separada."""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.btn_load.config(state=tk.DISABLED)
        self.update_status("Processando imagem...")
        
        # Executa a inferência em background
        threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()

    def process_image(self, file_path):
        """Executa a inferência com YOLO e a análise do spaCy."""
        try:
            # 1. Predição com YOLO
            results = self.yolo_model(file_path)
            r = results[0]

            # Renderiza a imagem com os bounding boxes (retorna em BGR do OpenCV)
            im_array = r.plot()
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(im_rgb)
            img.thumbnail((500, 300))
            img_tk = ImageTk.PhotoImage(img)

            # Extração dos objetos detectados
            detected_objects = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])
                detected_objects.append(f"{class_name} ({conf:.2f})")

            # 2. Montagem do Texto
            if detected_objects:
                description = f"Elementos detectados: {', '.join(detected_objects)}."
            else:
                description = "Nenhum objeto reconhecido na imagem."

            # 3. Análise NLP com spaCy
            nlp_analysis = ""
            if self.nlp:
                doc = self.nlp(description)
                tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
                nlp_analysis = f"\n[Tokens Relevantes (spaCy)]: {', '.join(tokens)}"

            # 4. Atualização da Interface (sincronizada na Thread Principal)
            self.root.after(0, self.update_ui_results, img_tk, description, nlp_analysis)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Erro de Processamento", str(e)))
            self.update_status("Erro ao processar imagem.")
            self.root.after(0, lambda: self.btn_load.config(state=tk.NORMAL))

    def update_ui_results(self, img_tk, description, nlp_analysis):
        """Atualiza a interface gráfica após a conclusão do processamento."""
        self.image_label.config(image=img_tk, text="")
        self.image_label.image = img_tk  # Previne garbage collection do PIL

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(
            tk.END, 
            f"--- RELATÓRIO DE DETECÇÃO ---\n{description}\n{nlp_analysis}"
        )
        self.result_text.config(state=tk.DISABLED)

        self.update_status("Análise concluída!")
        self.btn_load.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = YoloScannerApp(root)
    root.mainloop()
    