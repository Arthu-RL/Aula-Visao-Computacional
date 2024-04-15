"""
Pytorch é uma biblioteca de código aberto que fornece uma ampla gama de algoritmos de Machine Learning.
Torchvision é uma biblioteca que pode ser usada para transformações de imagem comuns em visão computacional.
Letterbox é uma função definida no módulo datasets do pacote utils. É usado para redimensionar uma imagem para uma dimensão específica sem alterar o aspecto original da imagem.
"""

# importa bibliotecas necessárias, incluindo torch para deep learning e transforms para pré-processamento de imagem
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.draw_kpts import desenhar_keypoints

# importa bibliotecas para manipulação de imagens e gráficos
import numpy as np
import cv2

# importa bibliotecas para medição de tempo e interação com o sistema
import time
import sys

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # define o dispositivo de computação

print("Dispositivo:", device)

print("Carregando modelo...")

# Carrega o modelo YOLOv7 pré-treinado e o coloca em modo de avaliação
# modelo = torch.hub.load('WongKinYiu/yolov7', 'yolov7-w6-pose.pt', pretrained=True, trust_repo=True, force_reload=True).autoshape()
modelo = torch.load('yolov7-w6-pose.pt', map_location=torch.device(device))['model']
modelo = modelo.to(device).float().eval()

print("Modelo carregado!")

# Abre um vídeo para processamento
video_path = './dataset/video1.mp4'
print("Abrindo vídeo:", video_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Falha ao abrir o vídeo")
    exit(1)

# Lendo uma imagem para redimensiona-la
lido, imagem = cap.read()

if not lido:
    sys.exit(1)

# Redimensionando imagem, sem a afetar a quantidade de detalhes dela
imagem_reduzida = letterbox(imagem, 512, stride=64, auto=True)[0]
# Capturando dimensões para criar um VideoWriter com estas dimensões
altura, largura, _ = imagem_reduzida.shape

# Nome da imagem
nome_arquivo = f"{video_path.split('/')[-1].split('.')[0]}"

# Definindo codec como MP4V
codec = cv2.VideoWriter_fourcc(*'MP4V')

# Criando um VideoWriter para escrever vídeo de inferência com dimensôes da imaem que sofreu resize
output_video = cv2.VideoWriter(f"./{nome_arquivo}_keypoints.mp4", codec, 30, (largura, altura))

print("Vídeo foi aberto e suas informações como, altura e largura das imagens, foram definidas com sucesso!")

print("Começando inferência do modelo e escrita do vídeo de saída...")

# inicializa contadores
frame_count = 0 # Contador de frames
total_fps = 0 # Total de frames por segundo

while True:
    # Captura cada frame (frame) do video
    # ret é um bool que diz se o frame foi capturado ou não
    ret, frame = cap.read()

    if not ret:
        break
    
    # Capturando imagem original e convertendo seus canais para RGB
    # Passando imagem na LetterBox Para o Resize
    imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem_redimensionada = letterbox(imagem_rgb, 512, stride=64, auto=True)[0]
    imagem_tensor = transforms.ToTensor()(imagem_redimensionada).unsqueeze(0).to(device).float()

    # Marca o tempo de início e posteriormente o fim da inferência para calcular FPS
    start_time = time.time()

    print("On frame:", frame_count)

    # Realiza a detecção de pose usando o modelo YOLOv7
    with torch.no_grad():
        """
        model(image) -> retorna, coordenadas das bounding boxes, class predictions (previsões), e
        confidencia (float) para cada objeto detectado na imagem
        """
        saida, _ = modelo(imagem_tensor)

    end_time = time.time()

    # calculando FPS
    fps = 1 / (end_time - start_time)
    total_fps += fps

    frame_count += 1

    # Escreve keypoints detectados em cada frame
    imagem_com_kpts = desenhar_keypoints(modelo, saida, imagem_redimensionada)

    # Escreve FPS em frame
    # image = image.numpy().astype(np.uint8)
    cv2.putText(imagem_com_kpts, f"{fps:.1f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Escreve imagem no vídeo de output
    output_video.write(imagem_com_kpts)

# Libera captura do video de output
cap.release()

# Fecha todos os frames e janelas do video
cv2.destroyAllWindows()

# Calcula e retorna o FPS
avg_fps = total_fps / frame_count

print("Vídeo escrito com sucesso!")
print(f"Average FPS: {avg_fps:.1f}")
