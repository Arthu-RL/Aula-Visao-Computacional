"""
Pytorch é uma biblioteca de código aberto que fornece uma ampla gama de algoritmos de Machine Learning.
Torchvision é uma biblioteca que pode ser usada para transformações de imagem comuns em visão computacional.
Letterbox é uma função definida no módulo datasets do pacote utils. É usado para redimensionar uma imagem para uma dimensão específica sem alterar o aspecto original da imagem.
non_max_suppression_kpt função é uma técnica usada para suprimir (ou seja, remover) Bounding Boxes redundantes em uma imagem após a detecção de objetos.
output_to_keypoint e plot_skeleton_kpts são funções usadas para converter a saída do modelo em keypoints e escrever o esqueleto baseado em keypoints na imagem.
"""

# importa bibliotecas necessárias, incluindo torch para deep learning e transforms para pré-processamento de imagem
import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.draw_kpts import draw_keypoints

# importa bibliotecas para manipulação de imagens e gráficos
import numpy as np
import cv2

# importa bibliotecas para medição de tempo e interação com o sistema
import time
import sys

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # define o dispositivo de computação

# Carrega o modelo YOLOv7 pré-treinado e o coloca em modo de avaliação
weigths = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
model = weigths['model']
model = model.to(device).eval()

# Abre um vídeo para processamento
video_path = '../inference_data/video_5.mp4'
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo foi aberto com sucesso, senão, sai do programa
if cap.isOpened() == False:
    print("Video não foi aberto!")
    sys.exit(1)

lido, imagem_dimensionar_video = cap.read()[1]

# Verifica se a imagem foi lida com sucesso, senão, sai do programa
if not lido:
    print("Frame não foi lido com sucesso!")
    sys.exit(1)

# Captura largura e altura do vídeo
largura = int(cap.get(3))
altura = int(cap.get(4))

# Criando uma imagem com resize para funcionar no modelo yolov7 pose usando LetterBox 
# LetterBox: função usada para mudar as dimensões da imagem, sem afetar a quantidade de detalhes na imagem
imagem_escrever_video = letterbox(imagem_dimensionar_video, (largura), stride=64, auto=True)[0]

# Altura e largura da imagem que sobre resize
Altura, largura = imagem_escrever_video.shape[:2]

save_name = f"{video_path.split('/')[-1].split('.')[0]}"

# Criando um VideoWriter para escrever vídeo de inferência com dimensôes da imaem que sofreu resize
output_video = cv2.VideoWriter(f"{save_name}_keypoint.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (largura, Altura))

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
    orig_image = frame
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Passando imagem na LetterBox Para o Resize
    image = letterbox(image, (largura), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    image = image.to(device)
    image = image.half()

    # Marca o tempo de início e posteriormente o fim da inferência para calcular FPS
    start_time = time.time()

    # Realiza a detecção de pose usando o modelo YOLOv7
    with torch.no_grad():
        """
        model(image) -> retorna, coordenadas das bounding boxes, class predictions (previsões), e
        confidencia (float) para cada objeto detectado na imagem
        """
        output, _ = model(image)

    end_time = time.time()

    # calculando FPS
    fps = 1 / (end_time - start_time)
    total_fps += fps

    frame_count += 1

    # Escreve keypoints detectados em cada frame
    nimg = draw_keypoints(model, output, image)

    # Escreve FPS em frame
    cv2.putText(nimg, f"{fps:.1f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    
    # Escreve imagem no vídeo de output
    output_video.write(nimg)

# Libera captura do video de output
cap.release()

# Fecha todos os frames e janelas do video
cv2.destroyAllWindows()

# Calcula e retorna o FPS
avg_fps = total_fps / frame_count

print(f"Average FPS: {avg_fps:.1f}")
