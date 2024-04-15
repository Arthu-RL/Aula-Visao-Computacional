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
# from utils.draw_kpts import draw_keypoints
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

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
weigths = torch.load('yolov7.pt', map_location=torch.device(device))
model = weigths['model']
model = model.to(device)
model = model.float().eval()

# model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True, trust_repo=True, force_reload=True).autoshape()
# model = model.to(device).eval()

print("Modelo carregado!")

# Abre um vídeo para processamento
video_path = './dataset/video_teste.mp4'
print("Abrindo vídeo:", video_path)
cap = cv2.VideoCapture(video_path)

# Verifica se o vídeo foi aberto com sucesso, senão, sai do programa
if cap.isOpened() == False:
    print("Video não foi aberto!")
    sys.exit(1)

lido, imagem_dimensionar_video = cap.read()

# Verifica se a imagem foi lida com sucesso, senão, sai do programa
if not lido:
    print("Frame não foi lido com sucesso!")
    sys.exit(1)

# Captura largura e altura do vídeo
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

dimensoes_reduzidas = 512   
if dimensoes_reduzidas < largura and dimensoes_reduzidas < altura:
    largura = dimensoes_reduzidas
    altura = dimensoes_reduzidas

# Criando uma imagem com resize para funcionar no modelo yolov7 pose usando LetterBox 
# LetterBox: função usada para mudar as dimensões da imagem, sem afetar a quantidade de detalhes na imagem
imagem_escrever_video = letterbox(imagem_dimensionar_video, (largura, altura), stride=64, auto=True)[0]

# Altura e largura da imagem que sobre resize
Altura, largura = imagem_escrever_video.shape[:2]

save_name = f"{video_path.split('/')[-1].split('.')[0]}"

# Criando um VideoWriter para escrever vídeo de inferência com dimensôes da imaem que sofreu resize
output_video = cv2.VideoWriter(f"./{save_name}_keypoints.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (largura, Altura))

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
    orig_image = frame

    # Passando imagem na LetterBox Para o Resize
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = letterbox(image, (largura, altura), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]), dtype=torch.float32)
    image = image.to(device)

    # Marca o tempo de início e posteriormente o fim da inferência para calcular FPS
    start_time = time.time()

    print("On frame:", frame_count)

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

    if 'nc' in model.yaml and 'nkpt' in model.yaml:
        # Escreve keypoints detectados em cada frame
        output = non_max_suppression_kpt(output, 
                                0.25, # Confidence Threshold
                                0.65, # IoU Threshold
                                nc=model.yaml['nc'], # Number of Classes
                                nkpt=model.yaml['nkpt'] , # Number of Keypoints
                                kpt_label=True)

        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)  

            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
            cv2.rectangle(
                nimg,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        # Escreve FPS em frame
        # image = image.numpy().astype(np.uint8)
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

print("Vídeo escrito com sucesso!")
print(f"Average FPS: {avg_fps:.1f}")
