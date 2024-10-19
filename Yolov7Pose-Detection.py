"""
Pytorch é uma biblioteca de código aberto que fornece uma ampla gama de algoritmos de Machine Learning.
Torchvision é uma biblioteca que pode ser usada para transformações de imagem comuns em visão computacional.
Letterbox é uma função definida no módulo datasets do pacote utils. É usado para redimensionar uma imagem para uma dimensão específica sem alterar o aspecto original da imagem.
"""

# importa bibliotecas necessárias, incluindo torch para deep learning e transforms para pré-processamento de imagem
import torch
from torchvision import transforms

# from urllib.parse import urlparse
from typing import Any, Tuple, Union, Optional
import argparse
import logging as log
import os

# from utils.datasets import letterbox
from utils.draw_kpts import desenhar_keypoints
from utils.outputvideowriter import OutputVideoWriter
from utils.inputvideocapture import InputVideoCapture, StreamType
# from models.yolo import Model

# importa bibliotecas para manipulação de imagens e gráficos
import numpy as np
import cv2

# importa bibliotecas para medição de tempo e interação com o sistema
import time

log.basicConfig(level=log.INFO, format='%(levelname)s: %(message)s')

def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value

def tensor_float16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float16)

def tensor_float32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float32)

parser: argparse.ArgumentParser = argparse.ArgumentParser(description="YoloV7Pose Detection")

parser.add_argument('-s', '--src', dest='src', type=int_or_str, help='Caminho para o vídeo', default='./dataset/video0.mp4')
parser.add_argument('-t', '--stream_type', dest='stream_type', type=int, help='Tipo de streaming', default=StreamType.FILE.value)
parser.add_argument('-l', '--width', dest='width', type=int, help='Largura de cada frame', default=1024)
parser.add_argument('-a', '--height', dest='height', type=int, help='Altura de cada frame', default=768)
parser.add_argument('-o', '--output', dest='output', action="store_true", help='Se teremos escrita de vídeo', required=False)
parser.add_argument('-n', '--window_name', dest='window_name', type=str, help='Nome da janela', default='Video Streaming')

args: argparse.Namespace = parser.parse_args()

src: Union[int, str] = args.src
stream_type: int = args.stream_type
size: Tuple[int, int] = (args.width, args.height)
output: bool = args.output
win_name: str = args.window_name

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu: bool = device.type == "cuda"
# device = torch.device("cpu") # define o dispositivo de computação

log.info(f"Dispositivo: {device}")

log.info("Carregando modelo...")

# Carrega o modelo YOLOv7 pré-treinado e o coloca em modo de avaliação
# modelo = torch.hub.load('WongKinYiu/yolov7', 'yolov7-w6-pose.pt', pretrained=True, trust_repo=True, force_reload=True).autoshape()
# torch.serialization.add_safe_globals([Model])
modelo = torch.load('yolov7-w6-pose.pt', map_location=torch.device(device), weights_only=False)['model']
modelo = modelo.to(device)

tensor_type_callback: callable = None

if gpu:
    modelo = modelo.half()
    tensor_type_callback = tensor_float16
else:
    modelo = modelo.float()
    tensor_type_callback = tensor_float32

modelo = modelo.eval()

log.info("Modelo carregado!")

cap: InputVideoCapture = InputVideoCapture(src=src, stream_type=stream_type, resize=size)
out_video_writer: Optional[OutputVideoWriter] = None

if output:
    os.makedirs("./output", exist_ok=True)
    out_video_writer = OutputVideoWriter(output_path="./output", fps=cap.fps, size=size)


# lido, imagem = cap.read()

log.info("Começando inferência do modelo e escrita do vídeo de saída...")

# inicializa contadores
frame_count = 0
# total_fps = 0

cv2.namedWindow(win_name)

while True:
    # Captura cada frame (frame) do video
    # ret é um bool que diz se o frame foi capturado ou não
    ret, frame = cap.read()

    if not ret:
        break
    
    # # Capturando imagem original e convertendo seus canais para RGB
    # # Passando imagem na LetterBox Para o Resize
    # imagem_redimensionada = letterbox(frame, 1024, stride=32, auto=False)[0] # shape: (567, 960, 3) HWC

    # tensor -> # torch.Size([3, 567, 960]) CHW
    # unsqueeze(0) -> transformação para batch (lote), torch.Size([1, 3, 567, 960]) 1 -> tamanho lote 1 imagem
    # Float() -> float32, aumenta a precisão dos números, o que é bom para CPU
    imagem_tensor: torch.Tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
    imagem_tensor: torch.Tensor = tensor_type_callback(tensor=imagem_tensor)

    # Marca o tempo de início e posteriormente o fim da inferência para calcular FPS
    start_time = time.time()

    # log.info(f"Frame {frame_count}")

    # Realiza a detecção de pose usando o modelo YOLOv7
    with torch.no_grad():
        """
        model(image) -> retorna, coordenadas das bounding boxes, class predictions (previsões), e
        confidencia (float) para cada objeto detectado na imagem
        """
        saida, _ = modelo(imagem_tensor)

    end_time = time.time()

    # # calculando FPS
    
    # total_fps += fps

    frame_count += 1

    # Escreve keypoints detectados em cada frame
    imagem_com_kpts = desenhar_keypoints(modelo, saida, frame)

    fps = 1 / (end_time - start_time)

    # Escreve FPS em frame
    # image = image.numpy().astype(np.uint8)
    cv2.putText(imagem_com_kpts, f"{fps:.1f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow(win_name, imagem_com_kpts)

    # Escreve imagem no vídeo de output
    if output:
        out_video_writer.write(np.ndarray(imagem_com_kpts))
    
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        log.info(f"Finished! Frames processed {frame_count}")
        if stream_type == StreamType.RTSP.value:
            cap.destroy()
        break
    elif key == ord('p'):
        cv2.imwrite(f"opencv_frame_{frame_count}.png", imagem_com_kpts)

# Libera captura do video de output
cap.destroy()

# Fecha todos os frames e janelas do video
cv2.destroyAllWindows()

# Calcula e retorna o FPS
# avg_fps = total_fps / frame_count

log.info("Vídeo processado com sucesso!")
# log.info(f"Average FPS: {avg_fps:.1f}")
