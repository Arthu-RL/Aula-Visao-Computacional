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


def carregar_modelo(dispositivo):
    model = torch.load('yolov7-w6-pose.pt', map_location=dispositivo)['model']
    # Coloca no modo inferência (avaliação)
    model.float().eval()

    if torch.cuda.is_available():
        # half() deixa as previsões para float16, o que torna a inferência mais eficiente
        model.half().to(dispositivo)

    return model


def avaliacao(modelo, dispositivo, imagem):
    # Redimensionar a imagem para uma dimensão específica sem alterar o aspecto original da imagem.
    imagem = letterbox(imagem, 960, stride=64, auto=True)[0] # shape: (567, 960, 3) HWC

    # Transformação para tensor
    imagem = transforms.ToTensor()(imagem) # torch.Size([3, 567, 960]) CHW

    if torch.cuda.is_available():
        # half() deixa as previsões para float16, o que torna a inferência mais eficiente
        imagem = imagem.half().to(dispositivo)

    # transformação para batch (lote)
    imagem = imagem.unsqueeze(0) # torch.Size([1, 3, 567, 960]) 1 -> tamanho lote 1 imagem
    
    with torch.no_grad():
        saida, _ = modelo(imagem)

    return saida, imagem


def desenhar_keypoints(modelo, saida, imagem):
    saida = non_max_suppression_kpt(saida, 
                                        0.25, # Confidência da detecção
                                        0.65, # IoU (Interseção sobre União) Threshold (Tirar bounding boxes reduntes)
                                        nc=modelo.yaml['nc'], # Número de classes
                                        nkpt=modelo.yaml['nkpt'], # Número de Keypoints
                                        kpt_label=True)
    
    with torch.no_grad():
        saida = output_to_keypoint(saida) # Converte para formato -> [batch_id, class_id, x, y, w, h, conf]

    imagem_kpts = imagem[0].permute(1, 2, 0) * 255 # TO COMMENT
    imagem_kpts = imagem_kpts.cpu().numpy().astype(np.uint8) # TO COMMENT
    imagem_kpts = cv2.cvtColor(imagem_kpts, cv2.COLOR_RGB2BGR) # TO COMMENT
    
    for idx in range(saida.shape[0]):
        plot_skeleton_kpts(imagem_kpts, saida[idx, 7:].T, 3) # Desenha os keypoints na imagem

    return imagem_kpts


