# non_max_suppression_kpt função é uma técnica usada para suprimir (ou seja, remover) Bounding Boxes redundantes em uma imagem após a detecção de objetos.
# output_to_keypoint e plot_skeleton_kpts são funções usadas para converter a saída do modelo em keypoints e escrever o esqueleto baseado em keypoints na imagem.
from torch import no_grad
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import cv2
from numpy import uint8


def desenhar_keypoints(modelo,  saida, imagem): 
    saida = non_max_suppression_kpt(saida, 
                                        0.25, # Confidência da detecção
                                        0.65, # IoU (Interseção sobre União) Threshold (Tirar bounding boxes reduntes)
                                        nc=modelo.yaml['nc'], # Número de classes
                                        nkpt=modelo.yaml['nkpt'], # Número de Keypoints
                                        kpt_label=True)
    
    with no_grad():
        saida = output_to_keypoint(saida) # Converte para formato -> [batch_id, class_id, x, y, w, h, conf]

    imagem_kpts = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR) # TO COMMENT
    
    for idx in range(saida.shape[0]):
        plot_skeleton_kpts(imagem_kpts, saida[idx, 7:].T, 3) # Desenha os keypoints na imagem

    return imagem_kpts