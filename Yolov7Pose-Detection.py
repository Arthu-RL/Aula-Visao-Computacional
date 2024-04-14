"""
Pytorch é uma biblioteca de código aberto que fornece uma ampla gama de algoritmos de Machine Learning.
Torchvision é uma biblioteca que pode ser usada para transformações de imagem comuns em visão computacional.
Letterbox é uma função definida no módulo datasets do pacote utils. É usado para redimensionar uma imagem para uma dimensão específica sem alterar o aspecto original da imagem.
from utils.general import non_max_suppression_kpt: non_max_suppression_kpt é uma função definida no módulo general do pacote utils. É uma técnica usada para suprimir (ou seja, remover) caixas delimitadoras redundantes em uma imagem após a detecção de objetos.
from utils.plots import output_to_keypoint, plot_skeleton_kpts: output_to_keypoint e plot_skeleton_kpts são funções definidas no módulo plots do pacote utils. Eles são usados para converter a saída do modelo em pontos-chave e plotar o esqueleto baseado em pontos-chave, respectivamente.
"""

import torch
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt

import numpy as np
import cv2

import time
import sys

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
weigths = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
model = weigths['model']
model = model.to(device).eval()

video_path = '../inference_data/video_5.mp4'

cap = cv2.VideoCapture(video_path)

if cap.isOpened() == False:
    print("Video não foi aberto!")
    sys.exit(1)

# Captura largura e altura
largura = int(cap.get(3))
altura = int(cap.get(4))

# Criando uma imagem com resize para funcionar no modelo yolov7 pose usando LetterBox 
# LetterBox 
vid_write_image = letterbox(cap.read()[1], (largura), stride=64, auto=True)[0]

# Altura e largura da imagem que sobre resize
resize_height, resize_width = vid_write_image.shape[:2]

save_name = f"{video_path.split('/')[-1].split('.')[0]}"

# Criando um VideoWriter para escrever vídeo de inferência com dimensôes da imaem que sofreu resize
out = cv2.VideoWriter(f"{save_name}_keypoint.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (resize_width, resize_height))

frame_count = 0 # Contador de frames
total_fps = 0 # Total de frames por segundo

while True:
    # Captura cada frame (frame) do video
    # ret é um bool que diz seo frame foi capturado ou não
    ret, frame = cap.read()

    if ret:
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

    start_time = time.time()
    with torch.no_grad():
        output, _ = model(image)

    end_time = time.time()

    # calculando FPS
    fps = 1 / (end_time - start_time)
    total_fps += fps

    frame_count += 1

    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

        # Comment/Uncomment the following lines to show bounding boxes around persons.
        xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
        xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
        cv2.rectangle(
            nimg, 
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color=(255, 0, 0),
            thickness=2, 
            lineType=cv2.LINE_AA
        )


    cv2.putText(nimg, f"{fps:.1f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow('image', nimg)
    out.write(nimg)
    # Press `q` to exit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()
# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.1f}")