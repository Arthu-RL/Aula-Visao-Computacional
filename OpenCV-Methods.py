import cv2
import numpy as np
import sys

# Carregar uma imagem
caminhoImagem = './passaro.jpg' # Substitua por seu caminho de imagem
imagem = cv2.imread(caminhoImagem)
imagem = cv2.resize(imagem, dsize=(900, 500))
cv2.imshow('Original imagem', imagem) 
cv2.waitKey(0)

# Nome da imagem
nomeImagem = f"{caminhoImagem.split('/')[-1].split('.')[0]}"

# Informações da imagem
cloneImagem = imagem.copy()
altura, largura, canais = cloneImagem.shape

print(f"Altura: {altura}, largura: {largura}, canais: {canais}")

# Colocar texto na imagem clone
cv2.putText(cloneImagem, "Área da imagem: "+str(largura*altura), (20, 30), 1, 1.2, (255, 222, 124), 1)
cv2.putText(cloneImagem, "Canais: "+str(canais), (20, 60), 1, 1.2, (0, 255, 255), 1)

# Dessenhar formas geométricas na imagemm
cv2.rectangle(cloneImagem, (50, 100, 100, 50), (255, 255, 255), 2)
cv2.line(cloneImagem, (50, 200), (400, 200), (255, 255, 255), 2)
cv2.circle(cloneImagem, (100, 250), 20, (255, 255, 255), 2)
cv2.ellipse(cloneImagem, (300, 400), (100, 200), 90, 0, 360, (255, 255, 255), 2) # TO FIX

# cv2.imwrite('imagem_Clone.jpg', cloneImagem)
cv2.imshow('imagem Clone', cloneImagem)
cv2.waitKey(0)

# Converter para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem cinza', cinza)
# cv2.imwrite(nomeImagem+"_Imagem_cinza.jpg", cinza)
cv2.waitKey(0)

# Detecção de bordas
bordas = cv2.Canny(imagem, 100, 200)
cv2.imshow('Bordas Canny', bordas)
# cv2.imwrite(nomeImagem+"_Bordas.jpg", bordas)
cv2.waitKey(0)

# Encontrando contornos
contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornosImagem = imagem.copy()
cv2.drawContours(contornosImagem, contornos, -1, (0,255,0), 3)
cv2.imshow('Contornos', contornosImagem)
# cv2.imwrite(nomeImagem+"_Contornos.jpg", contornosImagem)
cv2.waitKey(0)

# Aplicar desfoque
desfoque = cv2.GaussianBlur(imagem, (7, 7), 0)
cv2.imshow('Imagem Desfocada', desfoque)
# cv2.imwrite(nomeImagem+"_Imagem_Desfocada.jpg", desfoque)
cv2.waitKey(0)

# Erosão
erosão = cv2.erode(imagem, None, iterations=2)
cv2.imshow('Imagem Erodida', erosão)
# cv2.imwrite(nomeImagem+"_Imagem_Erodida.jpg", erosão)
cv2.waitKey(0)

# Dilatação
dilatada = cv2.dilate(imagem, None, iterations=2)
cv2.imshow('Imagem Dilatada', dilatada)
# cv2.imwrite(nomeImagem+"_Imagem_Dilatada.jpg", dilatada)
cv2.waitKey(0)

# Equalização
equalizacao = cv2.equalizeHist(cinza)
cv2.imshow('Equalização Histograma', equalizacao)
# cv2.imwrite(nomeImagem+"_Equalização_Histograma.jpg", equalizacao)
cv2.waitKey(0)

# Detecção de canto com Harris
gray = np.float32(imagem)
harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
imagem_harris = imagem.copy()
imagem_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 255, 0]
cv2.imshow('Harris Corners', imagem_harris)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Capturar vídeo 
caminhoVideo = "video.mp4"
cap = cv2.VideoCapture(caminhoVideo)

if cap.isOpened() == False:
    print("Video não foi aberto!")
    sys.exit(1)

numFrames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    numFrames += 1

print(f"Total de frames capturados: {numFrames}")

cap.release()
cv2.destroyAllWindows()