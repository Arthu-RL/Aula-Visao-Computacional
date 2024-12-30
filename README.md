# Aula-Visao-Computacional

## [Utilizado recursos deste repositório](https://github.com/WongKinYiu/yolov7.git)

## [OpenCv Official page](https://docs.opencv.org/4.x/d1/dfb/intro.html)

## [Clues](https://stackabuse.com/real-time-pose-estimation-from-video-in-python-with-yolov7/)

## [More Clues](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/)

## Preparar ambiente

### Criar ambiente python
```sh
python -m venv vision
```

### Ativar ambiente e instalar dependências com pip
```sh
vision\Scripts\activate
```

```sh
pip install -r requirements.txt
```

### [Faça download do arquivo weights](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

### Rodar Código

FILE

```sh
python Yolov7Pose-Detection.py
```

RTSP

```sh
python Yolov7Pose-Detection.py --src="rtsp://192.168.x.x:8080/h264_ulaw.sdp" -t 1
```

WEBCAM

```sh
python Yolov7Pose-Detection.py --src 0 -t 0 -o
```

