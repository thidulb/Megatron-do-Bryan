import cv2
import numpy as np
print("Versão", cv2.__version__)

webcam = cv2.VideoCapture(0)

xInicial=0
yInicial=0

xFinal = 500
yFinal = 500

# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    conectou, imagem = webcam.read()
    
    height, width = imagem.shape[:2]
    print(height, width)

    cor = (0,0,255)

    imgLinha = cv2.line(imagem, (133, 600), (450, 14), cor, 2)
    imgLinha2 = cv2.line(imgLinha, (1180, 600), (780, 14), cor, 2)

    # imgRecortada = imgLinha2[yInicial:yFinal, xInicial:xFinal]
    cv2.imshow("Imagem", imagem)

    teclou = cv2.waitKey(1) & 0xff
    if teclou == ord('q') or teclou == 27:
        break

webcam.release()
cv2.destroyAllWindows()
