from vehicle import Driver
from controller import Camera, Display, Keyboard
import cv2
import numpy as np
from numpy import array

car = Driver()
# cameraFront = Camera("cameraFront")
cameraTop = Camera("cameraTop")
display = Display("displayTop")
display.attachCamera(cameraTop)
keyboard = Keyboard()

# cameraFront.enable(32)
cameraTop.enable(32)
keyboard.enable(32)


while car.step() != -1:
    display.setColor(0x000000)
    display.setAlpha(0.0)
    display.fillRectangle(0,0, display.getWidth(), display.getHeight())

    img = cameraTop.getImage()
    
    image = np.frombuffer(img, np.uint8).reshape((cameraTop.getHeight(), cameraTop.getWidth(), 4))
    # cv2.imwrite("img.png", image)
    gray = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2GRAY)

    #--- vira a imagem da camera em 90 graus
    #gray270 = np.rot90(gray, 3)
    #grayFlip = cv2.flip(gray270, 1)
    #cv2.imwrite("grayflip.jpeg", grayFlip)

    #--- gera o blur na imagem da camera
    kernel_size = 5
    blurGray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    #--- converte a imagem
    blurGrayConv = np.uint8(blurGray)
    #cv2.imwrite("imagem.jpeg",blurGrayConv)

    #--- Define um intervalo de saturacao para obter as bordas
    low_threshold = 0
    high_threshold = 30
    edges = cv2.Canny(blurGrayConv, low_threshold, high_threshold)
    
    #--- Aqui sao criados mascaras para as bordas usando cv2.fillPoly()
    maskEsquerda = np.zeros_like(edges)
    maskDireita = np.zeros_like(edges)   
    ignore_mask_color = 255   

    #--- Area de foco para um melhor processamento
    #--- Definimos os ponto para o corte (informar coordenada inicial e a proxima coordenda para ser gerada a reta)
    #--- Corte no reconhecimento de linha no campo de visão onde se encontra a faixa
    # vertices = np.array([[(39,359),(279, 249) ,(349, 249), (539, 359)]], dtype=np.int32)
    # vertices = np.array([[(79, 359), (34,359), (279,249), (349,249), (539,359), (599,359), (324,249), (299,249)]], dtype=np.int32)

    verticesEsquerda = np.array([[(0,479),(0, 370) ,(320, 250), (380, 250), (110, 479)]], dtype=np.int32)
    verticesDireita = np.array([[(719,479),(719, 370) ,(420, 250), (400, 250), (610, 479)]], dtype=np.int32)
    
    #--- reconhece as faixas de uma area predeterminada da camera
    cv2.fillPoly(maskEsquerda, verticesEsquerda, ignore_mask_color)
    cv2.fillPoly(maskDireita, verticesDireita, ignore_mask_color)
    masked_edgesEsquerda = cv2.bitwise_and(edges, maskEsquerda)
    masked_edgesDireita = cv2.bitwise_and(edges, maskDireita)

    #--- Define os parametros da transformada de Hough 
    rho = 1
    theta = np.pi/180
    threshold = 10
    min_line_length = 40
    max_line_gap = 20

    #--- executa a transformada na linha detectada
    linesEsquerda = cv2.HoughLinesP(masked_edgesEsquerda, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    linesDireita = cv2.HoughLinesP(masked_edgesDireita, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    display.setAlpha(1.0)
    display.setColor(0xFF0000)
    
    #--- cria a linha que vai ser pintada
    line_imageEsquerda = np.copy(np.float32(image))*0
    line_imageDireita = np.copy(np.float32(image))*0
    bAuxLinesE = np.any(linesEsquerda)
    if (bAuxLinesE == None):
        print("Erro ao ler lines")
        pass
    
    else:
        for i in range(len(linesEsquerda)):
            for x1,y1,x2,y2 in linesEsquerda[i]:
                #print(x1,y1,x2,y2)
                cv2.line(line_imageEsquerda,(x1,y1),(x2,y2),(0,0,255),10)
                display.drawLine(int(x1), int(y1), 
                int(x2), int(y2))
    
    bAuxLinesD = np.any(linesDireita)
    if (bAuxLinesD == None):
        print("Erro ao ler lines")
        pass
    
    else:
        for i in range(len(linesDireita)):
            for x1,y1,x2,y2 in linesDireita[i]:
                #print(x1,y1,x2,y2)
                cv2.line(line_imageDireita,(x1,y1),(x2,y2),(0,0,255),10)
                display.drawLine(int(x1), int(y1), 
                int(x2), int(y2))

    color_edgesE = np.dstack((masked_edgesEsquerda, masked_edgesEsquerda, masked_edgesEsquerda)) 
    color_edgesD = np.dstack((masked_edgesDireita, masked_edgesDireita, masked_edgesDireita)) 

    #--- junta a imagem da camera com a linha criada
    comboE = cv2.addWeighted(np.float32(color_edgesE), 0.8, np.float32(line_imageEsquerda), 1, 0)
    comboD = cv2.addWeighted(np.float32(color_edgesD), 0.8, np.float32(line_imageDireita), 1, 0)

    npimagec=np.array(comboE)
    npimagec1=np.array(comboD)

    # --- "Filtro para vermelho"
    red=np.array([0,0,255],dtype=np.uint8)

    # -- Encontra vermelho sendo [0][0] =y ; [1][0]
    redsc=np.where(np.all((npimagec==red),axis=-1))
    redsc1=np.where(np.all((npimagec1==red),axis=-1))

    nparrayc = array(redsc)
    nparrayc1 = array(redsc1)

    qtdVermelho0 = nparrayc.shape[1]

    qtdVermelho1 = nparrayc1.shape[1]

    print("Quantos pixes tem no combo 0: ", qtdVermelho0)
    print("Quantos pixes tem no combo 1: ", qtdVermelho1)

    #erro = qtdVermelho0 - qtdVermelho1
    # --- Se negativo virar para a X se positivo para Y 
    # --- Analisar trativa por causa do numero 

    #print(erro)

    k = keyboard.getKey()       
    if k == ord('W'):
        print("forward")
        car.setSteeringAngle(0)
        car.setCruisingSpeed(10)
    elif k == ord('D'):
        print("turn right")
        car.setSteeringAngle(0.5)
        car.setCruisingSpeed(10)
    elif k == ord('A'):
        print("turn left")
        car.setSteeringAngle(-0.5)
        car.setCruisingSpeed(10)
    elif k == ord('S'):
        print("stop")
        car.setSteeringAngle(0)
        car.setCruisingSpeed(0)
 
   

