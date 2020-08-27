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
    conectou, image = webcam.read()
    
    height, width = image.shape[:2]
    print(height, width)

    # cor = (0,0,255)

    # imgLinha = cv2.line(imagem, (133, 600), (450, 14), cor, 2)
    # imgLinha2 = cv2.line(imgLinha, (1180, 600), (780, 14), cor, 2)

    # imgRecortada = imgLinha2[yInicial:yFinal, xInicial:xFinal]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on the blank
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

    # Draw the lines on the edge image
    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    cv2.imshow("Imagem", combo)

    teclou = cv2.waitKey(1) & 0xff
    if teclou == ord('q') or teclou == 27:
        break

webcam.release()
cv2.destroyAllWindows()
