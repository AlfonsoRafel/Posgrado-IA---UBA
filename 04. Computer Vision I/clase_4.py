import cv2 as cv
import numpy as np
import os


if __name__ == "__main__":
    folder = 'Images/Hough/'

    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Suavizamos la imagen
        #=====================
        gray = cv.medianBlur(gray, 5)
        edges = cv.Canny(gray, 100, 105, L2gradient=True)

        cv.imshow('Eyes', edges)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Obtenemos los círculos por la transformada de Hough
        # (imagen en grises, método, flag de resolución del acumulador, dist mín entre centros de círculos
        # umbral alto de Canny, umbral del acumulador, radio_min, radio_max)
        #===================================================================
        img_out = img.copy()
        circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, minDist=85, param1=130, param2=20, minRadius=10, maxRadius=40)
        circles = np.uint16(np.around(circles))

        mask = np.full((img_out.shape[0], img_out.shape[1]), 0, dtype=np.uint8)  # mask is only
        for i in circles[0,:]:
            # Dibujamos el círculo externo
            cv.circle(img_out,(i[0],i[1]),i[2],(0,255,0),2)
            # Dibujamos el centro del círculo
            cv.circle(img_out,(i[0],i[1]),2,(0,0,255),3)
            cv.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

        fg = cv.bitwise_or(img, img, mask=mask)
        gray = cv.medianBlur(fg, 5)
        edges = cv.Canny(gray, 75, 90, L2gradient=True)
        circles_p = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, minDist=60, param1=130, param2=20, minRadius=5,
                                  maxRadius=19)
        circles_p = np.uint16(np.around(circles_p))
        for i in circles_p[0,:]:
            # Dibujamos el círculo externo
            cv.circle(img_out,(i[0],i[1]),i[2],(255,0,0),2)

        cv.imshow('Pupils', edges)
        cv.imshow('Detected Iris', img_out)
        cv.imshow('Mask', mask)
        cv.imshow('Masked Image', fg)
        #for j in range(0, img.shape[0], img.shape[0]/5):
        #    end = j + img.shape[0]/5 if j + img.shape[0]/5 <= img.shape[0] else img.shape[0]
        #    parse = img_out[j:end, :, :]
        #cv.imshow('Pupils', parse)


        
        cv.waitKey(0)
        cv.destroyAllWindows()





