import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


path = 'C:/Users/kldyu/OneDrive/Desktop/\MelanimaCanserProject/orginal images/melanoma/*.*'
outPut_path = 'C:/Users/kldyu/OneDrive/Desktop/\MelanimaCanserProject/pre processed images/melanoma/'
print(path)
print(outPut_path)



image_num=0

for file in glob.glob(path):
    print(file)
    
    im = cv2.imread(file)
    
    print( im.shape )
    """
    cv2.imshow("original Image" , im )
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    """
################################################## remove hair
    
# Convert the original image to grayscale
    grayScale = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
    """
    cv2.imshow("GrayScale",grayScale)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    cv2.imwrite('grayScale_sample1.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    """
# Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1,(17,17))

# Perform the blackHat filtering on the grayscale image to find the 
# hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    """
    cv2.imshow("BlackHat",blackhat)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    """
#cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

# intensify the hair countours in preparation for the inpainting 
# algorithm
    ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    print( thresh2.shape )
    
    """
    cv2.imshow("Thresholded Mask",thresh2)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    #cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    """
    
# inpaint the original image depending on the mask
    dst = cv2.inpaint(im,thresh2,1,cv2.INPAINT_TELEA)
    
    resize_img = cv2.resize(dst, (300, 300),interpolation = cv2.INTER_NEAREST)
    """
    cv2.imshow("InPaint",resize_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    
    image_num+=1
    num=str(image_num)
    
    p=outPut_path+num+'.jpg'
    print(p)
      
    cv2.imwrite( p, dst)
    """
    
    #####################################################  Sharpening  ######################################################################################################################


#-----Reading the image-----------------------------------------------------

#img = cv2.imread('Dog.jpg', 1)
    '''
    cv2.imshow("img",dst) 
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    '''
#-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    
    """
    cv2.imshow("lab",lab)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    """
#-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    
    """
    cv2.imshow('l_channel', l)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    cv2.imshow('a_channel', a)
    cv2.waitKey(100)
    cv2.destroyAllWindows()    
    cv2.imshow('b_channel', b)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    """
    
#-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    """
    cv2.imshow('CLAHE output', cl)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    
    """
#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    
    """
    cv2.imshow('limg', limg)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    """
#-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    """
    cv2.imshow('final', final)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
#cv2.imwrite("D:\Acedemic\FYP\Code Test\Test1\sample1.jpg",final)

    """
    
    cv2.imshow("InPaint",final)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    
    image_num+=1
    num=str(image_num)
    
    p=outPut_path+num+'.jpg'
    print(p)
      
    cv2.imwrite( p, final)

    


