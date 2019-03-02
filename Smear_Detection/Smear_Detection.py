import numpy as np
import cv2
import glob
from random import randint
import sys

#Defining the characterstics of a smear in a image
min_radius = 10
max_radius = 12

min_area = 3.14 * min_radius ** 2
max_area = 3.14 * max_radius ** 2


def isSmearDetected(src):
	
	data = glob.glob(src+"/*.jpg")
	total_data_length = len(data)
	average = np.zeros((500, 500, 3), np.float)
		
	for image in data:
		curr_image = cv2.imread(image)
        
        #Step 1: Resizing images from 2032 x 2032 to 500 x 500
		resize_curr_image = cv2.resize(curr_image,(500,500))
        
        #Step 2: Using Gaussian Filter
		resized_image = cv2.GaussianBlur(resize_curr_image, (3,3), 0)
        
		i = np.array(resize_curr_image,dtype=np.float)
		average += i

	average = average / total_data_length

    #Saving Mean image  
	cv2.imwrite("Mean_"+src.split("/")[1] +"_"+ src.split("/")[2]+".jpg", average)

	average = np.array(np.round(average),dtype=np.uint8)

    #Step 3: Converting Mean image to  grayscale
	grayscale_image = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY)

    #Step 4: Finding the Threshold using built-in adaptive threshold method
	adaptive_threshold_image = cv2.adaptiveThreshold(grayscale_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 105, 11)
    
    #Step 5: Finding the invert of the image which will act as a mask
	mask = cv2.bitwise_not(adaptive_threshold_image)

    #Reading a random image from the source directory to detect the smear.
	read = data[randint(0,total_data_length)]
	read_rand_image = cv2.imread(read)
	resized_rand_image = cv2.resize(read_rand_image,(500,500))

    #Step 6: Finding Contours on the masked image
	contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if contours:
        # check to see if the contours are the size of a smear in the randomly picked image
		if(cv2.contourArea(contours[0]) > min_area and cv2.contourArea(contours[0]) < max_area):
            
            #Drawing contours around the smear on the original image
			result = cv2.drawContours(resized_rand_image,contours,-1,(0,255,255),2)

            # Saving the image to the disk.
			cv2.imwrite("Final_" + src.split("/")[1] + "_" + src.split("/")[2]+".jpg", result)
			cv2.imwrite("Masked_" + src.split("/")[1] +"_"+ src.split("/")[2] + ".jpg", mask)
			return True
	return False

if __name__ == "__main__":
	print("Usage: python Smear_Detection.py <path>/")
	args = sys.argv[1:]
	if not args[0]:
		print ("Error: Directory is invalid.")
		sys.exit()

	print("Directory Found. \n Smear Detection in Progress.")
	if(isSmearDetected(args[0])):
		print ("Smear is detected for "+args[0]+" source.")
	else:
		print("No Smear in "+ args[0])