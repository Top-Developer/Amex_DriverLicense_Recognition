import os
import cv2


# image file read with opencv
def read_image_file_with_opencv(dir, fileName):
    filePath = os.path.join(dir, fileName)
    img = cv2.imread(filePath)

    return img


# delay for image showing
def delay():
	k = cv2.waitKey(0) & 0xff
	if k == 27:
		exit(1)

	if k > 0:
		return 

	return 