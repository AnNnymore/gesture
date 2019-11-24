import cv2
import numpy as np
#import math
import fourierDescriptor as fd
#from scipy.ndimage.interpolation import geometric_transform

MIN_DESCRIPTOR = 500
NUMBER_OF_HARMONICS = 20

#二值化处理
def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,255,0)) #画出截取的手势框图
	roi = frame[y0:y0+height, x0:x0+width] #获取手势框图
	res = skinMask(roi) #进行肤色检测

	kernel = np.ones((3,3), np.uint8) #设置卷积核
	res = cv2.erode(res, kernel) #腐蚀操作
	res = cv2.dilate(res, kernel)#膨胀操作

	ret, fourier_result = fd.fourierDesciptor(res)
	return roi, res, ret, fourier_result


####YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #Ostu处理
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res


'''
########Cr，Cb范围筛选法###########
def skinMask(roi):
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,cr,cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	skin = np.zeros(cr.shape, dtype = np.uint8)
	(x,y) = cr.shape
	for i in range(0, x):
		for j in range(0, y):
			#每个像素点进行判断
			if(cr[i][j] > 130) and (cr[i][j] < 175) and (cb[i][j] > 77) and (cb[i][j] < 127):
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res
'''
