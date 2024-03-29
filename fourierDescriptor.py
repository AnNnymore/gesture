#!/usr/bin/env python
# -*-coding:utf-8 -*-
import cv2
import numpy as np

MIN_DESCRIPTOR = 32  # surprisingly enough, 2 descriptors are already enough

##计算傅里叶描述子
def fourierDesciptor(res):
    #Laplacian算子进行八邻域检测
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = find_contours(Laplacian)#提取轮廓点坐标
    contour_array = contour[0][:, 0, :]#注意这里只保留区域面积最大的轮廓点坐标
    ret_np = np.ones(dst.shape, np.uint8) #创建黑色幕布#################
    ret = cv2.drawContours(ret_np,contour[0],-1,(255,255,255),1) #绘制白色轮廓#####################
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换

    descirptor_in_use = truncate_descriptor(fourier_result)#截短傅里叶描述子###########

    return ret,descirptor_in_use###############ret#################

def find_contours(Laplacian):
    #binaryimg = cv2.Canny(res, 50, 200) #二值化，canny检测
    h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    contour = h[1]
    contour = sorted(contour, key = cv2.contourArea, reverse=True)#对一系列轮廓点坐标按它们围成的区域面积进行排序
    return contour

#截短傅里叶描述子
def truncate_descriptor(fourier_result):
    descriptors_in_use = np.fft.fftshift(fourier_result)

    #取中间的MIN_DESCRIPTOR项描述子
    center_index = int(len(descriptors_in_use) / 2)
    low, high = center_index - int(MIN_DESCRIPTOR / 2), center_index + int(MIN_DESCRIPTOR / 2)
    descriptors_in_use = descriptors_in_use[low:high]

    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use




