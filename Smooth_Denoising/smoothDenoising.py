import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
import scipy
from scipy import signal
from PIL import Image
from scipy.ndimage import median_filter

# 由于卷积核的大小一般是奇数，因此这里假设卷积核是奇数的


'''
    ####################
    图像处理的基本函数
    ####################
'''


# 图像加框
def addBoundary(img, kernel):
    '''
    给图像添加边界
    :param img: 输入图像
    :param kernel:卷积核
    :return: 加边界后的图像
    '''
    kernel_size = kernel.shape[0]
    addLine = (int)((kernel_size - 1) / 2)
    img_ = cv2.copyMakeBorder(img, addLine, addLine, addLine, addLine, cv2.BORDER_CONSTANT, value=0);
    return img_


def convolve1(img, kernel, filter_type, mode='same'):
    '''
    单通道图像与卷积核的卷积，主要用于灰度图
    :param img: 输入单通道图像矩阵
    :param kernel: 卷积核
    :param model: medium,gauss,mean, 即选择中值滤波、高斯滤波、还是均值滤波，其他滤波方式以后添加
    :return: 卷积后的图像
    '''
    if mode == 'same':
        img_ = addBoundary(img, kernel)
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    # 横向卷积、纵向卷积的次数
    conv_height = img_.shape[0] - kernel_height + 1
    conv_width = img_.shape[1] - kernel_width + 1
    # 卷积结果存储在conv中
    conv = np.zeros((conv_height, conv_width), dtype='uint8')

    for i in range(conv_height):
        for j in range(conv_width):
            conv[i][j] = wise_element_sum(img_[i:i + kernel_height, j:j + kernel_width], kernel, filter_type)
    return conv


def wise_element_sum(img, kernel, filter_type):
    '''
    对于某一次卷积结果的取值
    :param img: 输入的图片片段矩阵
    :param kernel: 卷积核
    :param modle: medium,gauss,mean, 即选择中值滤波、高斯滤波、还是均值滤波，其他滤波方式以后添加
    :return: 返回该像素值
    '''
    if filter_type == 'medium_Filter':
        temp = img * kernel
        list = []
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                list.append(temp[i][j])
        list.sort()
        if list[int(len(list) / 2)] > 255:
            return 255
        elif list[int(len(list) / 2)] < 0:
            return 0
        else:
            return list[int(len(list) / 2)]
    # 均值、高斯滤波等
    else:
        result = (img * kernel).sum()
        if result < 0:
            return 0
        elif result > 255:
            return 255
        else:
            return result


def convolve(img, kernel, filter_type, mode='same'):
    '''
    三通道卷积，主要用于彩色图
    :param img: 输入图像矩阵
    :param kernel: 卷积核
    :param mode: medium,gauss,mean, 即选择中值滤波、高斯滤波、还是均值滤波，其他滤波方式以后添加
    :return: 卷积后的图像矩阵
    '''

    R = np.mat(img[:, :, 0])
    G = np.mat(img[:, :, 1])
    B = np.mat(img[:, :, 2])
    conv_B = convolve1(img[:, :, 0], kernel, filter_type, mode)
    conv_G = convolve1(img[:, :, 1], kernel, filter_type, mode)
    conv_R = convolve1(img[:, :, 2], kernel, filter_type, mode)

    conv_img = np.dstack([conv_B, conv_G, conv_R])
    return conv_img


'''
   ############################################
                  噪声函数
   脉冲噪声：add_PulseNoise(img, SNR)
   椒盐噪声：add_Salt_PepperNoise(img, SNR)
   高斯噪声：add_Gauss_Noise(img, mean, sigma)
   #############################################

'''


# 添加脉冲噪声
def add_PulseNoise(img, SNR):
    '''
    给图像添加脉冲噪声
    :param img: 输入图像
    :param SNR: 信噪比，决定添加多少噪声
    :return: 添加噪声后的图像
    '''
    rows, cols, dims = img.shape
    # 创建与图像大小一样的矩阵
    R = np.mat(img[:, :, 0])
    G = np.mat(img[:, :, 1])
    B = np.mat(img[:, :, 2])

    # RGB图转换为灰度图的著名公式: Grap = R*0.299+G*0.587+B*0.114
    Grey = R * 0.299 + G * 0.587 + B * 0.114
    # 噪声点数目
    noise = int((1 - SNR) * rows * cols)
    # 添加噪声
    for i in range(noise):
        # 随机选择图片矩阵的一个格子，设置为脉冲噪声值
        rand_rows = random.randint(0, rows - 1)
        rand_cols = random.randint(0, cols - 1)
        Grey[rand_rows, rand_cols] = 255
        # img[rand_rows, rand_cols] = 255
    return Grey


# 添加椒盐噪声
def add_Salt_PepperNoise(img, SNR):
    '''
    给图像添加椒盐噪声
    :param img: 输入图像
    :param SNR: 输入信噪比，决定添加多少噪声
    :return: 输出添加噪声后的图像
    '''
    rows, cols, dims = img.shape
    # 创建与图像大小一样的矩阵
    R = np.mat(img[:, :, 0])
    G = np.mat(img[:, :, 1])
    B = np.mat(img[:, :, 2])

    # RGB图转换为灰度图的著名公式: Grap = R*0.299+G*0.587+B*0.114
    Grey = R * 0.299 + G * 0.587 + B * 0.114
    # 噪声点数目
    noise = int((1 - SNR) * rows * cols)
    # 添加噪声
    for i in range(noise):
        # 随机选择图片矩阵的一个格子，设置为椒盐噪声值
        rand_rows = random.randint(0, rows - 1)
        rand_cols = random.randint(0, cols - 1)
        if random.randint(0, 1) == 0:
            Grey[rand_rows, rand_cols] = 0  # 盐噪声为255
        else:
            Grey[rand_rows, rand_cols] = 255  # 椒噪声为0
    return Grey


def add_Gauss_Noise(img, mean, sigma):
    '''
    添加高斯噪声
    :param img:输入图像
    :param mean: 高斯分布的均值
    :param sigma: 高斯分布的标准差
    :return: 添加高斯噪声后的图像
    '''
    rows, cols, dims = img.shape
    R = np.mat(img[:, :, 0])
    G = np.mat(img[:, :, 1])
    B = np.mat(img[:, :, 2])
    # 产生灰度图
    Grey = R * 0.299 + G * 0.587 + B * 0.114
    # numpy.random.normal(mean,sigma,shape)是正态分布函数，mean是均值，sigma是标准差，shape表示输出值放在size里
    noise = np.random.normal(mean, sigma, Grey.shape)
    # 将噪声和图片叠加
    Grey = noise + Grey

    # np.min(Grey)：取Grey中的最小值；np.full(arry,num)：给arry全部赋值num
    Grey = Grey - np.full(Grey.shape, np.min(Grey))
    Grey = Grey * 255 / np.max(Grey)
    Grey_p = Grey.astype(np.uint8)  # 类型转换
    return Grey


'''
    ##################
    均值滤波器：mean_Fileter(img, size)
    中值滤波器： medium_Fileter(img, size)
    高斯滤波器：gauss_Kernel(mean, sigma, kernel_size)
'''


def mean_Fileter(img, kernel_size):
    '''
    均值滤波器
    :param img: 输入图像
    :param kernel_size:卷积核大小
    :return: 均值滤波后的图像
    '''
    # kernel_size * kernel_size 滤波器, 每个系数都是 1/9
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    # mode = same 表示输出尺寸等于输入尺寸
    # boundary 表示采用对称边界条件处理图像边缘
    # img_out = scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')
    img_out = convolve1(img, kernel, filter_type='mean_Fileter', mode='same')
    return img_out.astype(np.uint8)


def medium_Filter(img, kernel_size):
    '''
    中值滤波器
    :param img: 输入图像
    :param size: 卷积核大小
    :return: 中值滤波后的图像
    '''
    kernel = np.ones((kernel_size, kernel_size))
    # mode = same 表示输出尺寸等于输入尺寸
    # boundary 表示采用对称边界条件处理图像边缘
    img_out = convolve1(img, kernel, filter_type='medium_Filter', mode="same")
    # img_out = scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')
    return img_out.astype(np.uint8)


def Gauss_Fileter(img, kernel_size, sigma):
    '''
    高斯滤波器
    :param img: 输入图像
    :param kernel_size: 卷积核大小
    :param sigma: 高斯函数的标准差
    :return: 高斯滤波后的图片
    '''
    #避免除0
    if sigma == 0:
        sigma = 6
    kernel = np.zeros([kernel_size, kernel_size])
    kernel_center = kernel_size / 2  # 卷积核中心位置
    sum_val = 0  # 记录卷积核中数字之和
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            kernel[i, j] = np.exp(-((i - kernel_center) ** 2 + (j - kernel_center) ** 2) / (2 * (sigma ** 2)))
            sum_val += kernel[i, j]
    # 得到卷积核
    kernel = kernel / sum_val
    img_out = convolve1(img, kernel, filter_type='Gauss_Fileter', mode='same')
    # img_out = scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')
    # 返回图片
    return img_out


def main():
    img = np.array(Image.open('LenaRGB.bmp'))

    # 加上各种噪声
    img1 = add_PulseNoise(img, 0.9)
    img2 = add_Salt_PepperNoise(img, 0.9)
    img3 = add_Gauss_Noise(img, 0, 8)

    plt.subplot(321)
    plt.title('PulseNoise')
    plt.imshow(img1, cmap='gray')
    plt.subplot(322)
    plt.title('Salt_PepperNoise')
    plt.imshow(img2, cmap='gray')
    plt.subplot(323)
    plt.title('GaussNoise')
    plt.imshow(img3, cmap='gray')
    '''
    #三种滤波器对脉冲噪声的效果
    img1_1 = mean_Fileter(img1, 3)
    img1_2 = medium_Filter(img1, 3)
    img1_3 = Gauss_Fileter(img1, 3, 8)

    plt.subplot(324)
    plt.title('PauseNoise_meanfilter')
    plt.imshow(img1_1, cmap='gray')
    plt.subplot(325)
    plt.title('PauseNoise_mediumfilter')
    plt.imshow(img1_2, cmap='gray')
    plt.subplot(326)
    plt.title('PauseNoise_Gaussfilter')
    plt.imshow(img1_3, cmap='gray')
    plt.show()
    
    #三种滤波器对椒盐噪声的效果
    img2_1 = mean_Fileter(img2, 3)
    img2_2 = medium_Filter(img2, 3)
    img2_3 = Gauss_Fileter(img2, 3, 8)

    plt.subplot(327)
    plt.title('Salt_Pepper_Noise_meanfilter')
    plt.imshow(img2_1, cmap='gray')
    plt.subplot(328)
    plt.title('Salt_Pepper_Noise_mediumfilter')
    plt.imshow(img2_2, cmap='gray')
    plt.subplot(329)
    plt.title('Salt_PepperNoise_Gaussfilter')
    plt.imshow(img2_3, cmap='gray')

    #三种滤波器对高斯噪声的效果
    img3_1 = mean_Fileter(img3, 3)
    img3_2 = medium_Filter(img3, 3)
    img3_3 = Gauss_Fileter(img3, 3, 8)

    plt.subplot(330)
    plt.title('GaussNoise_meanfilter')
    plt.imshow(img3_1, cmap='gray')
    plt.subplot(331)
    plt.title('GaussNoise_mediumfilter')
    plt.imshow(img3_2, cmap='gray')
    plt.subplot(332)
    plt.title('GaussNoise_Gaussfilter')
    plt.imshow(img3_3, cmap='gray')

    plt.show()
    '''
    # 不同尺寸的box filter对噪声图片的效果
    createVar = locals()
    num = [3, 5, 7]
    j = 324
    for i in num:
        createVar['kernel_' + str(i)]=97
        print('kernel_' + str(i))
    kerral_num = []
    for i in num:
        kerral_num.append(Gauss_Fileter(img1, i, 6))
        plt.subplot(j)
        plt.title('GaussNoise_meanfilter_kernel_' + str(i))
        plt.imshow(kerral_num.pop(), cmap='gray')
        j = j + 1

    plt.show()
    '''
    img = np.array(Image.open('LenaRGB.bmp'))
    img1 = add_Gauss_Noise(img, 0, 6.5)
    plt.subplot(321)
    plt.title('Gauss')
    plt.imshow(img1, cmap='gray')
    
    plt.subplot(322)
    plt.title('Grey gauss noise')
    plt.imshow(img2, cmap='gray')
   # plt.show()
   
    img3 = add_Salt_PepperNoise(img, 0.99)

    plt.subplot(323)
    plt.title('Salt_Pepper')
    plt.imshow(img3, cmap='gray')

    # 中值滤波
    img1_mf = scipy.ndimage.median_filter(img1, (8, 8))
    img3_mf = scipy.ndimage.median_filter(img3, (8, 8))

    #高斯滤波
    img1_gf = cv2.GaussianBlur(img1, (3, 3), 0)
    # 均值滤波

    plt.subplot(324)
    plt.title('Salt_pepper_no')
    plt.imshow(img3_mf, cmap='gray')


    plt.subplot(325)
    plt.title('Gauss_no')
    plt.imshow(img1_mf, cmap='gray')
    plt.show()
    '''


if __name__ == '__main__':
    main()
