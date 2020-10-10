import imageio
import copy as cp
import numpy as np
import cv2
from PIL import Image

'''
    ###############
    addBoundary(img, kernel)
    convolve1(img, kernel, filter_type, mode='same')
    convolve(img, kernel, filter_type, mode='same')
    wise_element_sum(img, kernel, filter_type)
    上面四个函数用于构建高斯滤波器，与我写的第二章节的作业中的滤波器一样（我是先做第二章作业再做第一章的）
'''

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

def Gauss_Fileter(img, kernel_size, sigma):
    '''
    高斯滤波器
    :param img: 输入图像
    :param kernel_size: 卷积核大小
    :param sigma: 高斯函数的标准差
    :return: 高斯滤波后的图片
    '''
    # 避免除0
    if sigma == 0:
        sigma = 6
    kernel = np.zeros([kernel_size, kernel_size])
    kernel_center = kernel_size / 2  # 卷积核中心位置
    sum_val = 0  # 记录卷积核中数字之和
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            kernel[i, j] = np.exp((-(i - kernel_center) ** 2 + (j - kernel_center) ** 2) / (2 * (sigma ** 2)))
            sum_val += kernel[i, j]
    # 得到卷积核
    kernel = kernel / sum_val
    img_out = convolve(img, kernel, filter_type='Gauss_Fileter', mode='same')
    # img_out = scipy.signal.convolve2d(img, kernel, mode='same', boundary='symm')
    # 返回图片
    return img_out






def white_balance(img):
    '''
    原始灰度世界算法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''

    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb = K / B_ave
    Kg = K / G_ave
    Kr = K / R_ave
    Bnew = B * Kb
    Gnew = G * Kg
    Rnew = R * Kr

    for i in range(len(Bnew)):
        for j in range(len(Bnew[0])):
            Bnew[i][j] = 255 if Bnew[i][j] > 255 else Bnew[i][j]
            Gnew[i][j] = 255 if Gnew[i][j] > 255 else Gnew[i][j]
            Rnew[i][j] = 255 if Rnew[i][j] > 255 else Rnew[i][j]

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))

    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Bnew
    dst_img[:, :, 1] = Gnew
    dst_img[:, :, 2] = Rnew
    return dst_img

def deMosaic(raw_image):
    '''
    对图片插值，转为RGB图片
    :param raw_image: 输入单通道的图片
    :return: RGB图
    '''
    H = raw_image.shape[0]
    W = raw_image.shape[1]
    R = raw_image
    r_image = cp.deepcopy(R)
    g_image = cp.deepcopy(R)
    b_image = cp.deepcopy(R)

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):
            r_image[i + 1][j] = raw_image[i][j]
            r_image[i + 1][j + 1] = raw_image[i][j]
            r_image[i][j + 1] = raw_image[i][j]

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):
            temp = raw_image[i + 1][j] / 2 + raw_image[i][j + 1] / 2
            g_image[i][j] = temp
            g_image[i + 1][j + 1] = temp
            g_image[i + 1][j] = temp
            g_image[i][j + 1] = temp

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):
            b_image[i + 1][j] = raw_image[i + 1][j + 1]
            b_image[i][j] = raw_image[i + 1][j + 1]
            b_image[i][j + 1] = raw_image[i + 1][j + 1]

    rgb_image = cv2.merge([b_image, g_image, r_image])
    return rgb_image



def deMosaic1(raw_image):
    '''
    对图片插值，转为RGB图片，主要与deMosaic()函数的效果作对比
    :param raw_image: 输入单通道的图片
    :return: RGB图
    '''
    H = raw_image.shape[0]
    W = raw_image.shape[1]
    R = raw_image
    r_image = cp.deepcopy(R)
    g_image = cp.deepcopy(R)
    b_image = cp.deepcopy(R)

    for i in range(1, H - 1, 3):
        for j in range(1, W - 1, 3):
            temp = (raw_image[i-1][j-1]+raw_image[i+1][j-1]+raw_image[i-1][j+1]+raw_image[i+1][j+1])/4
            r_image[i - 1][j] = temp
            r_image[i+1][j] = raw_image[i][j]
            #r_image[i][j + 1] = raw_image[i][j]

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):
            #temp = raw_image[i + 1][j] / 2 + raw_image[i][j + 1] / 2
            g_image[i][j] = raw_image[i][j]
            g_image[i + 1][j + 1] = raw_image[i][j]
            g_image[i + 1][j] = raw_image[i][j]
            g_image[i][j + 1] = raw_image[i][j]

    for i in range(0, H - 1, 2):
        for j in range(1, W - 1, 2):
            b_image[i][j-1] = raw_image[i][j]
            b_image[i][j+1] = raw_image[i][j]
            #b_image[i][j + 1] = raw_image[i + 1][j + 1]

    rgb_image = cv2.merge([b_image, g_image, r_image])
    return rgb_image


def deMosaic2(raw_image):
    '''
    对图片插值，转为RGB图片，主要与deMosaic()函数的效果作对比
    :param raw_image: 输入单通道的图片
    :return: RGB图
    '''
    H = raw_image.shape[0]
    W = raw_image.shape[1]
    R = raw_image
    r_image = cp.deepcopy(R)
    g_image = cp.deepcopy(R)
    b_image = cp.deepcopy(R)

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):

            r_image[i,j]=raw_image[i][j]
            r_image[i + 1][j] = raw_image[i][j]/2
            r_image[i + 1][j + 1] = raw_image[i][j]/2
            r_image[i][j + 1] = raw_image[i][j]/2

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):
            temp = raw_image[i + 1][j] / 2 + raw_image[i][j + 1] / 2
            g_image[i][j] = temp
            g_image[i + 1][j + 1] = temp/2
            g_image[i + 1][j] = temp/2
            g_image[i][j + 1] = temp/2

    for i in range(0, H - 1, 2):
        for j in range(0, W - 1, 2):
            r_image[i][j] = raw_image[i+1][j+1]
            b_image[i + 1][j] = raw_image[i + 1][j + 1]/2
            b_image[i][j] = raw_image[i + 1][j + 1]/2
            b_image[i][j + 1] = raw_image[i + 1][j + 1]/2

    rgb_image = cv2.merge([b_image, g_image, r_image])
    return rgb_image


def gamma_correct(img,gamma):
    '''
    用于gamma校正
    :param img: 输入RGB图
    :return: gamma校正后的图
    '''
    img = np.power(img / 255.0, gamma)
    img = img * 255
    return img


def main():
    Image.MAX_IMAGE_PIXELS = None

    img = cv2.imread("raw-data-BayerpatternEncodedImage.tif", 1).astype(np.float)

    single_img = img[:, :, 0]
    imageio.imsave('单通道图片.jpg', single_img)

    #组合一
    deMosaic_img = deMosaic(single_img)
    imageio.imsave('RGB图片1.jpg', deMosaic_img)
    balance_img = white_balance(deMosaic_img)
    imageio.imsave('白平衡1.jpg', balance_img)
    gamma_img = gamma_correct(balance_img, 1.2)
    imageio.imsave('gamma校正1.jpg', gamma_img)
    Filter_img = Gauss_Fileter(gamma_img, 5, 25)
    imageio.imsave('高斯滤波1-sigma=25.jpg', Filter_img)
    '''
    
    #组合二
    deMosaic_img = deMosaic(single_img)
    imageio.imsave('RGB图片2.jpg', deMosaic_img)
    Filter_img = Gauss_Fileter( deMosaic_img, 5, 25)
    imageio.imsave('高斯滤波2-sigma=25.jpg', Filter_img)

    balance_img = white_balance(Filter_img)
    imageio.imsave('白平衡2.jpg', balance_img)

    gamma_img = gamma_correct(balance_img, 1.2)
    imageio.imsave('gamma校正2.jpg', gamma_img)

  '''







    '''
    #组合1各步骤不同参数对图片处理的效果
    
    deMosaic_img = deMosaic(single_img)
    imageio.imsave('RGB图片0.jpg', deMosaic_img)

    deMosaic_img1 = deMosaic1(single_img)
    imageio.imsave('RGB图片1.jpg', deMosaic_img1)

    deMosaic_img2 = deMosaic2(single_img)
    imageio.imsave('RGB图片2.jpg', deMosaic_img2)


    balance_img = white_balance(deMosaic_img)
    imageio.imsave('白平衡3.jpg', balance_img)
    balance_img1 = white_balance(deMosaic_img1)
    imageio.imsave('白平衡4.jpg', balance_img1)
    balance_img2 = white_balance(deMosaic_img2)
    imageio.imsave('白平衡5.jpg', balance_img2)


    gamma_img = gamma_correct(balance_img, 1.2)
    imageio.imsave('gamma校正.jpg', gamma_img)

    gamma_img1 = gamma_correct(balance_img, 2)
    imageio.imsave('gamma校正1.jpg', gamma_img1)

    gamma_img2 = gamma_correct(balance_img, 4)
    imageio.imsave('gamma校正2.jpg', gamma_img2)

    gamma_img3 = gamma_correct(balance_img, 0.8)
    imageio.imsave('gamma校正3.jpg', gamma_img3)

    gamma_img4 = gamma_correct(balance_img, 0.5)
    imageio.imsave('gamma校正4.jpg', gamma_img4)


    gamma_img5 = gamma_correct(balance_img, 0.1)
    imageio.imsave('gamma校正5.jpg', gamma_img5)



    Filter_img = Gauss_Fileter(gamma_img, 5, 15)
    imageio.imsave('高斯滤波-sigma=15.jpg', Filter_img)
    Filter_img1 = Gauss_Fileter(gamma_img, 5, 25)
    imageio.imsave('高斯滤波-sigma=25.jpg', Filter_img1)
    Filter_img3 = Gauss_Fileter(gamma_img, 5, 5)
    imageio.imsave('高斯滤波-sigma=5.jpg', Filter_img3)
     '''

if __name__ == '__main__':
    main()
