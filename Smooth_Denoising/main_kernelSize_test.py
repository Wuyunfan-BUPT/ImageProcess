import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import smoothDenoising


'''
    ########################################
    此函数用于观察不同尺寸的滤波器的效果
    注：实验时需要输入信息，请按照控制台提示操作
    #######################################
'''
def main():

    img = np.array(Image.open('LenaRGB.bmp'))
    # 加上各种噪声
    img1 = smoothDenoising.add_PulseNoise(img, 0.9)
    img2 = smoothDenoising.add_Salt_PepperNoise(img, 0.9)
    img3 = smoothDenoising.add_Gauss_Noise(img, 0, 20)

    print("******************************")
    print("***请输入要选择滤波的噪声图片***")
    print("****脉冲噪声请输入：1")
    print("****椒盐噪声请输入：2")
    print("****高斯噪声请输入：3")
    score1 = int(input('请输入您要选择的噪声图片：'))
    if score1 == 1:
        img4 = img1
        NoiseType = "PulseNoise"
    elif score1 == 2:
        img4 = img2
        NoiseType = "SaltPepperNoise"
    else:
        img4 = img3
        NoiseType = "GaussNoise"

    createVar = locals()
    j = 322
    kernelSize = [3, 5, 7, 9, 11]
    for i in kernelSize:
        createVar['kernel_' + str(i)] = 97
        print('kernel_' + str(i))
    list = []
    print("******************************")
    print("不同尺寸的   均值滤波器效果   请输入：1")
    print("不同尺寸的   中值滤波器效果   请输入：1")
    print("不同尺寸的   高斯滤波器效果   请输入：1")
    score2 = int(input('请输入您要选择的滤波器：'))
    if score2==1:
        plt.subplot(321)
        plt.title(NoiseType)
        plt.imshow(img4, cmap='gray')

        for i in  kernelSize:
            list.append(smoothDenoising.mean_Fileter(img4,i))
            plt.subplot(j)
            plt.title('meanfilter_kernelsize_' + str(i))
            plt.imshow(list.pop(), cmap='gray')
            j = j + 1
        plt.show()
    elif score2 == 2:
        plt.subplot(321)
        plt.title(NoiseType)
        plt.imshow(img4, cmap='gray')

        for i in kernelSize:
            list.append(smoothDenoising.medium_Filter(img4, i))
            plt.subplot(j)
            plt.title('mediumfilter_kernelsize_' + str(i))
            plt.imshow(list.pop(), cmap='gray')
            j = j + 1
        plt.show()
    else:
        plt.subplot(321)
        plt.title(NoiseType)
        plt.imshow(img4, cmap='gray')

        for i in kernelSize:
            list.append(smoothDenoising.Gauss_Fileter(img4, i, 6))
            plt.subplot(j)
            plt.title('Gaussfilter_kernelsize_' + str(i))
            plt.imshow(list.pop(), cmap='gray')
            j = j + 1
        plt.show()


if __name__ == '__main__':
    main()
