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

    sigma = [5, 10, 15, 20, 25]
    list = []

    plt.subplot(321)
    plt.title(NoiseType)
    plt.imshow(img4, cmap='gray')

    j = 322
    for i in sigma:
        list.append(smoothDenoising.Gauss_Fileter(img4, 5, i))
        plt.subplot(j)
        plt.title('Gaussfilter_sigma=' + str(i))
        plt.imshow(list.pop(), cmap='gray')
        j = j + 1

    plt.show()

if __name__ == '__main__':
    main()
