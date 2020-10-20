import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import smoothDenoising

'''
    ###################################
    此函数用于观察图片加上各种噪声后的结果
    ###################################

'''
def main():
    Image.MAX_IMAGE_PIXELS = None
    img = np.array(Image.open('LenaRGB.bmp'))
    #原图像灰度图
    r, g, b = [img[:, :, i] for i in range(3)]
    img_gray = r * 0.299 + g * 0.587 + b * 0.114

    # 加上各种噪声
    img1 = smoothDenoising.add_PulseNoise(img, 0.9)
    img2 = smoothDenoising.add_Salt_PepperNoise(img, 0.9)
    img3 = smoothDenoising.add_Gauss_Noise(img, 0, 20)


    plt.subplot(321)
    plt.title('original image')
    plt.imshow(img_gray,cmap='gray')
    plt.subplot(322)
    plt.title('PulseNoise, SNR=0.9')
    plt.imshow(img1, cmap='gray')
    plt.subplot(323)
    plt.title('Salt_PepperNoise, SNR=0.9')
    plt.imshow(img2, cmap='gray')
    plt.subplot(324)
    plt.title('GaussNoise, sigma = 20')
    plt.imshow(img3, cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()


