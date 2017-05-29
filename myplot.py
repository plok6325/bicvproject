

'''this is file contain functions of plotting things '''


import matplotlib.pyplot as plt
from scipy.misc import imresize

def plot_comparison(O_image, down_image):
    O_size = O_image.shape
    plt.subplot(2,3,1)
    plt.imshow(O_image)
    plt.title('original size')

    plt.subplot(2,3,2)
    imrecov = imresize(down_image,size=O_size,interp='nearest')
    plt.imshow(imrecov)
    plt.title('nearest ')

    plt.subplot(2,3,3)
    imrecov = imresize(down_image,size=O_size,interp='lanczos')
    plt.imshow(imrecov)
    plt.title('lanczos ')

    plt.subplot(2,3,4)
    imrecov = imresize(down_image,size=O_size,interp='bilinear')
    plt.imshow(imrecov)
    plt.title('bilinear ')

    plt.subplot(2,3,5)
    imrecov = imresize(down_image,size=O_size,interp='cubic')
    plt.imshow(imrecov)
    plt.title('cubic ')


    plt.subplot(2,3,6)
    imrecov = imresize(down_image,size=O_size,interp='bicubic')
    plt.imshow(imrecov)
    plt.title('bicubic ')

    plt.show()


def plot_2(original,SR):
    plt.subplot(211)
    plt.title('original')
    plt.imshow(original)
    plt.subplot(212)
    plt.title('SR')
    plt.imshow(SR)
    plt.show()

def plot_3(O_image,down_image,SR):
    O_size = O_image.shape
    plt.subplot(2,3,1)
    plt.imshow(O_image)
    plt.title('original size')

    plt.subplot(2,3,2)
    imrecov = imresize(down_image,size=O_size,interp='nearest')
    plt.imshow(imrecov)
    plt.title('nearest ')

    plt.subplot(2,3,3)
    imrecov = imresize(down_image,size=O_size,interp='lanczos')
    plt.imshow(imrecov)
    plt.title('lanczos ')

    plt.subplot(2,3,4)
    imrecov = imresize(down_image,size=O_size,interp='bilinear')
    plt.imshow(imrecov)
    plt.title('bilinear ')

    plt.subplot(2,3,5)
    imrecov = imresize(down_image,size=O_size,interp='bicubic')
    plt.imshow(imrecov)
    plt.title('bicubic ')


    plt.subplot(2,3,6)
    plt.imshow(SR)
    plt.title('Super R ')

    plt.show()
