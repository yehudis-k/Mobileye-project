import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
from random import randint


def open_images(data_path):
    subdirs = [x[0] for x in os.walk(data_path)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for file in files:
                if file.endswith('.png'):
                    label_path = subdir.replace('leftImg8bit_trainvaltest/leftImg8bit', 'gtFine_trainvaltest/gtFine')
                    label_path += "/" + file.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
                    original_path = os.path.join(subdir, file)
                    treat_image(original_path, label_path)


def crop(img, x, y):
    cropped = img[x-10:x + 71, y-20:y + 61]
    return cropped


def add_border(image):
    return ImageOps.expand(image, border=71, fill='black')


def add_noise(img):
    noise = np.random.randint(15, size=img.shape, dtype='uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if (img[i][j][k] != 255):
                    img[i][j][k] += noise[i][j][k]

    return img


def treat_image(data_path, label_path):
    if "/train" in data_path:
        type = "train"
    else:
        type = "val"

    label = Image.open(label_path)
    label = add_border(label)
    label = np.array(label, dtype='uint8')

    img = Image.open(data_path)
    img = add_border(img)
    image = np.array(img, dtype='uint8')

    tfl = np.argwhere(label == 19)

    if len(tfl) > 0:
        cropped_tfl = crop(image, tfl[0][0], tfl[0][1])
        add_to_file(cropped_tfl, 1, type)
        cropped_tfl = np.flip(cropped_tfl, 1)
        add_to_file(cropped_tfl, 1, type)
        #noise = add_noise(cropped_tfl)
        #add_to_file(noise, 1, type)
        while True:
            i = randint(71, len(image)-71)
            new = crop(label, i, i)
            if len(np.argwhere(new == 19)) == 0:
                cropped_not_tfl = crop(image, i, i)
                add_to_file(cropped_not_tfl, 0, type)
                cropped_not_tfl = np.flip(cropped_not_tfl, 0)
                add_to_file(cropped_not_tfl, 0, type)
                #noise = add_noise(cropped_not_tfl)
                #add_to_file(noise, 0, type)
                break


def add_to_file(data, labels, type):
    data_array = np.array(data, dtype='uint8')
    f = open(f"cropped_data/{type}/data.bin", "ab")
    data_array.tofile(f)
    f.close()
    label_array = np.array(labels, dtype='uint8')
    f = open(f"cropped_data/{type}/labels.bin", "ab")
    label_array.tofile(f)
    f.close()


def show_with_label(path, index):
    d_file = np.memmap(path + '/data.bin', offset=index*(81*81*3), shape = (81,81,3))
    l_file = np.memmap(path + '/labels.bin', offset=index, shape = (1,))
    plt.imshow(d_file)
    if l_file == 0:
        plt.title("no traffic light")
    else:
        plt.title("TRAFFIC LIGHT!!")
    plt.show()
    print(l_file)

# open_images('leftImg8bit_trainvaltest/leftImg8bit/train')
# open_images('leftImg8bit_trainvaltest/leftImg8bit/val')
# for i in range(100):
#     show_with_label("cropped_data/train", i)

