try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    import skimage.transform as st

    from skimage.feature import peak_local_max
    #c_image = st.resize(c_image, (int(c_image.shape[0]*1), int(c_image.shape[1]*1)))

    green = c_image[:,:,1]
    red = c_image[:,:,0]

    filter_for_green = np.array([[-2, -2, -2, -2, -2],
                                [-2, -2, -2, -2, -2],
                                [-2, -1, -0.5, -1, -2],
                                [-1, 3, 5, 3, -1],
                                [-0.5, 5, 6, 5, -0.5],
                                [-1, 3, 5, 3, -1],
                                [-2, -1, -0.5, -1, -2]])

    filter_for_red = np.array([[-2, -1, -0.5, -1, -2],
                               [-1, 3, 5, 3, -1],
                               [-0.5, 5, 6, 5, -0.5],
                               [-1, 3, 5, 3, -1],
                               [-2, -1, -0.5, -1, -2],
                               [-2, -2, -2, -2, -2],
                               [-2, -2, -2, -2, -2]])


    green_processed_image = sg.convolve2d(green, filter_for_green)
    red_processed_image = sg.convolve2d(red, filter_for_red)

    coordinates_red = peak_local_max(red_processed_image, min_distance=20, num_peaks=10)
    coordinates_green = peak_local_max(green_processed_image, min_distance=20, num_peaks=10)
    colors = [1]*len(coordinates_red)
    new_points = []
    for p in coordinates_red:
        new_points.append(p[::-1])
    for green in coordinates_green:
        for red in coordinates_red:
            if abs(green[0] - red[0]) < 20 and abs(green[1] - red[1]):
                # TODO: check RGB
                break
        else:
            new_points.append(green[::-1])
            colors.append(0)

    # red_x = [i[1]*1 for i in coordinates_red]
    # red_y = [i[0]*1 for i in coordinates_red]
    # green_x = [i[1]*1 for i in coordinates_green]
    # green_y = [i[0]*1 for i in coordinates_green]

    #first line is for other parts:
    return new_points, colors
    #return red_x, red_y, green_x, green_y


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
    plt.show(block=True)

def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'tfl_images_for_part_1'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
