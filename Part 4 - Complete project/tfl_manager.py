from part_1_find_lights import *
from part_3_SFM import *
from PIL import ImageOps

from tensorflow.keras.models import load_model


class TFL_manager:
    def __init__(self, pp, focal):
        self.pp = pp
        self.focal = focal
        self.all_tfls = []
        self.prev_img = None
        self.CNN = load_model("model.h5")

    def add_border(self, image):
        return ImageOps.expand(image, border=41, fill='black')

    def find_tfls(self, curr_image, points, colors):
        img = self.add_border(curr_image)
        img = np.array(img, dtype='uint8')
        tfls = []
        tfl_colors = []
        cropped = []
        for p in points:
            x = p[1]+41
            y = p[0]+41
            a = img[x - 40:x + 41, y - 40:y + 41] #

            cropped.append(a)

        predictions = self.CNN.predict(np.array(cropped))
        for index, predict in enumerate(predictions[:, 1]):
            if predict > 0.5:
                tfls.append(points[index])
                tfl_colors.append(colors[index])
        good_tfls = []
        good_colors = []
        good_prev = []
        if len(self.all_tfls) == 0:
            return tfls, tfl_colors

        for i in range(len(tfls)):
            for j in range(len(self.all_tfls[-1])):
                if abs(tfls[i][0] - self.all_tfls[-1][j][0]) < 30 and abs(tfls[i][1] - self.all_tfls[-1][j][1]) < 30:
                    good_tfls.append(tfls[i])
                    good_colors.append(colors[i])
                    good_prev.append(self.all_tfls[-1][j])
                    break

        self.all_tfls[-1] = good_prev[:]
        return good_tfls, good_colors

    def find_distances(self, curr_image, prev_image, ego_motion):
        prev_container = FrameContainer(prev_image)
        curr_container = FrameContainer(curr_image)
        prev_container.traffic_light = np.array(self.all_tfls[-2])
        curr_container.traffic_light = np.array(self.all_tfls[-1])
        curr_container.EM = ego_motion
        curr_container = calc_TFL_dist(prev_container, curr_container, self.focal, self.pp)
        return prev_container, curr_container


    def split_points(self, points, colors):
        red_x = []
        red_y = []
        green_x = []
        green_y = []
        for i in range(len(points)):
            if colors[i] == 1:
                red_x.append(points[i][0])
                red_y.append(points[i][1])
            elif colors[i] == 0:
                green_x.append(points[i][0])
                green_y.append(points[i][1])
        return red_x, red_y, green_x, green_y


    def plot_light_points(self, points, colors, part):
        red_x, red_y, green_x, green_y = self.split_points(points, colors)
        part.plot(red_x, red_y, 'ro', color='r', markersize=4)
        part.plot(green_x, green_y, 'ro', color='g', markersize=4)


    def plot_distances(self, curr_container, foe, rot_pts, part):
        curr_p = curr_container.traffic_light
        if len(curr_p) > 0:
            part.plot(curr_p[:, 0], curr_p[:, 1], 'b+')
        for i in range(len(curr_p)):
            part.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
            if curr_container.valid[i]:
                part.text(curr_p[i, 0], curr_p[i, 1],
                                r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
        part.plot(foe[0], foe[1], 'r+')
        part.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')


    def set_subplot(self, title, part, image):
        part.set_title(title)
        part.imshow(image)


    def visualize_all_parts(self, points, colors, image1, tfls, tfl_colors, prev_container, curr_container):
        fig, (first_part, second_part, third_part) = plt.subplots(1, 3,  figsize=(12, 6))
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, self.focal, self.pp)
        norm_rot_pts = rotate(norm_prev_pts, R)
        rot_pts = unnormalize(norm_rot_pts, self.focal, self.pp)
        foe = np.squeeze(unnormalize(np.array([norm_foe]), self.focal, self.pp))

        self.set_subplot('Lights:', first_part, image1)
        self.set_subplot('Traffic Lights:', second_part, image1)
        self.set_subplot('Distances:', third_part, image1)

        self.plot_light_points(points, colors, first_part)
        self.plot_light_points(tfls, tfl_colors, second_part)
        self.plot_distances(curr_container, foe, rot_pts, third_part)

        plt.show(block=True)

    def manage(self, curr_image, ego_motion):
        img1 = np.array(curr_image, dtype='uint8')
        points, colors = find_tfl_lights(img1)

        tfls, tfl_colors = self.find_tfls(curr_image, points, colors)
        self.all_tfls.append(tfls)
        if len(self.all_tfls) > 1:
            prev_container, curr_container = self.find_distances(curr_image, self.prev_img, ego_motion)
            self.visualize_all_parts(points, colors, img1, tfls, tfl_colors, prev_container, curr_container)

        self.prev_img = curr_image


