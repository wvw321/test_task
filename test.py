import math
import random
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from sklearn.cluster import DBSCAN


class Datagen:
    def __init__(self,
                 max_h: int = 250, min_h: int = 150,
                 max_w: int = 250, min_w: int = 150,
                 max_angle: int = 89, min_angle: int = 0,
                 img_size: tuple = (640, 480)):
        self.max_h = max_h
        self.min_h = min_h
        self.max_w = max_w
        self.min_w = min_w
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.img_size = img_size

    def _color_list(self):
        color_list = []
        for name, _ in ImageColor.colormap.items():
            color_list.append(name)
        return color_list

    def _parameter_generator(self):
        # при использовании всех цветов из ImageColor.colormap
        # возникают проблемы при переходе в градации серого
        # теряется объект
        random.seed()
        list_color = ["red", "orange", "yellow", "green", "blue", "violet"]
        h = random.randrange(self.min_h, self.max_h + 1)
        if h & 1:
            h += 1
        w = random.randrange(self.min_w, self.max_w + 1)
        if w & 1:
            w += 1
        angle = random.randrange(self.min_angle, self.max_angle + 1)
        direction = random.randrange(0, 2)
        if direction != 1:
            angle = -angle

        x = random.randrange(0, self.img_size[0] + 1)
        y = random.randrange(0, self.img_size[1] + 1)
        background_color = random.choice(list_color)
        object_color = random.choice(list_color)

        if background_color == object_color:
            while background_color == object_color:
                object_color = random.choice(list_color)

        return h, w, angle, x, y, background_color, object_color

    def _rectangle_creation(self, h, w, angle):
        def turn(x, y, angl):
            angl = math.radians(angl)
            x_new = x * math.cos(angl) - y * math.sin(angl)
            y_new = x * math.sin(angl) + y * math.cos(angl)
            return int(x_new), int(y_new)

        h_half = h / 2
        w_half = w / 2

        x1 = -w_half
        y1 = h_half

        x2 = w_half
        y2 = h_half

        x3 = w_half
        y3 = -h_half

        x4 = -w_half
        y4 = -h_half

        x1, y1 = turn(x1, y1, angle)
        x2, y2 = turn(x2, y2, angle)
        x3, y3 = turn(x3, y3, angle)
        x4, y4 = turn(x4, y4, angle)
        return (x1, y1), (x2, y2), (x3, y3), (x4, y4)

    def _rectangle_bound_box(self, coord):
        x_max = max(coord, key=lambda item: item[0])[0]
        x_min = min(coord, key=lambda item: item[0])[0]
        y_max = max(coord, key=lambda item: item[1])[1]
        y_min = min(coord, key=lambda item: item[1])[1]

        return x_max, y_max, x_min, y_min

    def _box(self, rectangl_bounding_box):
        x_max, y_max, x_min, y_min = rectangl_bounding_box

        return (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)

    def _coordinate_corrector(self, rectangl_bounding_box, coordinates_rectangl, x, y):
        x_max, y_max, x_min, y_min = rectangl_bounding_box
        const = 15
        # "отступ от нижнего края изображения "
        x += x_max + const
        y += y_max + const
        # "отступ от нижнего края изображения "
        if x > self.img_size[0] - const - x_max:
            x -= x - (self.img_size[0] - const - x_max)
        if y > self.img_size[1] - const - y_max:
            y -= y - (self.img_size[1] - const - y_max)

        new_coordinates_rectangl = []
        for x_, y_ in coordinates_rectangl:
            new_x = int(x_ + x)
            new_y = int(y_ + y)
            new_coordinates_rectangl.append((new_x, new_y))

        box = self._box(rectangl_bounding_box)
        new_box = []
        for x_, y_ in box:
            new_x = x_ + x
            new_y = y_ + y
            new_box.append((new_x, new_y))

        return tuple(new_coordinates_rectangl), tuple(new_box)

    def _return_box_param(self, rectangl_bounding_box):
        (x_min, _), (x_max, y_max), (_, y_min), (_, _) = rectangl_bounding_box
        w = abs(x_max - x_min)
        h = abs(y_max - y_min)
        return x_min, y_min, w, h

    def img_creation(self):
        h, w, angle, x, y, background_color, object_color = self._parameter_generator()

        coordinates_rectangl = self._rectangle_creation(h, w, angle)
        rectangl_bounding_box = self._rectangle_bound_box(coordinates_rectangl)
        coordinates_rectangl, rectangl_bounding_box = self._coordinate_corrector(rectangl_bounding_box,
                                                                                 coordinates_rectangl, x, y)
        rectangl_bounding_box = self._return_box_param(rectangl_bounding_box)

        img = Image.new('RGB', self.img_size, background_color)
        draw = ImageDraw.Draw(img)
        draw.polygon(
            xy=coordinates_rectangl, fill=object_color, outline=None)

        # draw.polygon(
        #     xy=rectangl_bounding_box, outline=object_color)
        # img.show()
        return img, coordinates_rectangl, rectangl_bounding_box, background_color, object_color


def hough_transform(img_path: str, drawlines: bool = False):
    def accumulator(edge):
        H, W = edge.shape
        dtheta = 1
        # rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(int)
        # матрица аккумулятор Хафа
        hough = np.zeros((rho_max, 180), dtype=int)
        # координаты пикселей со значением 255 ( границ )
        ind = np.where(edge == 255)
        # zip-функция возвращает кортеж
        for y, x in zip(ind[0], ind[1]):
            for theta in range(0, 180, dtheta):
                # Перевод в полярные координаты
                t = np.pi / 180 * theta
                rho = int(x * np.cos(t) + y * np.sin(t))

                # запись в аккумулятор Хафа
                hough[rho, theta] += 1
        out = hough.astype(np.uint8)
        return out

    def return_coordinates(lines, H, W):
        lines_list = []
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            rho, t = arr
            out1 = []
            out2 = []
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out1.append((x, y))
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out2.append((x, y))
            out1.extend(out2)
            out = set(out1)
            lines_list.append(out)
        return lines_list

    def coefficients(line):
        line = list(line)
        x1 = line[0][0]
        x2 = line[100][0]
        y1 = line[0][1]
        y2 = line[100][1]
        k = (y2 - y1 + 0.00001) / (x2 - x1 + 0.00001)
        b = (y1 + 0.00001) / (k * x1 + 0.00001)
        return k, b

    def angle_lines(k1, k2):
        tg_y = abs((k1 - k2 + 0.00001) / (1 + (k1 * k2) + 0.00001))
        gamma = math.degrees(np.arctan(tg_y))
        return gamma

    def coordinates_of_line_intersection(lines_list):
        angles = []
        len_lines_list = len(lines_list)
        for i, list_ in enumerate(lines_list):
            k1, b1 = coefficients(list_)
            for j in range(i + 1, len_lines_list):
                k2, b2 = coefficients(lines_list[j])
                gamma = angle_lines(k1, k2)
                if gamma < 30: continue
                result = list_ & lines_list[j]
                if result:
                    for dot in result:
                        angles.append(dot)

        return angles

    def cluster_coord(box_list, conf: dict):
        clusters_boxes = defaultdict(list)
        clustered = DBSCAN(**conf).fit_predict(box_list)
        clustered = clustered.tolist()
        for step in range(len(box_list)):
            _class = clustered[step]

            value = box_list[step]
            clusters_boxes[_class].append(value)
        clusters_boxes = dict(clusters_boxes)
        return clusters_boxes

    def avg(clustered: dict):
        coordinate = []
        for key in clustered:
            cluster = clustered[key]
            if len(cluster) == 1:
                coordinate.append(cluster[0])
            else:
                sum_x = sum_y = 0
                len_ = len(cluster)
                for x, y in cluster:
                    sum_x += x
                    sum_y += y
                avg_x = sum_x / len_
                avg_y = sum_y / len_
                coordinate.append((int(avg_x), int(avg_y)))
        return coordinate

    def max_points(coordinate):

        x_max = max(coordinate, key=lambda item: item[0])[0]
        y_max = max(coordinate, key=lambda item: item[1])[1]
        x_min = min(coordinate, key=lambda item: item[0])[0]
        y_min = min(coordinate, key=lambda item: item[1])[1]

        x_max_list = []
        y_max_list = []
        x_min_list = []
        y_min_list = []
        for x, y in coordinate:
            if x == x_max:
                x_max_list.append((x, y))
            if x == x_min:
                x_min_list.append((x, y))
            if y == y_max:
                y_max_list.append((x, y))
            if y == y_min:
                y_min_list.append((x, y))

        coordinate_ = (x_min_list[len(x_min_list) - 1], y_max_list[len(y_max_list) - 1], x_max_list[0], y_min_list[0],)
        return coordinate_

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 100, 200, apertureSize=7)
    len_lines = 0
    threshold = 100

    while len_lines < 4:
        lines = cv2.HoughLines(edges, rho=0.5, theta=np.pi / 180, threshold=threshold)
        if lines is None:
            threshold -= 10

            continue
        len_lines = len(lines)
        threshold -= 10

    H, W, _ = img.shape
    lines_list = return_coordinates(lines, H, W)
    angles = coordinates_of_line_intersection(lines_list)
    conf_DBSCAN = {'eps': 5,
                   'min_samples': 1}
    clustered = cluster_coord(angles, conf_DBSCAN)
    coord = avg(clustered)
    coord = max_points(coord)
    accumulator_array = accumulator(edges)
    if drawlines is True:
        for list_line in lines_list:
            for x, y in list_line:
                img[y, x] = [0, 0, 0]
        return coord, accumulator_array, img

    return coord, accumulator_array


def loss(list_сoord1, list_сoord2):
    all_loss = 0
    for x1, y1 in list_сoord1:
        loss_ = []
        for x2, y2 in list_сoord2:
            loss = abs(x1 - x2) + abs(y1 - y2)
            loss_.append(loss)
        all_loss += min(loss_)
    return all_loss


if __name__ == '__main__':

    for i in range(5):
        img, coordinates_rectangl, rectangl_bounding_box, background_color, object_color = Datagen().img_creation()
        name = str(i) + ".png"
        img.save(name)
        print("x1=", coordinates_rectangl[0][0], "y1=", coordinates_rectangl[0][1])
        print("x2=", coordinates_rectangl[1][0], "y2=", coordinates_rectangl[1][1])
        print("x3=", coordinates_rectangl[2][0], "y3=", coordinates_rectangl[2][1])
        print("x4=", coordinates_rectangl[3][0], "y4=", coordinates_rectangl[3][1])
        print("x_min=", rectangl_bounding_box[0], "y_min=", rectangl_bounding_box[1], "w=", rectangl_bounding_box[2],
              "h=", rectangl_bounding_box[3])

        coord, accumulator_array, img1 = hough_transform(name, True)
        print("x1=", coord[0][0], "y1=", coord[0][1])
        print("x2=", coord[1][0], "y2=", coord[1][1])
        print("x3=", coord[2][0], "y3=", coord[2][1])
        print("x4=", coord[3][0], "y4=", coord[3][1])

        for x, y in coordinates_rectangl:
            img1 = cv2.circle(img1, (x, y), radius=3, color=(0, 0, 0), thickness=-1)

        loss_ = loss(coordinates_rectangl, coord)
        print("loss=", loss_)
        cv2.imwrite(str(i) + "_1.png", img1)
        cv2.imwrite(str(i) + "_2.png", accumulator_array)

        cv2.imshow("img1", img1)
        cv2.imshow("accumulator_array", accumulator_array)
        cv2.waitKey()
