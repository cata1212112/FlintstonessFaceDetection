import os
import numpy as np
import matplotlib.pyplot as plt
from constants import *
from PIL import ImageDraw, Image


def intersect(window, boxes):
    for box in boxes:
        if not (box[2] < window[0] or window[3] < box[1] or window[2] < box[0] or box[3] < window[1]):
            return True
    return False


def choose_non_intersecting_window(image_size, rectangles, image_boxes):
    windows = []
    for __ in range(4):

        window_index = np.random.choice(len(rectangles), 1)[0]
        window = rectangles[window_index]

        def get_window(window):
            for _ in range(100):
                window_y = np.random.randint(0, image_size[1] - window[1])
                window_x = np.random.randint(0, image_size[0] - window[0])

                win = (window_x, window_y, window_x + window[0], window_y + window[1])

                if not intersect(win, image_boxes):
                    return win

        window = get_window(window)
        if window is not None:
            windows.append(window)
            image_boxes.append(window)

    return windows


def draw_rectangles_on_image_and_save(image, rectangles, file):
    for (rect, color) in rectangles:
        draw = ImageDraw.Draw(image)
        draw.polygon([(rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1])],
                     outline=color,
                     width=2)

    image.save(file)


def intersection_over_union(a, b):
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    return intersection_area / float(a_area + b_area - intersection_area)


def intersection_over_minimum_size(a, b):
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    return intersection_area / float(min(a_area, b_area))


def non_maximum_suppression(detections, scores):
    sorted_indices = np.flipud(np.argsort(scores))
    sorted_detections = detections[sorted_indices]
    sorted_scores = scores[sorted_indices]

    sorted_detections = sorted_detections[:20]
    sorted_scores = sorted_scores[:20]

    is_maximal = np.ones(len(sorted_detections), dtype=bool)
    iou_threshold = 0.3

    for i in range(len(sorted_detections) - 1):
        if is_maximal[i] == True:
            for j in range(i + 1, len(sorted_detections)):
                if is_maximal[j] == True:
                    if intersection_over_minimum_size(sorted_detections[i], sorted_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    if intersection_over_union(sorted_detections[i], sorted_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:
                        c_x = (sorted_detections[j][0] + sorted_detections[j][2]) / 2
                        c_y = (sorted_detections[j][1] + sorted_detections[j][3]) / 2

                        if sorted_detections[i][0] <= c_x <= sorted_detections[i][2] and sorted_detections[i][
                            1] <= c_y <= sorted_detections[i][3]:
                            is_maximal[j] = False

    return sorted_detections[is_maximal], sorted_scores[is_maximal]

def plot_heatmap_width_height(widths, heights, file):
    widths = np.array(widths)
    heights = np.array(heights)
    hist, xedges, yedges = np.histogram2d(widths, heights, bins=10)

    plt.figure()
    plt.imshow(hist, cmap='plasma', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto',
               origin='lower')
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("2D Heatmap of Widths and Heights")
    plt.colorbar(label="Frequency")
    plt.savefig(file)
    plt.show()


def plot_histogram(hist, name, file):
    aspect_ratios = np.array(hist)
    plt.hist(aspect_ratios, bins=30, density=True)
    x_ticks = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    plt.xticks(x_ticks)
    plt.xlabel(name)
    plt.savefig(file)
    plt.show()

