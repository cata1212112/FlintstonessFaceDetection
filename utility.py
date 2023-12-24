import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def plot_window_aspect_ratio():
    files = ["antrenare/barney_annotations.txt", "antrenare/betty_annotations.txt", "antrenare/fred_annotations.txt", "antrenare/wilma_annotations.txt"]
    aspect_ratios = []
    heights = []
    widths = []

    positive_examples = 0

    for file in files:
        with open(file, "r") as f:
            for line in f:
                positive_examples += 1
                points = list(map(int, line.split()[1:5]))
                w = points[3] - points[1]
                h = points[2] - points[0]
                aspect_ratios.append(w / h)
                heights.append(h)
                widths.append(w)

    aspect_ratios = np.array(aspect_ratios)
    plt.hist(aspect_ratios, bins=30, density=True)
    x_ticks = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    plt.xticks(x_ticks)
    plt.xlabel("Window Aspect Ratio")
    plt.show()

    heights = np.array(heights)
    plt.hist(heights, bins=30)
    plt.xlabel("Height")
    plt.show()

    widths = np.array(widths)
    plt.hist(widths, bins=30)
    plt.xlabel("Width")
    plt.show()

    print(positive_examples)

def create_images_with_annotations():
    pass

def intersect(window, boxes):
    for box in boxes:
        if not (box[2] < window[0] or window[3] < box[1] or window[2] < box[0] or box[3] < window[1]):
            return True
    return False

def choose_non_intersecting_window(image_size, window_size, image_boxes):
    windows = []
    for __ in range(2):
        def get_window():
            for _ in range(100):
                window_y = np.random.randint(0, image_size[0] - window_size[0] + 1)
                window_x = np.random.randint(0, image_size[1] - window_size[1] + 1)

                window = (window_x, window_y, window_x + window_size[0], window_y + window_size[1])

                if not intersect(window, image_boxes):
                    return window

        window = get_window()
        if window != None:
            windows.append(window)
            image_boxes.append(window)

    return windows

def draw_rectangles_on_image_and_save(image, rectangles, file):
    for (rect, color) in rectangles:
        image = cv.rectangle(image, (rect[0], rect[1]), (rect[2], rect[3]), color, 3)

    cv.imwrite(file, image)

def intersection_over_union(a, b):
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])

    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    b_area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)

    return intersection_area / float(a_area + b_area - intersection_area)

def non_maximum_suppression(detections, scores):
    sorted_indices = np.flipud(np.argsort(scores))
    sorted_detections = detections[sorted_indices]
    sorted_scores = scores[sorted_indices]

    is_maximal = np.ones(len(detections), dtype=bool)
    iou_threshold = 0.3

    for i in range(len(sorted_detections) - 1):
        if is_maximal[i] == True:
            for j in range(i+1, len(sorted_detections)):
                if is_maximal[j]:
                    if intersection_over_union(sorted_detections[i], sorted_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:
                        c_x = (sorted_detections[j][0] + sorted_detections[j][2]) / 2
                        c_y = (sorted_detections[j][1] + sorted_detections[j][3]) / 2

                        if sorted_detections[i][0] <= c_x <= sorted_detections[i][2] and sorted_detections[i][1] <= c_y <= sorted_detections[i][3]:
                            is_maximal[j] = False

    return sorted_detections[is_maximal], sorted_scores[is_maximal]


def compute_average_precision(rec, prec):
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision