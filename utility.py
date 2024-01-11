import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import rectangle_perimeter
from constants import *
from skimage.color import rgb2gray
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
        if window != None:
            windows.append(window)
            image_boxes.append(window)

    return windows


def draw_rectangles_on_image_and_save(image, rectangles, file, scores=None):

    if scores is None:
        for (rect, color) in rectangles:
            draw = ImageDraw.Draw(image)
            draw.polygon([(rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1])], outline=color,
                         width=2)

        image.save(file)
    else:
        for (rect, color), score in zip(rectangles, scores):
            draw = ImageDraw.Draw(image)
            draw.polygon([(rect[0], rect[1]), (rect[0], rect[3]), (rect[2], rect[3]), (rect[2], rect[1])], outline=color,
                         width=2)
            draw.text((rect[0], rect[1]), str(score), fill=(255, 0, 0))

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

    sorted_detections = sorted_detections[:10]
    sorted_scores = sorted_scores[:10]

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


def compute_average_precision(rec, prec):
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision


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


def sliding_window(image, step_size, window_size):
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]

def get_validation_images():
    files = os.listdir("validare/validare")
    for file in files:
        image = Image.open(os.path.join("validare/validare", file))
        image = image.convert("RGB")
        yield image, file

def compute_average_precision(rec, prec):
    # functie adaptata din 2010 Pascal VOC development kit
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision

def eval_detections(detections, scores, file_names, ground_truth_file="validare/task1_gt_validare.txt", plot_name=""):
    ground_truth_file = np.loadtxt(ground_truth_file, dtype='str')
    ground_truth_file_names = np.array(ground_truth_file[:, 0])
    ground_truth_detections = np.array(ground_truth_file[:, 1:], np.uint32)

    num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
    gt_exists_detection = np.zeros(num_gt_detections)
    # sorteazam detectiile dupa scorul lor
    sorted_indices = np.argsort(scores)[::-1]
    file_names = file_names[sorted_indices]
    scores = scores[sorted_indices]
    detections = detections[sorted_indices]

    num_detections = len(detections)
    true_positive = np.zeros(num_detections)
    false_positive = np.zeros(num_detections)
    duplicated_detections = np.zeros(num_detections)

    for detection_idx in range(num_detections):
        indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

        gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
        bbox = detections[detection_idx]
        max_overlap = -1
        index_max_overlap_bbox = -1
        for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
            overlap = intersection_over_union(bbox, gt_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
                index_max_overlap_bbox = indices_detections_on_image[gt_idx]

        # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
        if max_overlap >= 0.3:
            if gt_exists_detection[index_max_overlap_bbox] == 0:
                true_positive[detection_idx] = 1
                gt_exists_detection[index_max_overlap_bbox] = 1
            else:
                false_positive[detection_idx] = 1
                duplicated_detections[detection_idx] = 1
        else:
            false_positive[detection_idx] = 1

    cum_false_positive = np.cumsum(false_positive)
    cum_true_positive = np.cumsum(true_positive)

    rec = cum_true_positive / num_gt_detections
    prec = cum_true_positive / (cum_true_positive + cum_false_positive)
    average_precision = compute_average_precision(rec, prec)
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Average precision {plot_name} %.3f' % average_precision)
    plt.savefig(f"precizie_medie{plot_name}.png")
    plt.show()

def get_training_images():
    for character in characters:
        annotation_file = f"antrenare/{character}_annotations.txt"

        images_file = f"antrenare/{character}"

        images = os.listdir(images_file)

        for img_name in images:
            file = os.path.join(images_file, img_name)

            image = Image.open(file)
            image = image.convert("RGB")

            boxes = []

            with open(annotation_file, "r") as f:
                for line in f:
                    image_name, xmin, ymin, xmax, ymax, ch = line.split()
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)


                    if image_name == img_name:
                        boxes.append([xmin, ymin, xmax, ymax])

            yield image, boxes

def hard_mining(model):
    cnt = 1
    for image, boxes in get_training_images():
        detections, scores, file_names = model.predict(image.copy(), "hard_mining")

        def get_detection():
            for detection, score in zip(detections, scores):
                ious = [intersection_over_union(detection, box) for box in boxes]

                if max(ious) < 0.3:
                    patch = image.crop(detection)
                    patch.save(f"hard_mining/{cnt}.jpg")
                    draw_rectangles_on_image_and_save(image, [(w, (0, 255, 0)) for w in [detection]],f"hard_mining_rectangles/{cnt}_{score}.jpg")
                    return

        get_detection()
        cnt += 1


def get_character_faces():
    cnt = 1
    for character in characters:
        if character == "unknown":
            continue
        annotation_file = f"antrenare/{character}_annotations.txt"

        with open(annotation_file, "r") as f:
            for line in f:
                image_name, xmin, ymin, xmax, ymax, ch = line.split()
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)

                image_path = f"{character}/{image_name}"

                image = Image.open(os.path.join("antrenare", image_path))
                image = image.convert("RGB")

                patch = image.crop((xmin, ymin, xmax, ymax))
                # patch.save(f"fete/{cnt}_{ch}.jpg")
                # cnt += 1
                # patch = rgb2gray(np.array(patch))

                yield patch, ch

