from utility import *
import os
from constants import *
from PIL import Image
from tqdm import tqdm


class Generator:
    def __init__(self):
        self.boxes = {}
        self.rectangles = []

    def generate_positive_examples(self):

        widths = []
        heights = []
        aspect_ratio = []

        for character in characters:
            if character == 'unknown':
                continue
            annotation_file = f"antrenare/{character}_annotations.txt"

            with open(annotation_file, "r") as f:
                for line in f:
                    image_name, xmin, ymin, xmax, ymax, ch = line.split()
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)

                    w = xmax - xmin
                    h = ymax - ymin

                    widths.append(w)
                    heights.append(h)

                    aspect_ratio.append(w / h)

                    self.rectangles.append((w, h))

                    image_path = f"{character}/{image_name}"
                    if image_path not in self.boxes:
                        self.boxes[image_path] = [(xmin, ymin, xmax, ymax)]
                    else:
                        self.boxes[image_path].append((xmin, ymin, xmax, ymax))

        for image_path in self.boxes:
            image = Image.open(os.path.join("antrenare", image_path))
            image = image.convert("RGB")

            for box in self.boxes[image_path]:
                patch = image.crop(box)
                patch.save(f"pozitive/{image_path.split('/')[1].split('.')[0]}_{box[0]}_{box[1]}_{box[2]}_{box[3]}.jpg")
                patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
                patch.save(
                    f"pozitive/{image_path.split('/')[1].split('.')[0]}_{box[0]}_{box[1]}_{box[2]}_{box[3]}_flip.jpg")

        plot_heatmap_width_height(widths, heights, "width_height.png")
        plot_histogram(aspect_ratio, "Windows aspect ratio", "aspect_ratio.png")

    def generate_negative_examples(self):
        for character in characters:
            for i in range(1, 1001):
                image_path = f"antrenare/{character}/{str(i).zfill(4)}.jpg"
                image = Image.open(image_path)
                image = image.convert("RGB")

                windows = choose_non_intersecting_window(image.size, self.rectangles,
                                                         self.boxes[f"{character}/{str(i).zfill(4)}.jpg"].copy())
                draw_rectangles_on_image_and_save(image.copy(),
                                                  [(win, (255, 0, 0)) for win in windows] + [(win, (0, 255, 0)) for win
                                                                                             in self.boxes[
                                                                                                 f"{character}/{str(i).zfill(4)}.jpg"]],
                                                  f"dreptunghiuri/{image_path.split('/')[1]}_{image_path.split('/')[2].split('.')[0]}.jpg")
                for win in windows:
                    patch = image.crop(win)
                    patch.save(
                        f"negative/{image_path.split('/')[1]}_{image_path.split('/')[2].split('.')[0]}_{win[0]}_{win[1]}_{win[2]}_{win[3]}.jpg")

    @classmethod
    def get_training_images(cls):
        boxes = {}
        for character in characters:
            if character == 'unknown':
                continue
            annotation_file = f"antrenare/{character}_annotations.txt"

            with open(annotation_file, "r") as f:
                for line in f:
                    image_name, xmin, ymin, xmax, ymax, ch = line.split()
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)

                    name = f"{character}/{image_name}"

                    if name not in boxes:
                        boxes[name] = [(xmin, ymin, xmax, ymax)]
                    else:
                        boxes[name].append([xmin, ymin, xmax, ymax])

        for character in characters:
            if character == "unknown":
                continue
            for i in range(1, 1001):
                image_path = f"antrenare/{character}/{str(i).zfill(4)}.jpg"
                image = Image.open(image_path)
                image = image.convert("RGB")

                yield image, boxes[f"{character}/{str(i).zfill(4)}.jpg"]

    @classmethod
    def hard_mining(cls, model):
        cnt = 1
        for image, boxes in tqdm(cls.get_training_images(), total=4000):
            detections, scores, file_names = model.predict(image.copy(), "hard_mining")

            def get_detection():
                for detection, score in zip(detections, scores):
                    ious = [intersection_over_union(detection, box)[0] for box in boxes]

                    if max(ious) < 0.3:
                        patch = image.crop(detection)
                        patch.save(f"hard_mining/{cnt}.jpg")
                        draw_rectangles_on_image_and_save(image, [(w, (0, 255, 0)) for w in [detection]],
                                                          f"hard_mining_rectangles/{cnt}_{score}.jpg")
                        return

            get_detection()
            cnt += 1

    @classmethod
    def get_character_faces(cls):
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

                    yield patch, ch

    @classmethod
    def get_test_images(cls, path):
        files = os.listdir(path)
        for file in files:
            image = Image.open(os.path.join(path, file))
            image = image.convert("RGB")
            yield image, file
