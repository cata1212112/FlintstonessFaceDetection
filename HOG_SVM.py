import os
import cv2 as cv
import matplotlib.pyplot as plt

from constants import *
from skimage.feature import hog
import numpy as np
import pickle
from utility import *
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_validate
from sklearn import metrics
from tqdm import tqdm
from sklearn.decomposition import PCA
from skimage.transform import pyramid_gaussian
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.neural_network import MLPClassifier

class HogSVMClassifier:
    def __init__(self):
        self.window_sizes = [(64, 64)]
        self.pyramid_scales = 7
        self.boxes_per_image = {}
        self.orientations = 13
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (4, 4)

    def get_train_positive_descriptors(self):
        if not LOAD_POZITIVES:
            rectangles = []
            X_train = []
            y_train = []
            widths = []
            heigths = []
            cnt = 0
            for character in characters:
                annotation_file = f"antrenare/{character}_annotations.txt"
                image = None
                opened_image_name = None
                with open(annotation_file, 'r') as f:
                    for line in f:
                        image_name, xmin, ymin, xmax, ymax, ch = line.split()
                        xmin = int(xmin)
                        ymin = int(ymin)
                        xmax = int(xmax)
                        ymax = int(ymax)

                        if xmax - xmin >= self.window_sizes[0][0] and ymax - ymin >= self.window_sizes[0][1]:
                            rectangles.append((xmax - xmin, ymax - ymin))
                            widths.append(xmax - xmin)
                            heigths.append(ymax - ymin)

                        cnt += 1
                        dict_name = f"antrenare/{character}/{image_name}"
                        if dict_name not in self.boxes_per_image:
                            self.boxes_per_image[dict_name] = [(xmin, ymin, xmax, ymax)]
                        else:
                            self.boxes_per_image[dict_name].append((xmin, ymin, xmax, ymax))

                        if opened_image_name != image_name:
                            cnt = 1
                            image = cv.imread(f"antrenare/{character}/{image_name}")
                            opened_image_name = image_name

                        if xmax - xmin >= self.window_sizes[0][0] and ymax - ymin >= self.window_sizes[0][1]:
                            patch = image[ymin:ymax + 1, xmin:xmax + 1, :]
                            patch = cv.resize(patch, self.window_sizes[0], interpolation=cv.INTER_AREA)

                            cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch)

                            # features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)

                            # X_train.append(features)
                            # y_train.append(character_mapping[ch])

                            patch = cv.flip(patch, flipCode=1)
                            # features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)
                            cnt += 1
                            cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch)

                            # patch = cv.flip(patch, flipCode=0)
                            # # features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)
                            # cnt += 1
                            # cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch)
                            # #
                            # X_train.append(features)
                            # y_train.append(character_mapping[ch])
                            #
                            # patch = cv.flip(patch, flipCode=1)
                            # features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)
                            # cnt += 1
                            # cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch)
                            #
                            # X_train.append(features)
                            # y_train.append(character_mapping[ch])
                            #
                            # patch_45 = rotate_image(patch.copy(), 45)
                            # features = get_features(patch_45, self.orientations, self.pixels_per_cell, self.cells_per_block)
                            # cnt += 1
                            # cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch_45)
                            #
                            # X_train.append(features)
                            # y_train.append(character_mapping[ch])
                            #
                            # patch_135 = rotate_image(patch.copy(), 135)
                            # features = get_features(patch_135, self.orientations, self.pixels_per_cell, self.cells_per_block)
                            # cnt += 1
                            # cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch_135)
                            #
                            # X_train.append(features)
                            # y_train.append(character_mapping[ch])


            widths = np.array(widths)
            heights = np.array(heigths)

            plt.hist(heights, bins=30)
            plt.xlabel("Height pozitive")
            plt.show()

            plt.hist(widths, bins=30)
            plt.xlabel("Width pozitive")
            plt.show()

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            np.save("pozitive/x_train", X_train)
            np.save("pozitive/y_train", y_train)

            with open("pozitive/boxes.pkl", 'wb') as f:
                pickle.dump(self.boxes_per_image, f)

        else:
            X_train = np.load("pozitive/x_train.npy")
            y_train = np.load("pozitive/y_train.npy")

            with open("pozitive/boxes.pkl", 'rb') as f:
                self.boxes_per_image = pickle.load(f)

            return X_train, y_train, None

        return X_train, y_train, rectangles

    def get_negative_descriptors(self, rectangles):
        if not LOAD_NEGATIVES:
            widths = []
            heigths = []
            X_train = []
            all_images = []
            for i in range(1, 1001):
                for character in characters:
                    all_images.append(f"antrenare/{character}/{i:04d}.jpg")

            for img in all_images:
                image_boxes = self.boxes_per_image[img].copy()
                image = cv.imread(img)
                to_draw = image.copy()
                original_image = image.copy()

                scale = np.random.randint(0, self.pyramid_scales + 1)
                downscale = 1.25
                upscale = 1
                cnt = 0
                scaled_image = None
                ok = False
                for im_scaled in pyramid_gaussian(image, downscale=downscale, max_layer=scale, preserve_range=True,
                                                  channel_axis=2):
                    scaled_image = im_scaled.copy()
                    if not ok:
                        ok = True
                    else:
                        upscale *= 1.25
                        image_boxes = [(x[0] / downscale, x[1] / downscale, x[2] / downscale, x[3] / downscale) for x in
                                       image_boxes.copy()]

                image = scaled_image
                window_size_index = np.random.choice(len(self.window_sizes))
                window_size = self.window_sizes[window_size_index]

                windows = choose_non_intersecting_window(image.shape, window_size, image_boxes.copy())

                rectangles_to_draw = [
                    ((int(x[0]), int(x[1]), int(x[2]), int(x[3])), (0, 0, 255)) for x in
                    self.boxes_per_image[img].copy()]

                for i in range(len(windows)):
                    window = windows[i]
                    patch = image[window[1]:window[3], window[0]:window[2], :]
                    window = (int(window[0] * upscale), int(window[1] * upscale), int(window[2] * upscale),
                              int(window[3] * upscale))

                    widths.append(window[2] - window[0])
                    heigths.append(window[3] - window[1])
                    rectangles_to_draw.append((window, (0, 255, 0)))
                    patch = np.round(cv.resize(patch, self.window_sizes[0], interpolation=cv.INTER_AREA))
                    patch = np.array(patch, dtype=np.uint8)
                    filename = f"negative/{i}_" + img[10:].replace('/', '_')
                    cv.imwrite(filename, patch)

                    features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)

                    X_train.append(features)

                filename = "dreptunghiuri/" + img[10:].replace('/', '_')
                draw_rectangles_on_image_and_save(to_draw, rectangles_to_draw, filename)

            widths = np.array(widths)
            heights = np.array(heigths)

            plt.hist(heights, bins=30)
            plt.xlabel("Height negative")
            plt.show()

            plt.hist(widths, bins=30)
            plt.xlabel("Width negative")
            plt.show()

            X_train = np.array(X_train)
            shuffle = np.random.permutation(len(X_train))
            X_train = X_train[shuffle]
            # X_train = X_train[:13954]
            np.save("negative/x_train", X_train)
            return X_train

        else:
            X_train = np.load("negative/x_train.npy")

        return X_train

    def get_negative_descriptors_new_version(self, rectangles):
        if not LOAD_NEGATIVES:
            widths = []
            heigths = []
            X_train = []
            all_images = []
            for i in range(1, 1001):
                for character in characters:
                    all_images.append(f"antrenare/{character}/{i:04d}.jpg")

            for img in all_images:
                if img not in self.boxes_per_image:
                    continue
                image_boxes = self.boxes_per_image[img].copy()
                image = cv.imread(img)
                to_draw = image.copy()
                original_image = image.copy()

                windows = choose_non_intersecting_window_new(image.shape, rectangles, image_boxes.copy())

                rectangles_to_draw = [
                    ((int(x[0]), int(x[1]), int(x[2]), int(x[3])), (0, 0, 255)) for x in
                    self.boxes_per_image[img].copy()]

                for i in range(len(windows)):
                    window = windows[i]
                    patch = image[window[1]:window[3], window[0]:window[2], :]

                    widths.append(window[2] - window[0])
                    heigths.append(window[3] - window[1])
                    rectangles_to_draw.append((window, (0, 255, 0)))
                    patch = np.round(cv.resize(patch, self.window_sizes[0], interpolation=cv.INTER_AREA))
                    patch = np.array(patch, dtype=np.uint8)
                    filename = f"negative/{i}_" + img[10:].replace('/', '_')
                    cv.imwrite(filename, patch)

                    features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)

                    X_train.append(features)

                filename = "dreptunghiuri/" + img[10:].replace('/', '_')
                draw_rectangles_on_image_and_save(to_draw, rectangles_to_draw, filename)

            widths = np.array(widths)
            heights = np.array(heigths)

            plt.hist(heights, bins=30)
            plt.xlabel("Height negative")
            plt.show()

            plt.hist(widths, bins=30)
            plt.xlabel("Width negative")
            plt.show()

            X_train = np.array(X_train)
            shuffle = np.random.permutation(len(X_train))
            X_train = X_train[shuffle]
            # X_train = X_train[:13954]
            np.save("negative/x_train", X_train)
            return X_train

        else:
            X_train = np.load("negative/x_train.npy")

        return X_train

    def train(self):
        positives, characterssss, rectangles = self.get_train_positive_descriptors()
        # negatives = self.get_negative_descriptors(rectangles)
        negatives = self.get_negative_descriptors_new_version(rectangles)


        # shuffle = np.random.permutation(len(positives))
        # positives = positives[shuffle]
        #
        # shuffle = np.random.permutation(len(negatives))
        # negatives = negatives[shuffle]
        #
        #
        # # positives
        # negatives = negatives[:len(positives)]
        #
        # print(positives.shape, negatives.shape)
        #
        # X_train = np.concatenate((positives, negatives), axis=0)
        # y_train = np.ones(len(positives) + len(negatives))
        # y_train[len(positives):] = 0
        #
        # shuffle = np.random.permutation(len(X_train))
        # X_train = X_train[shuffle]
        # y_train = y_train[shuffle]
        # #
        # # model = MLPClassifier(random_state=1, max_iter=300, verbose=True, hidden_layer_sizes=(256,), batch_size=512)
        # # model.fit(X_train, y_train)
        # # acc = model.score(X_train, y_train)
        # # print(acc)
        # model = make_pipeline(StandardScaler(), LinearSVC(C=1, class_weight='balanced'))
        # model.fit(X_train, y_train)
        # acc = model.score(X_train, y_train)
        # print(acc)
        #
        # scores = model.decision_function(X_train)
        # positive_scores = scores[y_train > 0]
        # negative_scores = scores[y_train <= 0]
        #
        # plt.plot(np.sort(positive_scores))
        # plt.plot(np.zeros(len(positive_scores)))
        # plt.plot(np.sort(negative_scores))
        # plt.xlabel('Nr example antrenare')
        # plt.ylabel('Scor clasificator')
        # plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        # plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        # plt.show()
        #
        # joblib.dump(model, 'models/HogSVMClassifier.pkl')

    def test(self):
        svm_model = joblib.load('models/HogSVMClassifier.pkl')

        folder = TEST_PATH
        files = os.listdir(folder)

        detections = None
        scores = None
        filenames = np.array([])

        for file in tqdm(files):
            image_original = cv.imread(os.path.join(folder, file))

            image_detection = None
            image_scores = None
            downscale = 1.25
            upscale = 1
            for image in pyramid_gaussian(image_original, downscale=downscale, preserve_range=True, max_layer=self.pyramid_scales, channel_axis=2):
                scaled_image_scores = []
                scaled_image_detections = []
                window = self.window_sizes[0]
                for x in range(0, image.shape[0] - window[0], 1):
                    for y in range(0, image.shape[1] - window[1], 1):

                        patch = image[x:x + window[0], y:y + window[1], :].copy()
                        patch = np.array(patch, dtype=np.uint8)
                        features = get_features(patch, self.orientations, self.pixels_per_cell, self.cells_per_block)

                        score = svm_model.decision_function([features])[0]
                        # score = svm_model.predict_proba([features])[0][1]

                        # cv.imwrite(f"patches/{score}.jpg", patch)


                        if score > 0:
                            scaled_image_detections.append([x, y, x + window[0], y + window[1]])
                            scaled_image_scores.append(score)

                if len(scaled_image_scores) > 3:
                    scaled_image_detections, scaled_image_scores = non_maximum_suppression(
                        np.array(scaled_image_detections), np.array(scaled_image_scores))
                    scaled_image_detections = scaled_image_detections * upscale

                    if image_detection is None:
                        image_detection = scaled_image_detections
                        image_scores = scaled_image_scores
                    else:
                        image_detection = np.concatenate((image_detection.copy(), scaled_image_detections.copy()))
                        image_scores = np.concatenate((image_scores, scaled_image_scores))
                        image_detection, image_scores = non_maximum_suppression(image_detection.copy(), image_scores.copy())

                upscale *= 1.25

            if len(image_scores) > 0:
                image_detection, image_scores = non_maximum_suppression(image_detection.copy(), image_scores.copy())
                image_detection = np.round(image_detection)
                if DRAW:
                    to_draw = image_original.copy()
                    image_detection = np.array(image_detection, dtype=np.uint32)
                    rectangles = [((x[1], x[0], x[3], x[2]), (0, 255, 0)) for x in image_detection]
                    draw_rectangles_on_image_and_save(to_draw, rectangles, f"dreptunghiuri_validare/{file}")

                if detections is None:
                    detections = image_detection
                    scores = image_scores
                else:
                    detections = np.concatenate((detections, image_detection))
                    scores = np.concatenate((scores, image_scores))
                    image_names = [file for _ in range(len(scores))]
                    filenames = np.append(filenames, image_names)
        return detections, scores, filenames
