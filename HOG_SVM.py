import os
import cv2 as cv
from constants import *
from skimage.feature import hog
import numpy as np
import pickle
from utility import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn import metrics
from tqdm import tqdm


class HogSVMClassifier:
    def __init__(self):
        self.window_sizes = [(35, 45), (40, 40), (45, 35)]
        self.pyramid_scales = 2
        self.boxes_per_image = {}

    def get_train_positive_descriptors(self):
        if not LOAD_POZITIVES:
            X_train = []
            y_train = []
            for character in characters:
                annotation_file = f"antrenare/{character}_annotations.txt"
                image = None
                opened_image_name = None
                cnt = 0
                with open(annotation_file, 'r') as f:
                    for line in f:
                        image_name, xmin, ymin, xmax, ymax, ch = line.split()
                        xmin = int(xmin)
                        ymin = int(ymin)
                        xmax = int(xmax)
                        ymax = int(ymax)

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

                        patch = image[ymin:ymax + 1, xmin:xmax + 1, :]
                        patch = cv.resize(patch, self.window_sizes[1], interpolation=cv.INTER_AREA)

                        cv.imwrite(f"pozitive/{character}_{cnt}_{image_name}", patch)

                        patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                        features = hog(patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                       block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)

                        X_train.append(features)
                        y_train.append(character_mapping[ch])

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

        return X_train, y_train

    def get_negative_descriptors(self):
        if not LOAD_NEGATIVES:
            X_train = []
            all_images = []
            for i in range(1, 1001):
                for character in characters:
                    all_images.append(f"antrenare/{character}/{i:04d}.jpg")

            chosen_immages = np.random.choice(all_images, 6700, replace=True)

            for img in chosen_immages:
                image_boxes = self.boxes_per_image[img]
                image = cv.imread(img)
                to_draw = image.copy()

                scale = np.random.randint(0, self.pyramid_scales + 1)
                for i in range(scale):
                    image = cv.pyrDown(image)
                    image_boxes = [(x[0] // 2, x[1] // 2, x[2] // 2, x[3] // 2) for x in image_boxes.copy()]

                window_size_index = np.random.choice(len(self.window_sizes))
                window_size = self.window_sizes[window_size_index]

                windows = choose_non_intersecting_window(image.shape, window_size, image_boxes.copy())

                rectangles_to_draw = [
                    ((x[0] * 2 ** scale, x[1] * 2 ** scale, x[2] * 2 ** scale, x[3] * 2 ** scale), (0, 0, 255)) for x in
                    image_boxes.copy()]

                for i in range(len(windows)):
                    window = windows[i]
                    rectangles_to_draw.append((window, (0, 255, 0)))
                    patch = image[window[1]:window[3], window[0]:window[2], :]
                    patch = cv.resize(patch, self.window_sizes[1], interpolation=cv.INTER_AREA)
                    filename = f"negative/{i}_" + img[10:].replace('/', '_')
                    cv.imwrite(filename, patch)

                    patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                    features = hog(patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                   block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)

                    X_train.append(features)

                filename = "dreptunghiuri/" + img[10:].replace('/', '_')
                draw_rectangles_on_image_and_save(to_draw, rectangles_to_draw, filename)

            X_train = np.array(X_train)
            np.save("negative/x_train", X_train)
            return X_train

        else:
            X_train = np.load("negative/x_train.npy")

        return X_train

    def train(self):
        positives, characters = self.get_train_positive_descriptors()
        negatives = self.get_negative_descriptors()

        X_train = np.concatenate((positives, negatives), axis=0)
        y_train = np.ones(len(positives) + len(negatives))
        y_train[len(positives):] = -1

        shuffle = np.random.permutation(len(X_train))
        X_train = X_train[shuffle]
        y_train = y_train[shuffle]

        svm_model = SVC()
        svm_model.fit(X_train, y_train)

        scores = svm_model.decision_function(X_train)
        positive_scores = scores[y_train > 0]
        negative_scores = scores[y_train <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

        with open('models/HogSVMClassifier.pkl', 'wb') as f:
            pickle.dump(svm_model, f)

    def test(self):
        with open('models/HogSVMClassifier.pkl', 'rb') as f:
            svm_model = pickle.load(f)

        folder = TEST_PATH
        files = os.listdir(folder)

        detections = None
        scores = None
        filenames = np.array([])

        for file in tqdm(files):
            image = cv.imread(os.path.join(folder, file), cv.IMREAD_GRAYSCALE)

            image_detection = None
            image_scores = None

            for i in range(self.pyramid_scales):
                if i != 0:
                    image = cv.pyrDown(image)

                scaled_image_scores = []
                scaled_image_detections = []
                for window in self.window_sizes:
                    for x in range(0, image.shape[0] - window[0], 10):
                        for y in range(0, image.shape[1] - window[1], 10):
                            patch = image[x:x + window[0], y:y + window[1]].copy()
                            patch = cv.resize(patch, (40, 40))

                            features = hog(patch, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                           block_norm='L2-Hys', visualize=False, transform_sqrt=False,
                                           feature_vector=True)
                            score = svm_model.decision_function([features])[0]

                            if score > 0:
                                scaled_image_detections.append([x, y, x + window[0], y + window[0]])
                                scaled_image_scores.append(score)

                if len(scaled_image_scores) > 0:
                    scaled_image_detections, scaled_image_scores = non_maximum_suppression(
                        np.array(scaled_image_detections), np.array(scaled_image_scores))
                    scaled_image_detections = scaled_image_detections * 2 ** i

                    if image_detection is None:
                        image_detection = scaled_image_detections
                        image_scores = scaled_image_scores
                    else:
                        image_detection = np.concatenate((image_detection, scaled_image_detections))
                        image_scores = np.concatenate((image_scores, scaled_image_scores))

            if len(image_scores) > 0:
                image_detection, image_scores = non_maximum_suppression(image_detection, image_scores)

                if DRAW:
                    to_draw = image.copy()
                    rectangles = [(x, (0, 255, 0)) for x in image_detection]
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