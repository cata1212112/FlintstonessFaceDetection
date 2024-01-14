from Descriptor import *
from PIL import Image
import os
import joblib
from sklearn.svm import SVC
from skimage.transform import pyramid_gaussian
from utility import non_maximum_suppression
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray
from constants import *
from Data import Generator


class SVM:
    POSITIVES = "pozitive"
    NEGATIVES = "negative"

    def __init__(self):
        self.X = []
        self.y = []
        self.patch_size = 40
        self.descriptor = Descriptor(self.patch_size, 9, (10, 10), (4, 4), "L2-Hys", False, 8)
        self.character_patch_size = 40

    def get_patch_descriptors(self, image_folder, image_name):
        image = Image.open(os.path.join(image_folder, image_name))
        image = image.convert("RGB")

        width, height = image.size

        if width < self.patch_size or height < self.patch_size:
            return None

        image = rgb2gray(np.array(image))
        return self.descriptor.get_hog_features(image)

    def load_data(self, path):
        for image_name in os.listdir(path):
            descriptor = self.get_patch_descriptors(path, image_name)
            if descriptor is not None:
                self.X.append(descriptor)

                if path == self.POSITIVES:
                    self.y.append(1)
                else:
                    self.y.append(-1)

    def load_hard_negatives(self, path):
        for image_name in os.listdir(path):
            descriptor = self.get_patch_descriptors(path, image_name)
            if descriptor is not None:
                self.X.append(descriptor)

                self.y.append(-1)

    def train(self):
        self.load_data(self.POSITIVES)
        self.load_data(self.NEGATIVES)
        self.load_hard_negatives("hard_mining")

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        shuffle = np.random.permutation(len(self.X))
        self.X = self.X[shuffle]
        self.y = self.y[shuffle]

        model_hog = Pipeline([('scaler', StandardScaler()), ('svc', SVC(C=100, kernel='rbf', gamma='scale'))])

        model_hog.fit(self.X, self.y)
        joblib.dump(model_hog, 'models/hog.pkl')

        scores = model_hog.decision_function(self.X)
        positive_scores = scores[self.y > 0]
        negative_scores = scores[self.y <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def predict(self, original_image, image_name):
        model_hog = joblib.load('models/hog.pkl')
        detections = []
        scores = []
        file_names = np.array([])

        upscale = 1

        original_image = rgb2gray(np.array(original_image))

        for image in pyramid_gaussian(np.array(original_image), downscale=1.25, preserve_range=True):
            if image.shape[0] < self.patch_size or image.shape[1] < self.patch_size:
                break
            hog_rider = self.descriptor.get_hog_features(image, feature_vector=False, toResize=False)

            num_cols = image.shape[1] // self.descriptor.pixels_per_cell[0] - self.descriptor.cells_per_block[0] + 1
            num_rows = image.shape[0] // self.descriptor.pixels_per_cell[0] - self.descriptor.cells_per_block[0] + 1

            num_cell_in_template = self.patch_size // self.descriptor.pixels_per_cell[0] - self.descriptor.cells_per_block[0] + 1

            scaled_detections = []
            scaled_scores = []

            for y in range(0, num_rows - num_cell_in_template):
                for x in range(0, num_cols - num_cell_in_template):
                    descr = hog_rider[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                    prediction_hog = model_hog.decision_function([descr])[0]
                    if prediction_hog > 0.5:
                        x_min = int(x * self.descriptor.pixels_per_cell[0])
                        y_min = int(y * self.descriptor.pixels_per_cell[0])
                        x_max = int(x * self.descriptor.pixels_per_cell[0] + self.patch_size)
                        y_max = int(y * self.descriptor.pixels_per_cell[0] + self.patch_size)

                        scaled_detections.append(
                            [int(x_min * upscale), int(y_min * upscale), int(x_max * upscale), int(y_max * upscale)])
                        scaled_scores.append(prediction_hog)

            if len(scaled_detections) > 0:
                scaled_detections = np.array(scaled_detections)
                scaled_scores = np.array(scaled_scores)
                scaled_detections, scaled_scores = non_maximum_suppression(scaled_detections, scaled_scores)
                detections.extend(scaled_detections)
                scores.extend(scaled_scores)
            upscale *= 1.25

        detections = np.array(detections)
        scores = np.array(scores)
        detections, scores = non_maximum_suppression(detections, scores)
        image_names = [image_name for _ in range(len(scores))]
        file_names = np.append(file_names, image_names)
        return detections, scores, file_names

    def train_character_classifier(self):
        model_per_character = Pipeline([('scaler', StandardScaler()),
                                        ('svc', SVC(C=1, kernel='rbf', gamma='scale', decision_function_shape='ovr'))])

        X = []
        y = []
        for patch, ch in Generator.get_character_faces():
            if ch == 'unknown':
                continue

            flipped_patch = patch.transpose(Image.FLIP_LEFT_RIGHT)

            patch = np.array(patch)
            flipped_patch = np.array(flipped_patch)

            patch = resize(patch, (self.character_patch_size, self.character_patch_size, 3), anti_aliasing=True)
            flipped_patch = resize(flipped_patch, (self.character_patch_size, self.character_patch_size, 3),
                                   anti_aliasing=True)
            X.append(patch.flatten())
            y.append(character_mapping[ch])

            X.append(flipped_patch.flatten())
            y.append(character_mapping[ch])

        unknowns = []
        for image_name in os.listdir("hard_mining"):
            image = Image.open(os.path.join("hard_mining", image_name))
            image = image.convert("RGB")

            width, height = image.size

            if width < self.patch_size or height < self.patch_size:
                continue

            image = np.array(image)
            image = resize(image, (self.character_patch_size, self.character_patch_size, 3), anti_aliasing=True)
            unknowns.append(image.flatten())

        shuffle = np.random.permutation(len(unknowns))
        unknowns = np.array(unknowns)[shuffle]
        X.extend(unknowns)
        y.extend([4] * len(unknowns))

        X = np.array(X)
        y = np.array(y)

        shuffle = np.random.permutation(len(X))
        X = X[shuffle]
        y = y[shuffle]

        model_per_character.fit(X, y)

        joblib.dump(model_per_character, 'models/character.pkl')

    def test_character_classifier(self, patch):
        model_character = joblib.load('models/character.pkl')
        patch = resize(patch, (self.character_patch_size, self.character_patch_size), anti_aliasing=True)
        prediction = model_character.predict([patch.flatten()])
        return characters[prediction[0]]
