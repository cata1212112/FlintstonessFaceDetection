import numpy as np
from Data import *
from Classifier import *
from utility import *

# Used to generate positive examples and negative examples
# a = Generator()
# a.generate_positive_examples()
# a.generate_negative_examples()

classifier = SVM()
# Train the classifier
# classifier.train()
# classifier.train_character_classifier()


detections = np.array([])
scores = np.array([])
file_names = np.array([])

for image, filename in get_validation_images():
    current_detections, current_scores, current_file_names = classifier.predict(image.copy(), filename)

    draw_rectangles_on_image_and_save(image, [(w, (0, 255, 0)) for w in current_detections],
                                      f"dreptunghiuri_validare/{filename}", current_scores)

    detections = np.append(detections, current_detections)
    scores = np.append(scores, current_scores)
    file_names = np.append(file_names, current_file_names)


np.save("detections.npy", detections)
np.save("scores.npy", scores)
np.save("file_names.npy", file_names)

detections = np.load("detections.npy")
detections = detections.reshape((detections.shape[0] // 4, 4))
scores = np.load("scores.npy")
file_names = np.load("file_names.npy")
eval_detections(detections, scores, file_names)

barney_detections = []
barney_detections_file_names = []
betty_detections = []
betty_detections_file_names = []
fred_detections = []
fred_detections_file_names = []
wilma_detections = []
wilma_detections_file_names = []
cnt = 1
for detection, file_name in zip(detections, file_names):
    image = Image.open(os.path.join("validare/validare", file_name))
    image = image.convert("RGB")
    patch = image.crop(detection)
    patch_copy = patch.copy()

    patch = np.array(patch)
    # patch = rgb2gray(np.array(patch))
    character = classifier.test_character_classifier(patch)

    patch_copy.save(f"recunoastere_caractere/{cnt}_{character}.jpg")
    cnt += 1

    if character == "barney":
        barney_detections.append(detection)
        barney_detections_file_names.append(file_name)
    elif character == "betty":
        betty_detections.append(detection)
        betty_detections_file_names.append(file_name)
    elif character == "fred":
        fred_detections.append(detection)
        fred_detections_file_names.append(file_name)
    elif character == "wilma":
        wilma_detections.append(detection)
        wilma_detections_file_names.append(file_name)

eval_detections(np.array(barney_detections), np.array([1] * len(barney_detections)), np.array(barney_detections_file_names), "validare/task2_barney_gt_validare.txt", "barney")
eval_detections(np.array(betty_detections), np.array([1] * len(betty_detections)), np.array(betty_detections_file_names), "validare/task2_betty_gt_validare.txt", "betty")
eval_detections(np.array(fred_detections), np.array([1] * len(fred_detections)), np.array(fred_detections_file_names), "validare/task2_fred_gt_validare.txt", "fred")
eval_detections(np.array(wilma_detections), np.array([1] * len(wilma_detections)), np.array(wilma_detections_file_names), "validare/task2_wilma_gt_validare.txt", "wilma")

# used to get negative examples from hard mining
# hard_mining(classifier)