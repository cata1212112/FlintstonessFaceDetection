from Data import Generator
from constants import *
from tqdm import tqdm
from Classifier import *
from utility import *
from Save import SaveSolution
from fasterrcnn import MyFasterRCNN

# Used to generate positive examples and negative examples
if GENERATE_DATA:
    a = Generator()
    a.generate_positive_examples()
    a.generate_negative_examples()

classifier = SVM()

# used to get negative examples from hard mining
if HARD_MINING:
    Generator.hard_mining(classifier)

# Train the classifier
if TRINING:
    classifier.train()
    classifier.train_character_classifier()

detections = []
scores = np.array([])
file_names = np.array([])

for image, filename in tqdm(Generator.get_test_images(TEST_PATH), desc="Detecting faces"):
    current_detections, current_scores, current_file_names = classifier.predict(image.copy(), filename)

    if DRAW:
        draw_rectangles_on_image_and_save(image, [(w, (0, 255, 0)) for w in current_detections],
                                          f"dreptunghiuri_validare/{filename}")

    detections.extend(current_detections)
    scores = np.append(scores, current_scores)
    file_names = np.append(file_names, current_file_names)

detections = np.array(detections)

character_detections = {'barney': [], 'betty': [], 'fred': [], 'wilma': []}
character_detections_file_names = {'barney': [], 'betty': [], 'fred': [], 'wilma': []}

cnt = 1
for detection, file_name in tqdm(zip(detections, file_names), desc="Detecting characters"):
    image = Image.open(os.path.join("validare/validare", file_name))
    image = image.convert("RGB")
    patch = image.crop(detection)
    patch_copy = patch.copy()

    patch = np.array(patch)
    character = classifier.test_character_classifier(patch)

    if DRAW:
        patch_copy.save(f"recunoastere_caractere/{cnt}_{character}.jpg")
        cnt += 1

    character_detections[character].append(detection)
    character_detections_file_names[character].append(file_name)

SaveSolution("fisiere_solutie/task1", "all_faces", detections, scores, file_names).save()

for character in characters:
    if character == "unknown":
        continue
    SaveSolution("fisiere_solutie/task1", character, character_detections[character],
                 np.array([1] * len(character_detections[character])),
                 character_detections_file_names[character]).save()

fasterRCNN = MyFasterRCNN()

if FASTERRCNN_TRAIN:
    fasterRCNN.train()

fasterRCNN.test()
