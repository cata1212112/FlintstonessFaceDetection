import numpy as np
import os


class SaveSolution:
    def __init__(self, folder, character, detections, scores, file_names):
        self.folder = folder
        self.character = character
        self.detections = detections
        self.scores = scores
        self.file_names = file_names

    def save(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        np.save(os.path.join(self.folder, f"detections_{self.character}.npy"), self.detections, allow_pickle=True,
                fix_imports=True)
        np.save(os.path.join(self.folder, f"scores_{self.character}.npy"), self.scores, allow_pickle=True,
                fix_imports=True)
        np.save(os.path.join(self.folder, f"file_names_{self.character}.npy"), self.file_names, allow_pickle=True,
                fix_imports=True)
