from skimage.feature import hog
from skimage.transform import resize
import numpy as np


class Descriptor:
    def __init__(self, patch_size, orientations, pixels_per_cell, cells_per_block, block_norm, transform_sqrt, color_size):
        self.patch_size = patch_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
        self.color_size = color_size

    def get_hog_features(self, patch, feature_vector=True, toResize=True):
        patch = np.array(patch)
        if toResize:
            patch = resize(patch, (self.patch_size, self.patch_size), anti_aliasing=True)
        hog_features = hog(patch, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                           cells_per_block=self.cells_per_block, visualize=False, transform_sqrt=self.transform_sqrt,
                           feature_vector=feature_vector, block_norm=self.block_norm)

        return hog_features

    def get_color_features(self, patch, color_bins, bins_range, size):
        patch = np.array(patch)
        patch = resize(patch, (size, size), anti_aliasing=True)
        channel1 = np.histogram(patch[:, :, 0], bins=color_bins, range=bins_range)
        channel2 = np.histogram(patch[:, :, 1], bins=color_bins, range=bins_range)
        channel3 = np.histogram(patch[:, :, 2], bins=color_bins, range=bins_range)
        patch_features = np.concatenate((channel1[0], channel2[0], channel3[0]))
        return patch_features

    def get_features(self, patch, color_bins, bins_range):
        # patch = np.array(patch)
        # patch = resize(patch, (self.patch_size, self.patch_size), anti_aliasing=True)
        # # patch_features = self.get_color_features(patch, color_bins, bins_range, self.color_size)
        # patch_features = np.concatenate((patch_features, self.get_hog_features(patch)))
        patch_features = self.get_hog_features(patch)
        return patch_features
