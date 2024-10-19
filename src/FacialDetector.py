import glob
import os.path
import os
import pickle
import time
from copy import deepcopy

import cv2 as cv
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from skimage.transform import pyramid_gaussian
from sklearn.svm import LinearSVC
from skimage.exposure import exposure
from skimage.feature import hog
import src.Utilities as utils
import src.EvalUtils as eval_utils
from alive_progress import alive_bar

from src.TrainingClassifier import ClassifierTrainer


class FacialDetector:
    def __init__(self, params):
        self.params = params
        self.trainer = ClassifierTrainer(params=params)
        self.white_image = np.full((480, 360, 3), 255, dtype=np.uint8)

    def generate_random_valid_coords(self, image_coords):
        valid_coords = []
        tries = 0

        while len(valid_coords) < self.params.number_negative_examples_per_image:
            x_min = np.random.randint(0, 481 - self.params.window_size)  # generate a random integer in [0, 480]
            y_min = np.random.randint(0, 361 - self.params.window_size)  # generate a random integer in [0, 360]
            x_max = x_min + self.params.window_size
            y_max = y_min + self.params.window_size

            candidate_coords = (x_min, y_min, x_max, y_max)

            if not utils.does_any_rectangle_intersect(image_coords, candidate_coords):
                valid_coords.append(candidate_coords)
            else:
                tries += 1
            if tries > 50:
                tries += 1
                print(f"{tries} tries!")
                # utils.display_image_with_rois(self.white_image, image_coords, candidate_coords)
            if tries > 10 ** 3:
                return [(0, 1, self.params.window_size, self.params.window_size + 1)]
        return valid_coords

    def get_positive_descriptors(self, char_index):
        positive_descriptors = []
        start_time = time.perf_counter()  # Start timing image batch

        char_images = utils.read_image_data(self.params.dir_positive_examples[char_index], self.params.imgs_per_class)
        char_metadata = utils.read_image_metadata(self.params.base_dir, char_index, self.params.imgs_per_class)

        print(f"Computing positive descriptors for {self.params.dir_character_names[char_index]}!")

        for index, image_metadata in enumerate(char_metadata):
            images = char_images[image_metadata[0]]
            x_min, y_min, x_max, y_max = image_metadata[1]

            with alive_bar(len(images), title=f"Processing {index + 1}/{len(char_metadata)} images",
                           force_tty=True) as bar:
                for i, image in enumerate(images):
                    start_time_image = time.perf_counter()  # Start timing image
                    hog_features = self.process_image_operations(image, x_min, y_min, x_max, y_max)
                    positive_descriptors.extend(hog_features)
                    end_time_image = time.perf_counter()  # End timing image

                    bar()  # Update progress bar

                    print(f"Time taken for image {index + 1} part {i + 1}:"
                          f" {end_time_image - start_time_image:0.4f} seconds")

        end_time = time.perf_counter()  # End timing for image batch
        print(f"Time taken for {self.params.dir_character_names[char_index]}'s positive descriptors:"
              f" {end_time - start_time:0.4f} seconds")

        return np.array(positive_descriptors)

    def get_negative_descriptors(self, char_index):
        negative_descriptors = []
        start_time = time.perf_counter()  # Start timing

        char_images = utils.read_image_data(self.params.dir_positive_examples[char_index], self.params.imgs_per_class)
        char_metadata = utils.read_image_metadata(self.params.base_dir, char_index, self.params.imgs_per_class)
        char_image_coords_dict = utils.get_image_coords_dict(char_metadata)

        print(f"Computing negative descriptors for {self.params.dir_character_names[char_index]}!")

        for index, metadata in enumerate(char_metadata):

            print(f"Processing negative image {index} out of {len(char_metadata)}")
            images = char_images[metadata[0]]
            image_coords = self.generate_random_valid_coords(char_image_coords_dict[metadata[0]])
            start_time_image = time.perf_counter()  # Start timing

            with alive_bar(len(images), title=f"Processing {index + 1}/{len(char_metadata)} images",
                           force_tty=True) as bar:
                for image in images:
                    for (x_min, y_min, x_max, y_max) in image_coords:
                        hog_features = self.process_image_hog(image, x_min, y_min, x_max, y_max)
                        negative_descriptors.append(hog_features)
                    bar()

            end_time_image = time.perf_counter()  # End timing

            print(f"Time taken for image {index}: {end_time_image - start_time_image:0.4f} seconds")

        end_time = time.perf_counter()  # End timing
        print(f"Time taken for {self.params.dir_character_names[char_index]}'s negative descriptors:"
              f" {end_time - start_time:0.4f} seconds")

        return np.array(negative_descriptors)

    def process_image_operations(self, image, x_min, y_min, x_max, y_max):
        def apply_flip_lr(img):
            return np.fliplr(img)

        def apply_median_filter(img):
            return cv.medianBlur(img, 5)

        def apply_gaussian_blur(img):
            return cv.GaussianBlur(img, (5, 5), 0)

        operations = [lambda img: img,
                      apply_flip_lr,
                      # apply_gaussian_blur,
                      apply_median_filter]

        processed_images = []

        for operation in operations:
            processed_img = operation(image)
            hog_features = self.process_image_hog(processed_img, x_min, y_min, x_max, y_max)
            processed_images.append(hog_features)

        return processed_images

    def process_image_hog(self, image, x_min, y_min, x_max, y_max):
        """
        Applies hog transform to an image bounded by (x_min, x_max, y_min, y_max)
        :param image:
        :param x_min:
        :param y_min:
        :param x_max:
        :param y_max:
        :return:
        """
        image = image.astype(np.uint8)
        face = image[y_min: y_max + 1, x_min: x_max + 1]
        assert face.size > 0, "Attempting to resize an empty image!"
        face = cv.resize(face, (self.params.window_size, self.params.window_size))

        hog_features = hog(face,
                           orientations=8,
                           pixels_per_cell=(self.params.pixels_per_cell, self.params.pixels_per_cell),
                           cells_per_block=(self.params.hog_cell_size, self.params.hog_cell_size),
                           feature_vector=True)
        hog_features = hog_features.astype(np.float32)
        # print(hog_features.shape)
        # print(hog_features.flatten().shape)
        # print(image.shape, face.shape)
        return hog_features

    def process_single_image(self, file_name):
        """Process a single image and return its detections, scores, and file name."""
        start_time = time.perf_counter()
        original_img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
        all_detections, all_scores = self.process_image_pyramid(original_img)

        # Apply non-maximal suppression if detections were found
        if all_detections:
            image_detections, image_scores = eval_utils.non_maximal_suppression(
                np.array(all_detections), np.array(all_scores), original_img.shape)
        else:
            image_detections, image_scores = [], []

        end_time = time.perf_counter()
        print(f"Processed image in: {end_time - start_time} seconds")
        return image_detections, image_scores, os.path.basename(file_name)

    def process_image_pyramid(self, original_img):
        """Process an image through a Gaussian pyramid and return all detections and scores."""
        all_detections = []
        all_scores = []

        for (j, img) in enumerate(pyramid_gaussian(original_img, downscale=1.3)):
            if img.shape[0] < 30 or img.shape[1] < 30:
                break

            scale = original_img.shape[1] / float(img.shape[1])
            image_scores, image_detections = self.process_image(img)

            rescaled_detections = [[int(coord * scale) for coord in detection] for detection in image_detections]
            all_detections.extend(rescaled_detections)
            all_scores.extend(image_scores)

        return all_detections, all_scores

    def run(self):
        test_files = glob.glob(os.path.join(self.params.dir_test_examples, '*.jpg'))
        final_detections = []  # Store final detections
        final_scores = []  # Store final scores
        final_file_names = []  # Store final filenames

        with alive_bar(len(test_files), title="Processing images", force_tty=True) as bar:
            for index, file_name in enumerate(test_files):
                print(f'Processing image {index + 1}/{len(test_files)}')

                image_detections, image_scores, file_basename = self.process_single_image(file_name)

                final_detections.extend(image_detections)
                final_scores.extend(image_scores)
                final_file_names.extend([file_basename] * len(image_scores))

                bar()

        return final_detections, final_scores, final_file_names

    def process_image(self, img):
        image_scores = []
        image_detections = []
        img = img.astype(np.uint8)
        # print(img.shape)
        hog_descriptors = hog(img,
                              orientations=8,
                              pixels_per_cell=(self.params.pixels_per_cell, self.params.pixels_per_cell),
                              cells_per_block=(self.params.hog_cell_size, self.params.hog_cell_size),
                              visualize=False,
                              feature_vector=False)

        # print(hog_descriptors.shape)
        # print(hog_descriptors.flatten().shape)
        hog_descriptors = hog_descriptors.astype(np.float32)

        num_cols = img.shape[1] // self.params.pixels_per_cell - 1
        num_rows = img.shape[0] // self.params.pixels_per_cell - 1
        num_cell_in_template = self.params.window_size // self.params.pixels_per_cell - 1

        for y in range(0, num_rows - num_cell_in_template):
            for x in range(0, num_cols - num_cell_in_template):
                descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()

                # print(descr.shape)
                # print(self.trainer.best_model.coef_.T.shape)

                score = np.dot(descr, self.trainer.best_model.coef_.T)[0] + self.trainer.best_model.intercept_[0]

                if score > self.params.threshold:
                    x_min = int(x * self.params.pixels_per_cell)
                    y_min = int(y * self.params.pixels_per_cell)
                    x_max = int(x * self.params.pixels_per_cell + self.params.window_size)
                    y_max = int(y * self.params.pixels_per_cell + self.params.window_size)
                    image_detections.append([x_min, y_min, x_max, y_max])
                    image_scores.append(score)

        return image_scores, image_detections
