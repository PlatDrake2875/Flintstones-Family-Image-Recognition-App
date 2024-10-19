import glob
import os.path
import os
import pickle
import time
from copy import deepcopy

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import pyramid_gaussian
from sklearn.svm import LinearSVC
from skimage.exposure import exposure
from skimage.feature import hog


def display_hog_image(original_image, hog_image):
    """
    Display the original image and its Histogram of Oriented Gradients (HOG) representation.

    Parameters:
    - original_image: The original image.
    - hog_image: The HOG representation of the original image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # Display the original image
    ax1.axis('off')
    ax1.imshow(original_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display of the HOG image
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Display the HOG image
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.show()


class FacialDetectorChar:
    def __init__(self, params, index):
        self.images_dict_metadata = {}
        self.image_metadata = []
        self.best_model = None
        self.params = params
        self.images_dict = {}
        self.positive_image_meta_data = []
        self.negative_image_meta_data = []
        self.char_index = index

    def read_image_data(self):
        """
        Reads images of a character's dataset
        :param index:
        :return: a list of tuples (image_file_name, image)
        """
        images_path = os.path.join(self.params.dir_positive_examples[self.char_index], "*.jpg")
        file_names = glob.glob(images_path)

        self.images_dict = {}
        for image_file in file_names:
            key = f"{self.params.dir_character_names[self.char_index]}_{image_file[-8:][:4]}"
            image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

            if key not in self.images_dict:
                self.images_dict[key] = [image]
            else:
                self.images_dict[key].append(image)

    def read_all_image_data(self):
        """
        Reads images of a character's dataset
        :param index:
        :return: a list of tuples (image_file_name, image)
        """
        images_paths = [os.path.join(self.params.dir_positive_examples[index], "*.jpg") for index in range(4)]
        char_file_names = [glob.glob(char_path) for char_path in images_paths]
        self.images_dict = {}
        for char_index, file_names in enumerate(char_file_names):
            for image_file in file_names:
                key = f"{self.params.dir_character_names[char_index]}_{image_file[-8:][:4]}"
                image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)
                if key not in self.images_dict:
                    self.images_dict[key] = [image]
                else:
                    self.images_dict[key].append(image)

    def read_positive_image_metadata(self):
        """
        Reads metadata of images that contain character at pos index
        :return:
        """
        char_files = [glob.glob(f"{self.params.base_dir}/*.txt")[char_index] for char_index in range(4)]
        self.positive_image_meta_data = []

        for char_file in char_files:
            with open(char_file, "r") as f:
                line = f.readline().split()

                while len(line):
                    # print(line)

                    label = line[5]
                    if label == self.params.dir_character_names[self.char_index]:
                        image_index = f"{self.params.dir_character_names[self.char_index]}_{line[0][:4]}"
                        coords = list(map(int, line[1:5]))
                        self.positive_image_meta_data.append((image_index, coords))

                    line = f.readline().split()

    def read_negative_image_metadata(self):
        """
        Reads metadata of images that don't cotanin the character at index
        :param index:
        :return:
        """
        char_files = [glob.glob(f"{self.params.base_dir}/*.txt")[char_index] for char_index in range(4)]
        self.negative_image_meta_data = []

        for char_file in char_files:
            with open(char_file, "r") as f:
                line = f.readline().split()

                while len(line):
                    # print(line)

                    label = line[5]
                    image_index = f"{label}_{line[0][:4]}"
                    coords = tuple(map(int, line[1:5]))
                    self.negative_image_meta_data.append((image_index, coords, label))

                    line = f.readline().split()

    def get_images_dict_metadata(self):
        self.images_dict_metadata = {}
        for image_cnt, coords, char_label in self.negative_image_meta_data:
            if image_cnt not in self.images_dict_metadata:
                self.images_dict_metadata[image_cnt] = [(coords, char_label)]
            else:
                self.images_dict_metadata[image_cnt].append((coords, char_label))

    def get_images_dict_positive_metadata(self):
        self.images_dict_metadata = {}
        for image_cnt, coords in self.positive_image_meta_data:
            if image_cnt not in self.images_dict_metadata:
                self.images_dict_metadata[image_cnt] = [coords]
            else:
                self.images_dict_metadata[image_cnt].append(coords)

    def check_char_in_image(self, image_metadata):
        for roi, label in image_metadata:
            if label == self.params.dir_character_names[self.char_index]:
                return roi
        return 0, 1, 0, 1

    def generate_random_valid_coords(self, image_coords) -> list[tuple]:
        valid_coords = []

        while len(valid_coords) < self.params.number_negative_examples_per_image:
            x_min = np.random.randint(0, 481 - self.params.window_size)  # generate a random integer in [0, 480]
            y_min = np.random.randint(0, 361 - self.params.window_size)  # generate a random integer in [0, 360]
            x_max = x_min + self.params.window_size
            y_max = y_min + self.params.window_size
            candidate_coords = (x_min, y_min, x_max, y_max)

            if self.check_valid_coords(image_coords, candidate_coords):
                valid_coords.append(candidate_coords)

        return valid_coords

    def check_valid_coords(self, image_coords, candidate_coords):
        return self.intersection_over_union(image_coords, candidate_coords) <= 0.025

    @staticmethod
    def display_image_with_roi(image, x_min, y_min, x_max, y_max):
        """
        Display an image with a green rectangle marking the Region of Interest (ROI).

        Parameters:
        - image: The original image (as a NumPy array).
        - x_min, y_min: The coordinates of the top-left corner of the ROI.
        - x_max, y_max: The coordinates of the bottom-right corner of the ROI.
        """
        if image is None:
            print("Error: Image is None")
            return
        if not isinstance(image, np.ndarray):
            print(f"Error: Image is not a numpy array. Type: {type(image)}")
            return

        # Copy the image to avoid modifying the original
        image_with_roi = image.copy()

        # Draw a green rectangle on the image
        cv.rectangle(image_with_roi, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the image with ROI
        plt.imshow(cv.cvtColor(image_with_roi, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        plt.axis('off')
        plt.title('Image with ROI')
        plt.show()

    def process_positive_image_for_hog_features(self, image, x_min, y_min, x_max, y_max):
        image = image.astype(np.uint8)
        # self.display_image_with_roi(image, x_min, y_min, x_max, y_max)

        face = image[y_min: y_max + 1, x_min: x_max + 1]
        face_resized = cv.resize(face, (self.params.window_size, self.params.window_size))

        hog_features = hog(face_resized,
                           orientations=9,
                           pixels_per_cell=(self.params.pixels_per_cell, self.params.pixels_per_cell),
                           cells_per_block=(self.params.hog_cell_size, self.params.hog_cell_size),
                           visualize=False,
                           feature_vector=True)
        return hog_features

    def process_image_operations(self, image, x_min, y_min, x_max, y_max):
        def apply_flip_lr(img):
            return img.fliplr()

        def apply_median_filter(img):
            return cv.medianBlur(img, 5)

        def apply_gaussian_blur(img):
            return cv.GaussianBlur(img, (5, 5), 0)

        operations = [lambda img: img, apply_flip_lr, apply_median_filter, apply_gaussian_blur]

        processed_images = []

        for operation in operations:
            processed_img = operation(image)
            hog_features = self.process_positive_image_for_hog_features(processed_img, x_min, y_min, x_max, y_max)
            processed_images.append(hog_features)

        return processed_images

    def get_positive_descriptors(self):
        positive_descriptors = []
        start_time = time.perf_counter()

        self.read_all_image_data()
        self.read_positive_image_metadata()
        self.get_images_dict_positive_metadata()

        print(f"Computing positive descriptors for {self.params.dir_character_names[self.char_index]}!")

        for index, (image_key, roi) in enumerate(self.positive_image_meta_data):
            print(f"Processing positive image {index + 1} of {len(self.positive_image_meta_data)}")

            start_time_image = time.perf_counter()
            images = self.images_dict[image_key]
            x_min, y_min, x_max, y_max = roi

            for image in images:
                if image is not None:
                    hog_features = self.process_image_operations(image, x_min, y_min, x_max, y_max)
                    positive_descriptors.extend(hog_features)

            print(f"Time taken for image {index + 1}: {time.perf_counter() - start_time_image:.2f} seconds")

        print(
            f"Total time for {self.params.dir_character_names[self.char_index]} descriptors: "
            f"{time.perf_counter() - start_time:.2f} seconds")
        return np.array(positive_descriptors)

    def process_image_for_negative_hog_features(self, image, coords):
        """
        Processes a single image to extract HOG features for the negative examples.
        """
        x_min, y_min, x_max, y_max = coords
        # self.display_image_with_roi(image, x_min, y_min, x_max, y_max)
        face = image[y_min: y_max + 1, x_min: x_max + 1]
        face_resized = cv.resize(face, (self.params.window_size, self.params.window_size))

        hog_features = hog(face_resized,
                           orientations=9,
                           pixels_per_cell=(self.params.pixels_per_cell, self.params.pixels_per_cell),
                           cells_per_block=(self.params.hog_cell_size, self.params.hog_cell_size),
                           visualize=False,
                           feature_vector=True)
        return hog_features

    def get_negative_descriptors(self):
        negative_descriptors = []
        start_time = time.perf_counter()
        print(f"Computing negative descriptors for {self.params.dir_character_names[self.char_index]}!")

        self.read_all_image_data()
        self.read_negative_image_metadata()
        self.get_images_dict_metadata()

        for index, (key, images) in enumerate(self.images_dict.items()):
            for image in images:
                print(f"Processing negative image {index + 1} of {len(self.images_dict)}")
                if key not in self.images_dict_metadata:
                    print(f"Key {key} not found!!")
                    continue
                start_time_image = time.perf_counter()
                valid_coords = self.check_char_in_image(self.images_dict_metadata[key])
                image_coords = self.generate_random_valid_coords(valid_coords)

                for img_index, coords in enumerate(image_coords):
                    hog_features = self.process_image_for_negative_hog_features(image, coords)
                    negative_descriptors.append(hog_features)

                print(f"Time taken for image {index + 1}: {time.perf_counter() - start_time_image:.2f} seconds")

        print(f"Total time for {self.params.dir_character_names[self.char_index]} negative descriptors:"
              f" {time.perf_counter() - start_time:.2f} seconds")
        return np.array(negative_descriptors)

    @staticmethod
    def intersection_over_union(bbox_a, bbox_b):
        print(bbox_a, bbox_b)
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def get_validation_data_task_2(self):

        # Reading images
        images_path = glob.glob(os.path.join("..", "validare", "validare", "*.jpg"))
        validation_examples_images = {"{:04d}".format(img_index): cv.imread(image_file, cv.IMREAD_GRAYSCALE) for
                                      img_index, image_file in
                                      enumerate(images_path)}

        # Reading validation file
        validation_examples_gt = glob.glob(os.path.join("..", "validare", "*.txt"))
        validation_example_file = None
        for vegt in validation_examples_gt:
            if self.params.dir_character_names[self.char_index] in vegt:
                validation_example_file = vegt
                break

        # Creating descriptors
        validation_examples = []
        with open(validation_example_file, "r") as f:
            line = f.readline().split()

            while len(line):
                img_cnt = line[0][:4]
                x_min, y_min, x_max, y_max = tuple(map(int, line[1:5]))
                hog_features = self.process_positive_image_for_hog_features(validation_examples_images[img_cnt], x_min,
                                                                            y_min, x_max, y_max)
                validation_examples.append(hog_features)
                line = f.readline().split()

        validation_labels = [1] * len(validation_examples)

        return validation_examples, validation_labels

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.hog_cell_size, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            print("Loaded SVM classified!")
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        print("Computing SVM classifier!")
        validation_examples, validation_labels = self.get_validation_data_task_2()
        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(validation_examples, validation_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[
                        j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        test_files = glob.glob(os.path.join(self.params.dir_test_examples, 'validare', '*.jpg'))
        final_detections = []  # Store final detections
        final_scores = []  # Store final scores
        final_file_names = []  # Store final filenames

        for index, file_name in enumerate(test_files):
            print(f'Processing image {index}/{len(test_files)}')
            start_time = time.perf_counter()
            original_img = cv.imread(file_name, cv.IMREAD_GRAYSCALE)

            all_detections = []  # Store all detections for current image
            all_scores = []  # Store all scores for current image

            for (j, img) in enumerate(pyramid_gaussian(original_img, downscale=1.3)):
                if img.shape[0] < 30 or img.shape[1] < 30:
                    break

                scale = original_img.shape[1] / float(img.shape[1])
                image_scores, image_detections = self.process_image(img)

                rescaled_detections = [[int(coord * scale) for coord in detection] for detection in
                                       image_detections]
                all_detections.extend(rescaled_detections)
                all_scores.extend(image_scores)

            if len(all_detections) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(all_detections),
                                                                              np.array(all_scores),
                                                                              original_img.shape)
                final_detections.extend(image_detections)
                final_scores.extend(image_scores)
                final_file_names.extend([os.path.basename(file_name)] * len(image_scores))
            end_time = time.perf_counter()  # End timing
            print(f"Processed image in: {end_time - start_time} seconds")

        return final_detections, final_scores, final_file_names

    def process_image(self, img):
        image_scores = []
        image_detections = []
        hog_descriptors = hog(img,
                              pixels_per_cell=(self.params.hog_cell_size, self.params.hog_cell_size),
                              cells_per_block=(self.params.pixels_per_cell, self.params.pixels_per_cell),
                              feature_vector=False)
        num_cols = img.shape[1] // self.params.hog_cell_size - 1
        num_rows = img.shape[0] // self.params.hog_cell_size - 1
        num_cell_in_template = self.params.window_size // self.params.hog_cell_size - 1

        for y in range(0, num_rows - num_cell_in_template):
            for x in range(0, num_cols - num_cell_in_template):
                descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                score = np.dot(descr, self.best_model.coef_.T)[0] + self.best_model.intercept_[0]
                if score > self.params.threshold:
                    x_min = int(x * self.params.hog_cell_size)
                    y_min = int(y * self.params.hog_cell_size)
                    x_max = int(x * self.params.hog_cell_size + self.params.window_size)
                    y_max = int(y * self.params.hog_cell_size + self.params.window_size)
                    image_detections.append([x_min, y_min, x_max, y_max])
                    image_scores.append(score)

        return image_scores, image_detections

    @staticmethod
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

    def eval_detections(self, detections, scores, file_names, char_index):
        ground_truth_file = np.loadtxt(self.params.dir_paths_annotations[char_index], dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int32)
        num_gt_detections = len(ground_truth_detections)
        gt_exists_detection = np.zeros(num_gt_detections)

        detections = np.array(detections)
        scores = np.array(scores)
        file_names = np.array(file_names)

        sorted_indices = np.argsort(scores)[::-1]
        detections = detections[sorted_indices]
        scores = scores[sorted_indices]
        file_names = file_names[sorted_indices]

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
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # Classify detection as true positive or false positive
            if max_overlap >= 0.05:  # Threshold for considering as a match
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

        average_precision = self.compute_average_precision(rec, prec)

        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(
            f'{self.params.dir_character_names[self.char_index]} faces: average precision: {average_precision:.3f}')
        plt.savefig(os.path.join(self.params.dir_save_files,
                                 f'Task2_{self.params.dir_character_names[self.char_index]}.png'))
        plt.show()
