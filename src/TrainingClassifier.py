import os
import pickle
from copy import deepcopy

import cv2 as cv
import numpy as np
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
import Utilities as utils


class ClassifierTrainer:
    def __init__(self, params):
        self.params = params
        self.best_model = None

    def load_or_train_classifier(self, train_examples, train_labels):
        print(train_examples.shape, train_labels.shape)
        svm_file_name = self._get_model_filename()
        if os.path.exists(svm_file_name):
            self.best_model = self._load_model(svm_file_name)
        else:
            validation_images, validation_metadata, validation_labels = utils.get_validation_data_task_1(
                "task1_gt_validare.txt")
            validation_examples = self.get_validation_examples(validation_images, validation_metadata)
            self.best_model = self._train_and_select_best_model(train_examples, train_labels,
                                                                validation_examples, validation_labels)
            self._save_model(svm_file_name, self.best_model)
            self._visualize_model_performance(train_examples, train_labels)

    def _get_model_filename(self):
        return os.path.join(self.params.dir_save_files, 'best_model_{}_{}_{}'.format(
            self.params.hog_cell_size, self.params.number_negative_examples, self.params.number_positive_examples))

    @staticmethod
    def _load_model(filename):
        return pickle.load(open(filename, 'rb'))

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
        face = cv.resize(face, (self.params.window_size, self.params.window_size))

        hog_features = hog(face,
                           orientations=8,
                           pixels_per_cell=(self.params.pixels_per_cell, self.params.pixels_per_cell),
                           cells_per_block=(self.params.hog_cell_size, self.params.hog_cell_size),
                           visualize=False,
                           feature_vector=True)
        return hog_features

    def get_validation_examples(self, validation_images, validation_metadata):
        validation_examples_hog = []

        for idx, (x_min, y_min, x_max, y_max) in validation_metadata:
            image = validation_images[idx]
            hog_image = self.process_image_hog(image, x_min, y_min, x_max, y_max)
            validation_examples_hog.append(hog_image)

        return validation_examples_hog

    @staticmethod
    def _train_and_select_best_model(training_examples, train_labels, validation_examples, validation_labels):
        print(len(training_examples), len(train_labels))
        best_accuracy = 0
        best_model = None
        c_values = [10 ** -5
                    # , 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0
                    ]

        with alive_bar(len(c_values), title='Training Models') as bar:
            for c in c_values:
                print(f"Training model with C={c}...")
                model = LinearSVC(C=c)
                model.fit(training_examples, train_labels)
                acc = model.score(validation_examples, validation_labels)
                print(f"Model with C={c} achieved accuracy: {acc * 100:.2f}%")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = deepcopy(model)

                bar()  # Indicate progress

        print(f"Best model achieved an accuracy of: {best_accuracy * 100:.2f}%")
        return best_model

    @staticmethod
    def _save_model(filename, model):
        pickle.dump(model, open(filename, 'wb'))

    def _visualize_model_performance(self, training_examples, train_labels):
        scores = self.best_model.decision_function(training_examples)
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
