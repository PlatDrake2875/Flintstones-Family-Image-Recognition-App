import os.path
import time

import numpy as np
from Parameters import Parameters
from FacialDetector import FacialDetector
from src.Vizualize import show_detections_with_ground_truth, show_detections_without_ground_truth
from src.EvalUtils import eval_detections


class Task1:

    def __init__(self):
        self.params: Parameters = Parameters()
        self.facial_detector = FacialDetector(self.params)

    def get_positive_descriptors(self):
        positive_descriptors = []

        for i in range(4):
            start_time = time.perf_counter()
            char_name = self.facial_detector.params.dir_character_names[i]
            file_name = (f"Task1"
                         f"_{char_name}"
                         f"_Positive"
                         f"_{self.facial_detector.params.imgs_per_class}"
                         f"_{self.facial_detector.params.dir_descriptor_type}"
                         f"_{self.facial_detector.params.number_positive_examples}.npy")
            save_dir = self.facial_detector.params.dir_descriptor
            file_path = os.path.join(save_dir, file_name)

            if os.path.exists(file_path):
                positive_desc = np.load(file_path, allow_pickle=True)
                print(f"Loaded {char_name}'s positive descriptors!")
            else:
                positive_desc = self.facial_detector.get_positive_descriptors(i)
                np.save(file_path, positive_desc)

            positive_descriptors.append(positive_desc)
            end_time = time.perf_counter()
            print(f"Processed positive descriptors for {char_name} in: {end_time - start_time} seconds")

        final_positive_descriptors = np.concatenate(positive_descriptors, axis=0)
        return final_positive_descriptors

    def get_negative_descriptors(self):
        negative_descriptors = []

        for i in range(4):
            start_time = time.perf_counter()
            char_name = self.facial_detector.params.dir_character_names[i]
            file_name = (f"Task1"
                         f"_{char_name}"
                         f"_Negative"
                         f"_{self.facial_detector.params.imgs_per_class}"
                         f"_{self.facial_detector.params.dir_descriptor_type}"
                         f"_{self.facial_detector.params.number_positive_examples}.npy")
            save_dir = self.facial_detector.params.dir_descriptor
            file_path = os.path.join(save_dir, file_name)

            if os.path.exists(file_path):
                negative_desc = np.load(file_path, allow_pickle=True)
                print(f"Loaded {char_name}'s negative descriptors!")
            else:
                negative_desc = self.facial_detector.get_negative_descriptors(i)
                np.save(file_path, negative_desc)

            negative_descriptors.append(negative_desc)
            end_time = time.perf_counter()
            print(f"Processed negative descriptors for {char_name} in: {end_time - start_time} seconds")

        final_negative_descriptors = np.concatenate(negative_descriptors, axis=0)
        return final_negative_descriptors

    def get_or_create_descriptors(self, Positive: bool):
        """
        Loads in or creates descriptors based on type (Positive/Negative) and returns them.
        :param Positive: Type of descriptor
        :return: Feature array
        """
        descriptor_type = "Positive" if Positive else "Negative"
        if Positive:
            desc_path = os.path.join(
                f"{self.params.dir_save_files}",
                f"PositiveDescriptors_"
                f"HogSize{self.params.hog_cell_size}_"
                f"{self.params.number_positive_examples}"
                f"ImgSize{self.params.window_size}.npy")
        else:
            desc_path = os.path.join(
                f"{self.params.dir_save_files}",
                f"NegativeDescriptors_"
                f"HogSize{self.params.hog_cell_size}_"
                f"{self.params.number_negative_examples}"
                f"ImgSize{self.params.window_size}.npy")

        if os.path.exists(desc_path):
            features = np.load(desc_path, allow_pickle=True)
            print(
                f"Loaded in {descriptor_type} Descriptors of "
                f"cell_size={self.params.hog_cell_size}, "
                f"cnt_pos_examples={self.params.number_positive_examples}.")
        else:
            print(
                f"Creating {descriptor_type} Descriptors of "
                f"cell_size={self.params.hog_cell_size}, "
                f"cnt_pos_examples={self.params.number_positive_examples}.")

            features = self.get_positive_descriptors() if Positive else self.get_negative_descriptors()

            print(
                f"Created {descriptor_type} Descriptors of "
                f"cell_size={self.params.hog_cell_size}, "
                f"cnt_pos_examples={self.params.number_positive_examples}.")

        print(f"Loaded {descriptor_type} descritors!")
        return features

    @staticmethod
    def save_arrays_to_task1_folder(detections, scores, file_names):
        parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        task1_folder_path = os.path.join(parent_directory, 'task1')
        os.makedirs(task1_folder_path, exist_ok=True)

        detections_path = os.path.join('..', 'task1', 'detections_all_faces.npy')
        scores_path = os.path.join('..', 'task1', 'scores_all_faces.npy')
        file_names_path = os.path.join('..', 'task1', 'file_names_all_faces.npy')

        np.save(detections_path, detections)
        np.save(scores_path, scores)
        np.save(file_names_path, file_names)

    def run(self):
        positive_descriptors = self.get_or_create_descriptors(Positive=True)
        negative_descriptors = self.get_or_create_descriptors(Positive=False)

        training_examples = np.concatenate((np.squeeze(positive_descriptors), np.squeeze(negative_descriptors)), axis=0)
        train_labels = np.concatenate((np.ones(len(positive_descriptors)), np.zeros(len(negative_descriptors))))
        self.facial_detector.trainer.load_or_train_classifier(training_examples, train_labels)

        detections, scores, file_names = self.facial_detector.run()
        self.save_arrays_to_task1_folder(detections, scores, file_names)
        if self.params.has_annotations:
            eval_detections(self.facial_detector.params.dir_save_files, detections, scores, file_names)
            show_detections_with_ground_truth(detections, scores, file_names, self.params)
        else:
            show_detections_without_ground_truth(detections, scores, file_names, self.params)
