import os


class Parameters:
    def __init__(self):
        # HOG Parameters
        self.pixels_per_cell = 6
        self.hog_cell_size = 2

        # Image Data Parameters
        self.window_size = 128
        self.number_positive_examples = 6977
        self.number_negative_examples_per_image = 2
        self.number_negative_examples = self.number_positive_examples * self.number_negative_examples_per_image
        self.imgs_per_class = 1000
        self.threshold = 0

        # Visualization Parameters
        self.has_annotations = True

        # Directory Setup
        self.base_dir = "../antrenare"
        self.dir_character_names = ["barney", "betty", "fred", "wilma"]
        self.dir_positive_examples = [self.construct_path(self.base_dir, char_name) for char_name in
                                      self.dir_character_names]
        self.dir_negative_examples = self.dir_positive_examples
        self.dir_save_files = "../saves"
        self.dir_descriptor = self.construct_path(self.dir_save_files,
                                                  f"ppc_{self.pixels_per_cell}"
                                                  f"_HogCellSize_{self.hog_cell_size}"
                                                  f"_Window_{self.window_size}")
        self.dir_descriptor_type = (f"ppc_{self.pixels_per_cell}"
                                    f"_HogCellSize_{self.hog_cell_size}"
                                    f"_Window_{self.window_size}")

        self.dir_test_examples = "../testare"
        self.dir_paths_annotations = ["../testare/task1_gt_testare.txt"]
        self.dir_paths_annotations.extend([self.construct_path(self.dir_test_examples,
                                                               f"task2_{self.dir_character_names[index]}_gt_validare.txt")
                                           for index in range(4)])

        # Ensure Directories Exist
        self.create_directory_if_not_exists(self.dir_save_files)
        self.create_directory_if_not_exists(self.dir_descriptor)

    @staticmethod
    def construct_path(*path_segments):
        """
        Constructs a file path from given segments.
        """
        return os.path.join(*path_segments)

    @staticmethod
    def create_directory_if_not_exists(directory):
        """
        Creates a directory if it does not exist.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}!")
        else:
            print(f"Directory: {directory} already exists!")
