import os
import glob
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.exposure import exposure
import EvalUtils as eval_utils


# READ IMAGES
def read_image_data(path, idx_stop=1000):
    """
    Reads images of a character's dataset
    :param path:
    :param idx_stop:
    :return: a dictionary {image_name: [image]}
    """
    images_path = os.path.join(path, "*.jpg")
    file_names = glob.glob(images_path)

    images_dict = {}
    for image_file in file_names:
        key = image_file[-8:][:4]
        if int(key) > idx_stop:
            break
        image = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

        if key not in images_dict:
            images_dict[key] = [image]
        else:
            images_dict[key].append(image)

    return images_dict


# READ IMAGES METADATA
def read_image_metadata(path, index, idx_stop=1000):
    """
    Parses the metadata of the image (image_index, coords, label) up until some index
    :param idx_stop:
    :param path:
    :param index:
    :return: A list of tuples with the image data (image_index, coords, label)
    """
    char_file = glob.glob(f"{path}/*.txt")[index]
    meta_data = []

    with open(char_file, "r") as f:
        line = f.readline().split()

        while len(line):
            image_index = line[0][:4]

            if int(image_index) > idx_stop:
                break

            coords = list(map(int, line[1:5]))
            label = line[5]

            meta_data.append((image_index, coords, label))

            line = f.readline().split()

    return meta_data


# GENERATE IMAGE -> COORDS DICTIONARY
def get_image_coords_dict(char_metadata):
    """
    Generates a dictionary of coordatinates for every image of a protagonist.
    Every image gets a list of coordinates for the characters it contains.
    :param char_metadata: List of tuples (image_cnt, coords, character_label)
    :return:
    """
    image_coords_dict = {}
    for image_cnt, coords, _ in char_metadata:
        if image_cnt not in image_coords_dict:
            image_coords_dict[image_cnt] = [coords]
        else:
            image_coords_dict[image_cnt].append(coords)

    return image_coords_dict


# DISPLAY ROI (REGION OF INTEREST) OF AN IMAGE
def display_image_with_roi(image, x_min, y_min, x_max, y_max):
    """
    Display an image with a green rectangle marking the Region of Interest (ROI).

    Parameters:
    - image: The original image (as a NumPy array).
    - x_min, y_min: The coordinates of the top-left corner of the ROI.
    - x_max, y_max: The coordinates of the bottom-right corner of the ROI.
    """
    # Copy the image to avoid modifying the original
    image_with_roi = image.copy()

    # Draw a green rectangle on the image
    cv.rectangle(image_with_roi, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the image with ROI
    plt.imshow(cv.cvtColor(image_with_roi, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.axis('off')
    plt.title('Image with ROI')
    plt.show()


# DISPLAY TWO REGIONS OF INTERESTS OF AN IMAGE
def display_image_with_rois(image, coords_1, coords_2):
    """
    Display an image with a green and a blue rectangle marking the Region of Interest (ROI).

    Parameters:
    - image: The original image (as a NumPy array).
    - x_min, y_min: The coordinates of the top-left corner of the ROI.
    - x_max, y_max: The coordinates of the bottom-right corner of the ROI.
    """
    # Copy the image to avoid modifying the original
    image_with_roi = image.copy()

    # Draw a green rectangle on the image
    print(coords_1)
    print(coords_2)
    for coords in coords_1:
        x_min, y_min, x_max, y_max = coords[0], coords[1], coords[2], coords[3]
        cv.rectangle(image_with_roi, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    x_min, y_min, x_max, y_max = coords_2[0], coords_2[1], coords_2[2], coords_2[3]
    cv.rectangle(image_with_roi, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Display the image with ROI
    plt.imshow(cv.cvtColor(image_with_roi, cv.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.axis('off')
    plt.title('Image with ROI')
    plt.show()


# HOG DEBUGGING (DISPLAY ORIGINAL + HOG)
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


def check_valid_coords(image_coords, candidate_coords):
    """
    Checks if two ROIs have IOU over 0.5%
    :param image_coords:
    :param candidate_coords:
    :return: True if they don't, False otherwise
    """
    for coords in image_coords:
        if eval_utils.intersection_over_union(coords, candidate_coords) > 0.05:
            return False
    return True


def do_rectangles_intersect(rect1, rect2):
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    if x1_max < x2_min or x2_max < x1_min:
        return False

    if y1_max < y2_min or y2_max < y1_min:
        return False

    return True


def does_any_rectangle_intersect(rectangles, rect2):
    for rect1 in rectangles:
        if do_rectangles_intersect(rect1, rect2):
            return True

    return False


# VALIDATION - TASK 1
# READ VALIDATION IMAGES
def get_validation_images():
    """
    Returns a dictionary of the form xxxx: [image] where xxxx ranges from 0000 to 1000
    :return:
    """
    images_path = os.path.join("..", "validare", "validare", "*.jpg")
    images_paths = glob.glob(images_path)
    validation_examples_images = {"{:04d}".format(idx + 1): cv.imread(image_file, cv.IMREAD_GRAYSCALE) for
                                  idx, image_file in
                                  enumerate(images_paths)}
    return validation_examples_images


# GET VALIDATION METADATA FILE
def get_validation_examples(target_file_name):
    validation_examples_gt = glob.glob(os.path.join("..", "validare", "*.txt"))
    validation_example_file = None

    for vegt in validation_examples_gt:
        if os.path.basename(vegt) == target_file_name:
            validation_example_file = vegt
            break

    return validation_example_file


# READ VALIDATION METADATA
def get_validation_metadata(validation_example_file):
    """
    Returns a list a tuples (image_index, face_coords)
    :param validation_example_file:
    :return: (image_index, coords)
    """
    validation_metadata = []
    with open(validation_example_file, "r") as f:
        line = f.readline().split()

        while len(line):
            img_cnt = line[0][:4]
            x_min, y_min, x_max, y_max = tuple(map(int, line[1:5]))
            validation_metadata.append((img_cnt, (x_min, y_min, x_max, y_max)))

            line = f.readline().split()
    return validation_metadata


# GET VALIDATION DATA
def get_validation_data_task_1(target_file_name):
    """
    Reads images, metadata and makes labels for images.
    :param target_file_name:
    :return: (validation_images, validation_metadata, validation_labels)
    """
    validation_ex_imgs = get_validation_images()
    validation_example_file = get_validation_examples(target_file_name)
    validation_metadata = get_validation_metadata(validation_example_file)
    validation_labels = [1] * len(validation_metadata)

    return validation_ex_imgs, validation_metadata, validation_labels
