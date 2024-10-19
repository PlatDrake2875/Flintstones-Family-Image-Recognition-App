import cv2 as cv
import numpy as np
import os.path
import glob
from Parameters import *


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = os.path.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Apasa orice tasta pentru a continua...')
        cv.imshow('image', np.uint8(image))
        cv.waitKey(0)


def show_detections_with_ground_truth(detections, scores, file_names, params, index=0):
    """
    Display and save annotated images. Draw predicted and correct bounding boxes.
    :param index:
    :param detections: NumPy array of shape NX4, where N is the number of detections across all images.
    :param scores: NumPy array of length N, scores for all detections across all images.
    :param file_names: NumPy array of length N, file names corresponding to each detection.
    :param params: Parameters object with configuration.
    """

    # Ensure that detections is a NumPy array
    detections = np.array(detections)
    scores = np.array(scores)
    file_names = np.array(file_names)

    ground_truth_bboxes = np.loadtxt(params.dir_paths_annotations[index], dtype='str')
    test_files = glob.glob(os.path.join(params.dir_test_examples, '*.jpg'))

    for test_file in test_files:
        image = cv.imread(test_file)
        short_file_name = os.path.basename(test_file)

        indices_detections_current_image = np.where(file_names == short_file_name)[0]
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        # Draw predicted bounding boxes and scores
        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])),
                         (0, 0, 255), thickness=1)
            cv.putText(image, 'score: ' + str(current_scores[idx])[:4], (int(detection[0]), int(detection[1])),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw ground truth bounding boxes
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]
        for detection in annotations[:, 1:]:
            cv.rectangle(image, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])),
                         (0, 255, 0), thickness=1)

        # Save and display the image with annotations
        cv.imwrite(os.path.join(params.dir_save_files, "detections_" + short_file_name), image)
        print('Press any key to continue...')
        cv.imshow('image', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
