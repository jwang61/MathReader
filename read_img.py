"""
This script reads an image of the template with hand-written character and attempts
to separate the characters and classify them
"""
import argparse

import numpy as np
import cv2
import tensorflow as tf

# Image processing parameters
BOX_SIZE = (28, 28)
EPSILON = 0.02
EMPTY_THRESH = 15

def overlap(bbox1, bbox2):
    if bbox1[0] > bbox2[0] + bbox2[2]:
        return False
    if bbox1[0] + bbox1[2] < bbox2[0]:
        return False
    if bbox1[1] > bbox2[1] + bbox2[3]:
        return False
    if bbox1[1] + bbox1[3] < bbox2[1]:
        return False
    return True

def bbox_to_points(bbox):
    return (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])

def main(args):
    # Read the image
    ori = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    img_size = ori.shape[0] * ori.shape[1]

    # Perform filters to find contours
    img = cv2.GaussianBlur(ori, (5,5), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,11,2)

    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours for good non-overlapping squares
    contours = [cnt for cnt in contours if (cv2.contourArea(cnt) > 0.01 * img_size and cv2.contourArea(cnt) < 0.2 * img_size)]

    squares = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True)*EPSILON, True)
        bounding_rect = cv2.boundingRect(approx)
        if len(approx) != 4:
            continue
        no_overlap = True
        for square in squares:
            if overlap(bounding_rect, square):
                no_overlap = False
                break

        if no_overlap:
            squares.append(bounding_rect)

    print("{} Squares found. Classifying...".format(len(squares)))

    # Classify each box
    graph_def = tf.GraphDef()
    with tf.gfile.Open(args.model, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        kernel = np.ones((3,3), np.uint8)
        output = ""
        for square in squares[::-1]:
            x, y, w, h = square
            h_crop = h//7
            w_crop = h//7

            # Get input image
            cropped_img = img[y+h_crop: y+h-h_crop, x+w_crop:x+w-w_crop]
            resized_img = cv2.resize(cropped_img, BOX_SIZE)
            resized_img = cv2.erode(resized_img, kernel, iterations=1)
            resized_img = cv2.bitwise_not(resized_img)

            if np.mean(resized_img) < EMPTY_THRESH:
                # If Image is empty
                output += " "
            else:
                x = np.expand_dims(resized_img, axis=0)
                x = x.astype('float32')
                x_in = x / 255
                res = sess.run(args.model_output, { args.model_input : x_in})
                output += str(np.argmax(res))

    print("Classified Output: " + output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Name of image to read")
    parser.add_argument("--model", default="model.pb", help="Name of classifer model to use")
    parser.add_argument("--model_input", default="input_1:0", help="Name of model input node")
    parser.add_argument("--model_output", default="Softmax:0", help="Name of model output node")
    args = parser.parse_args()
    main(args)
