# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def find_puzzle(image, debug=False):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find sudoku puzzle outline. "
            "Try debugging your thresholding and contour steps."))

    # check to see if we are visualizing the outline of the detected
    # sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down birds eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    # return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)


def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    blur = cv2.GaussianBlur(cell, (3, 3), 0)

    cv2.normalize(blur, blur, 0, 255, norm_type=cv2.NORM_MINMAX)
    if debug:
        cv2.imshow("augment", blur)
        cv2.waitKey(0)

    # thresh = cv2.threshold(blur, 0, 255,
    #     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(blur, 200, 255,
                           cv2.THRESH_BINARY_INV)[1]
    left = cv2.countNonZero(thresh[10:-10, 10:-10])
    if debug:
        cv2.imshow("Before clear", thresh)
        cv2.waitKey(0)
    thresh_copy = clear_border(thresh)
    origin = cv2.countNonZero(thresh_copy)
    if left > 0 and origin < max(8, left):
        thresh = thresh[5:-5, 5:-5]
        thresh = hard_border_clear(thresh)
    else:
        thresh = thresh_copy

    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None

    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    # c = max(cnts, key=cv2.contourArea)
    (h, w) = thresh.shape
    true_cnts = []
    for cnt in cnts:  # traverse all contours
        tmp_mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(tmp_mask, [cnt], -1, 255, -1)
        if cv2.countNonZero(tmp_mask) / float(w * h) > 0.001:
            true_cnts += [cnt]
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, true_cnts, -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.006:
        return None

    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # digit = cv2.dilate(digit, (2, 2), iterations=1)

    digit = center_digit(digit)
    pad_length = max(int(digit.shape[0] * 0.3), 2)
    digit = np.pad(digit, pad_length, 'constant')

    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("final digit", digit)
        cv2.waitKey(0)

    # return the digit to the calling function
    return digit


def center_digit(img):
    left = up = 0
    down, right = img.shape
    for i in range(img.shape[1]):
        if (img[:, i] != 0).any():
            left = i
            break
    for i in range(img.shape[1]-1, -1, -1):
        if (img[:, i] != 0).any():
            right = i
            break
    for i in range(img.shape[0]):
        if (img[i, :] != 0).any():
            up = i
            break
    for i in range(img.shape[0]-1, -1, -1):
        if (img[i, :] != 0).any():
            down = i
            break

    down = min(img.shape[0]-1, down+1)
    up = max(0, up-1)
    left = max(0, left-1)
    right = min(img.shape[1]-1, right+1)

    new_img = img[up: down, left: right]
    h, w = new_img.shape
    m = max(h, w)
    new_img = np.pad(new_img, (((m-h)//2, (m-h)//2), ((m-w)//2, (m-w)//2)))

    return new_img.astype('uint8')


def hard_border_clear(img):
    h, w = img.shape

    x_axis = np.sum(img, axis=0)
    x_threshold = np.max(x_axis) * 0.95
    if x_threshold > h * 255 * 0.75:
        for k in range(w):
            if np.sum(img[:, k]) > x_threshold:
                img[:, k] = 0

    y_axis = np.sum(img, axis=1)
    y_threshold = np.max(y_axis) * 0.95

    if y_threshold > w * 255 * 0.75:
        for k in range(h):
            if np.sum(img[k, :]) > y_threshold:
                img[k, :] = 0
    return img


if __name__ == '__main__':
    import os
    test_img_path = '../../handwriting1.jpg'
    if os.path.isfile(test_img_path):
        img = cv2.imread(test_img_path)
        _, w = find_puzzle(img, debug=False)
        # cv2.imshow('1', w)
        # cv2.waitKey(0)
        y, x = w.shape
        extract_digit(w[y//9: y // 9 * 2, x//9*2:x//9*3], debug=True)
