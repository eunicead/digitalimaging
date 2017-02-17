import sys
import numpy as np
import dlib 
import cv2

shape_predictor = "shape_predictor_68_face_landmarks.dat"

prediction = dlib.shape_predictor(shape_predictor)
detection = dlib.get_frontal_face_detector()


points_of_the_contour = list(range(17, 68))

points_of_the_lbrow = list(range(22, 27))
points_of_the_rbrow = list(range(17, 22))

points_of_the_leye = list(range(42, 48))
points_of_the_reye = list(range(36, 42))

points_of_the_noise = list(range(27, 35))

points_of_the_mouth = list(range(48, 61))

list_of_the_points = (points_of_the_contour + points_of_the_lbrow + points_of_the_rbrow + points_of_the_leye + points_of_the_reye + points_of_the_noise + points_of_the_mouth)

array_of_the_points = [ points_of_the_lbrow + points_of_the_rbrow + points_of_the_leye + points_of_the_reye, points_of_the_noise + points_of_the_mouth,]

def get_matrix(img):
	array = detection(img,1)
	return np.matrix([[p.x, p.y] for p in prediction(img, array[0]).parts()])

def contouring(points_of_img1, points_of_img2):
    points_of_img1 = points_of_img1.astype(np.float64)
    points_of_img2 = points_of_img2.astype(np.float64)

    c1 = np.mean(points_of_img1, axis=0)
    c2 = np.mean(points_of_img2, axis=0)
    points_of_img1 = points_of_img1 - c1
    points_of_img2 = points_of_img2 - c2

    s1 = np.std(points_of_img1)
    s2 = np.std(points_of_img2)
    points_of_img1 = points_of_img1 /s1
    points_of_img2 = points_of_img2 /s2

    M, S, N = np.linalg.svd(points_of_img1.T * points_of_img2)

    R = (M * N).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def warp(modification, dshape, img):
	newimg = np.zeros(dshape, dtype=img.dtype)
	cv2.warpAffine(img, modification[:2], (dshape[1], dshape[0]), dst=newimg, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
	return newimg
def convex_hull(img, marks, color):
	marks = cv2.convexHull(marks)
	cv2.fillConvexPoly(img, marks, color=color)

def place_for_the_mask(img, img_marks):
	img = np.zeros(img.shape[:2], dtype=np.float64)

	for group in array_of_the_points:
		convex_hull(img, img_marks[group], color=1)
	img = np.array([img, img, img]).transpose((1, 2, 0))
	img = (cv2.GaussianBlur(img, (11, 11), 0) > 0) * 1.0
	img = cv2.GaussianBlur(img, (11,11), 0)
	return img
def main():
	img1 = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
	img1 = cv2.resize (img1, (img1.shape[1]*1, img1.shape[0]*1))
	out1 = get_matrix(img1)

	img1_marks = out1;

	img2 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
	img2 = cv2.resize (img2, (img2.shape[1]*1, img2.shape[0]*1))
	out2 = get_matrix(img2)

	img2_marks = out2;

	modification = contouring(img1_marks[list_of_the_points], img2_marks[list_of_the_points])
	place_where_paste = place_for_the_mask(img2, img2_marks)
	face_to_copy = warp( modification, img1.shape, place_where_paste)
	mix = np.max([place_for_the_mask(img1, img1_marks), face_to_copy ], axis=0)
	warp_img2 = warp(modification, img1.shape, img2)

	finalresult = img1 * (1.0 - mix) + warp_img2 * mix

	cv2.imwrite('result.jpg', finalresult) 


main()
print("Done, please see the result on result.jpg")






