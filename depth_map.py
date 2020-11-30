import numpy as np
import cv2

def shift_x(img, d):
	'''
	Parameters:
		img-- 3 channel (row, col, colors) numpy array representing a picture
		d-- number of columns (x coordinates) to shift
			if d is positive -- shift right
			if d is negative -- shift left
	Returns:
		img-- shifted 3 channel (row, col, colors) numpy array representing a picture

	shifts the image by removing first d (if negative shift) columns or last d (if positive shift) columns
	'''
	if(d>0):
		return img[:, :-d, :];
	else:
		return img[:, -d:, :];

def sad(img1, img2, d):
	'''
	Parameters:
		img1-- 3 channel (row, col, colors) numpy array representing a picture
		img2-- 3 channel (row, col, colors) numpy array representing a picture
		d-- number of columns (x coordinates) to shift img2
			if d is positive -- shift right
			if d is negative -- shift left
	Returns:
		float -- sum of absolute differences between img1 and shifted img2
	'''
	img2_shift = shift_x(img2, d)
	img1 = img1[:img2_shift.shape[0], :img2_shift.shape[1]]
	return np.sum(np.abs(img1 - img2))

def stereo_match(left, right, window_size, search_size):
	im_left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY);
	im_right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY);
	[h,w] = im_left.shape;

	disparity = np.zeros(h, w, dtype='uint8');
	
	half_window_size = int(window_size/2);

	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			print(f'{y}, {x}');

rand = np.random.rand(3,2,3)
print(rand)
print(shift_x(rand, -1))

