import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import open3d as o3d
from open3d import JVisualizer


def sad(img1, img2):
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

	return np.sum(np.abs(np.subtract(img1, img2, dtype=np.float)))

def ssd(img1, img2):
	diff = img1 - img2
	return np.sum(diff * diff)


def get_block(img, y, x, half_window_size):
	'''
	Parameters:
		img1- 3 channel (row, col, colors) numpy array representing a picture
		x and y- coordinates of center pixel or block
		half_window_size-- half the size of the desired block
	Returns:
		get_block -- gets the block of (half_window_size * 2 + 1) centered at y, x
	'''
	row_start = y - half_window_size
	row_end = y + half_window_size + 1

	col_start = x - half_window_size
	col_end = x + half_window_size + 1

	return np.array(img[row_start:row_end, col_start:col_end])

def distance_to_best_block(block1, block1_coordinates, img2, search_size, half_window_size):
	'''
	Parameters:
		block1-- 3 channel (row, col, colors) numpy array representing a block of a picture
		block1_coordinates-- tuple(r, w) or (y, x) representing location of center of block1 (used to calculate distance)
		img2-- 3 channel (row, col, colors) numpy array representing a picture
		search_size-- maximum number of pixels away we can look for matching blocks in img2
		window_size-- half size of possible blocks
	Returns:
		float distance between center of block1 and the best matching block within search_size

	iterate through all blocks of (2 * window_size + 1) in img2 no further than search_size away
	find the block with the minimum SAD (sum of absolute differences) to block 1 and retain its location coordinates
	return the distance between block 1 and the best block.
	'''
	[y, block1_x] = block1_coordinates
	
	best_sad = float('inf')
	best_x = block1_x

	for x in range(max(half_window_size, block1_x - search_size), min(img2.shape[1] - half_window_size, block1_x + search_size)):

		block2 = get_block(img2, y, x, half_window_size)

		curr_sad = sad(block1, block2)
		if(curr_sad < best_sad):
			best_sad=curr_sad
			best_x = x
			best_block = block2

	return abs(block1_x - best_x)

def disparity_map(left, right, window_size, search_size, result):
	'''
	Parameters:
		left-- name of left stereo pair image file
		right-- name of right stereo pair image file
		window_size-- half size of possible blocks
		search_size-- maximum number of pixels away we can look for matching blocks in img2
	Returns:
		matrix containing displacement between xl and xr for a pixel (xl - xr)
	'''

	# resized to 244 x 300 for speed as recommended in matlab stencil
	im_left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY);
	im_right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY);
	[h,w] = im_left.shape;

	disparity = np.zeros((h, w), dtype='uint8');	
	half_window_size = int(window_size/2);

	scale = 255.0/search_size

	print("creating disparity map...")
	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			block = get_block(im_left, y, x, half_window_size)
			disparity[y, x] = scale * float(distance_to_best_block(block, (y, x), im_right, search_size, half_window_size))
		cv2.imshow("Disparity Map", disparity)
		cv2.waitKey(10)
	print("created disparity map!")

	cv2.imwrite("./disparity_maps/" + result, disparity)

	return disparity

disparity_map('./data/bowling_L.png', './data/bowling_R.png', 15, 100, "bowling.png")

