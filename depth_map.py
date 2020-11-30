import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

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
	#img2_shift = shift_x(img2, d)
	#img1 = img1[:img2_shift.shape[0], :img2_shift.shape[1]]
	return np.sum(np.abs(img1 - img2))

def get_block(img, y, x, half_window_size):
	row_start = y - half_window_size
	row_end = y + half_window_size + 1

	col_start = x - half_window_size
	col_end = x + half_window_size + 1

	return img[row_start:row_end, col_start:col_end]

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
	[block1_y, block1_x] = block1_coordinates
	
	best_sad = float('inf')
	best_y = block1_y
	best_x = block1_x

	for y in range(max(half_window_size, block1_y - search_size), min(img2.shape[0] - half_window_size, block1_y + search_size)):
		for x in range(max(half_window_size, block1_x - search_size), min(img2.shape[0] - half_window_size, block1_x + search_size)):

			block2 = get_block(img2, y, x, half_window_size)

			#if(block1.shape != block2.shape):
			#	continue

			curr_sad = sad(block1, block2)
			if(curr_sad < best_sad):
				best_sad=curr_sad
				best_y = y
				best_x = x
	
	y_diff_sq = (block1_y - best_y) * (block1_y - best_y)
	x_diff_sq = (block1_x - best_x) * (block1_x - best_x)

	return math.sqrt(y_diff_sq + x_diff_sq)

def depth_map(left, right, window_size, search_size):
	im_left = cv2.resize(cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY), (300, 244));
	im_right = cv2.resize(cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY), (300, 244));
	[h,w] = im_left.shape;

	disparity = np.zeros((h, w), dtype='uint8');
	
	half_window_size = int(window_size/2);

	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			block = get_block(im_left, y, x, half_window_size)
			disparity[y, x] = distance_to_best_block(block, (y, x), im_right, search_size, half_window_size)
			
			cv2.imshow("Disparity Map", disparity)
			cv2.waitKey(10)

	cv2.waitKey(5000)

	scale = 255.0 / search_size;
	disparity = uint8(disparity * scale);

	plt.imshow(disparity);
	plt.show()

depth_map("./data/bowling_1.png", "./data/bowling_2.png", 7, 30)



