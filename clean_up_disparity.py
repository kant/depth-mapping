import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
from scipy.stats import mode
import numpy as np

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

#replace the middle pixel not the whole block
def set_block(img, y, x, half_window_size, val):
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

	img[row_start:row_end, col_start:col_end] = val

	return img

#try median
def median_filter_block(block):
	'''
	Parameters:
		block-- a block of an image
	Returns:
		a block of the same size where all pixels are filled with the same number (the most frequently appearing color)
	'''
	block[:,:] = mode(block, axis=None)[0]
	return block

def edge_aware_mode_filter(image, edges, mask_set, window_size):
	if(window_size==0):
		return image;
	[h, w] = image.shape
	half_window_size = int(window_size/2)
	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			edge_block = get_block(edges, y, x, half_window_size)
			if(np.any(edge_block==1) or mask_set[y, x] == True):
				continue;
			img_block = get_block(image, y, x, half_window_size)
			mask_set[y, x] = True
			image[y, x] = np.median(img_block)

	return edge_aware_mode_filter(image, edges, mask_set, window_size - 1)

def filter_map(map_file, left, right, result, disparity_window_size):
	'''
	Parameters:
		map_file-- name of depth map file to clean up 
		left-- name of left stereo file used to make the depth map
		right-- name of right stereo file used to make the depth map
		disparity_window_size-- window size used to create the depth map, used to crop the combined edge image
	'''
	disparity_map = cv2.cvtColor(cv2.imread(map_file), cv2.COLOR_BGR2GRAY);
	left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY);
	right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY);

	half_window_size = int(disparity_window_size/2)
	[h, w] = right.shape

	edges_left = feature.canny(left, sigma=3)[half_window_size:h-half_window_size, half_window_size:w-half_window_size]
	edges_right = feature.canny(right, sigma=3)[half_window_size:h-half_window_size, half_window_size:w-half_window_size]
	edges_disparity = feature.canny(disparity_map, sigma=3)
	edges_right[edges_left==1] = 1
	edges_right[edges_disparity==1] = 1
	edges_both = edges_right
	
	filtered_map = edge_aware_mode_filter(disparity_map, edges_both, np.zeros(edges_left.shape), 100)

	plt.imshow(filtered_map, cmap='Greys_r')
	plt.show()


filter_map("./disparity_maps/visible/2006/bowling.png", './data/2006/bowling_L.png', './data/2006/bowling_R.png', 'bowling.png', 15)