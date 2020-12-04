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
	dist = abs(block1_x - best_x)

	return max(1, dist)

def disparity_map(left, right, window_size, search_size):
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

	disparity = np.full((h, w), 256, dtype='uint16');	
	half_window_size = int(window_size/2);

	print("creating disparity map...")
	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			block = get_block(im_left, y, x, half_window_size)
			disparity[y, x] = 588.503 * 16.19/float(distance_to_best_block(block, (y, x), im_right, search_size, half_window_size))
	print("created disparity map!")

	return disparity

def create_depth_map(disparity_matrix, f, t):
	'''
	Parameters:
		disparity_matrix-- matrix containing displacement between xl and xr for a pixel (xl - xr)
		f-- focal length in pixels
		t-- baseline in mm
	
	Returns:
		depth map in mm	
	'''
	return (f * t) / disparity_matrix

def display_depth_map(depth_map_file, color_img_file, fx, fy, cx, cy):
	'''
	Parameters:
		fx-- focal length in x dir
		fy-- focal length in y dir
		cx-- x axis principle point
		cy-- y axis principle point
	'''

	
	img = o3d.io.read_image(color_img_file)
	depth = o3d.io.read_image(depth_map_file)

	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)

	o3d_pinhole = o3d.camera.PinholeCameraIntrinsic()
	o3d_pinhole.set_intrinsics(290, 194, 588.503, 588.503, 119.102, 119.102)

	pcd_from_depth_map = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pinhole)
	pcd_from_depth_map.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	visualizer = JVisualizer()
	visualizer.add_geometry(pcd_from_depth_map)
	visualizer.show()

depth_map = disparity_map('./data/2006/tsukuba_L.png', './data/2006/tsukuba_R.png', 10, 15)
print(np.max(depth_map))
print(np.min(depth_map))
print(depth_map[0,0])
cv2.imwrite("./disparity_maps/2006/tsukuba.png", depth_map)
display_depth_map("./disparity_maps/2006/tsukuba.png", './data/2006/tsukuba_L.png', 588.503, 588.503, 119.102, 119.102)

