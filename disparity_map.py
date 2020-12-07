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

def depth_map(left, right, result, window_size, search_size, f, t):
	'''
	Parameters:
		left-- name of left stereo pair image file
		right-- name of right stereo pair image file
		result-- file name results will be stored in
		window_size-- half size of possible blocks
		search_size-- maximum number of pixels away we can look for matching blocks in img2
		f-- focal length (scaled if image was resized)
		t-- baseline (scaled if image was resized)
	Returns:
		tuple of--
			1) matrix containing displacement between xl and xr for a pixel (xl - xr)
			2) width of depth map + rgb image for point cloud
			3) height of depth map + rgb image for point cloud        
	'''

	# resized to 244 x 300 for speed as recommended in matlab stencil
	im_left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY);
	im_right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY);
	[h,w] = im_left.shape;

	depth = np.full((h, w), 256, dtype='uint16');	
	half_window_size = int(window_size/2);

	print("creating disparity map...")
	for y in range(half_window_size, h-half_window_size):
		for x in range(half_window_size, w-half_window_size):
			block = get_block(im_left, y, x, half_window_size)
			depth[y, x] = f * t/float(distance_to_best_block(block, (y, x), im_right, search_size, half_window_size))
	print("created disparity map!")

	cv2.imwrite("./point_cloud_rgb_data/" + result, im_left[half_window_size:h-half_window_size, half_window_size:w-half_window_size])
	cv2.imwrite("./disparity_maps/" + result, depth[half_window_size:h-half_window_size, half_window_size:w-half_window_size])

	return (depth[half_window_size:h-half_window_size, half_window_size:w-half_window_size], w - (2 * half_window_size), h - (2 * half_window_size))

def create_depth_map(disparity_matrix, f, t, scale):
	'''
	Parameters:
		disparity_matrix-- matrix containing displacement between xl and xr for a pixel (xl - xr)
		f-- focal length in pixels
		t-- baseline in mm
	
	Returns:
		depth map in mm	
	'''
	return ((f/scale) * (t/scale)) / disparity_matrix

#perhaps get fourier decomposition of images
#look at all three channels seperately after converting to LAB space
#--> three depth proposal, look for one to select/combine from
# also perform stereo matching on the edge image 
# --> rgb est, and edge estimate, and assign a weight to each
#median works best around edges
#smooth--> mean works better
#convert into a mesh surface-- meshlab, poisson reconstruction
def display_depth_map(depth_map_file, color_img_file, w, h, fx, fy, cx, cy):
	'''
	Parameters:
		fx-- focal length in x dir (scaled if resized)
		fy-- focal length in y dir (scaled if resized)
		cx-- x axis principle point (scaled if resized)
		cy-- y axis principle point (scaled if resized)
	'''

	
	img = o3d.io.read_image(color_img_file)
	depth = o3d.io.read_image(depth_map_file)

	rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)

	o3d_pinhole = o3d.camera.PinholeCameraIntrinsic()
	o3d_pinhole.set_intrinsics(w, h, fx, fy, cx, cy)

	pcd_from_depth_map = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_pinhole)
	pcd_from_depth_map.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	visualizer = JVisualizer()
	visualizer.add_geometry(pcd_from_depth_map)
	visualizer.show()

depth_map_data = depth_map('./data/2006/tsukuba_L.png', './data/2006/tsukuba_R.png', "2006/tsukuba.png", 10, 15, 588.503, 16.19)
display_depth_map("./disparity_maps/2006/tsukuba.png", './point_cloud_rgb_data/2006/tsukuba.png', depth_map_data[1], depth_map_data[2], 588.503, 588.503, 119.102, 119.102)