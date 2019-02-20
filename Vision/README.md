# Computer Vision Fundmentals

## Concepts
1. [Power of Cameras](https://www.youtube.com/watch?time_continue=23&v=lCPWJEEzUeo)
1. [Setting up the Problem](https://www.youtube.com/watch?v=aIkAcXVxf2w)
	* [Quiz](images/quiz1.png)
1. [Color Selection](https://www.youtube.com/watch?v=bNOWJ9wdmhk)
	* [Quiz](images/quiz2.png)
1. Color Selection Code Example
```
	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg
	import numpy as np
	# Read in the image and print out some stats
	image = mpimg.imread('test.jpg')
	print('This image is: ',type(image), 
         'with dimensions:', image.shape)

	# Grab the x and y size and make a copy of the image
	ysize = image.shape[0]
	xsize = image.shape[1]
	# Note: always make a copy rather than simply using "="
	color_select = np.copy(image)
	# Define our color selection criteria
	# Note: if you run this code, you'll find these are not sensible values!!
	# But you'll get a chance to play with them soon in a quiz
	red_threshold = 0
	green_threshold = 0
	blue_threshold = 0
	rgb_threshold = [red_threshold, green_threshold, blue_threshold]
	# Identify pixels below the threshold
	thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
	Color_select[thresholds] = [0,0,0]

	# Display the image                 
	plt.imshow(color_select)
	plt.show()
```
1. Color Selection
	> Eventually, I found that with red_threshold = green_threshold = blue_threshold = 200, I get a pretty good result, where I can clearly see the lane lines, but most everything else is blacked out.
1. [Region Masking](https://www.youtube.com/watch?v=ngN9Cr-QfiI)
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image), 
         'with dimensions:', image.shape)

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
region_select = np.copy(image)

# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz 
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(region_select)

# uncomment if plot does not display
# plt.show()
```
1. Color and Region Combined
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test.jpg')

# Grab the x and y sizes and make two copies of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection 
# overlaid on the original.
ysize = image.shape[0]
xsize = image.shape[1]
color_select= np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 0
green_threshold = 0
blue_threshold = 0
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest (Note: if you run this code, 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz ;)
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] = [0,0,0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display our two output images
plt.imshow(color_select)
plt.imshow(line_image)

# uncomment if plot does not display
# plt.show()
```
1. Quiz Color Region
	> red_threshold = green_threshold = blue_threshold = 200, I chose the vertices of my triangle to be left_bottom = [0, 539], right_bottom [900, 539], and apex = [475, 320]
1. Finding Lines of Any Color
	> As it happens, lane lines are not always the same color, and even lines of the same color under different lighting conditions (day, night, etc) may fail to be detected by our simple color selection.
1. [What is Computer Vision?](https://www.youtube.com/watch?v=wxQhfSdxjKU)
	* Free Course:[Intro to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810)
1. Canny Edge Detection [1](https://www.youtube.com/watch?time_continue=2&v=Av2GsgQWX8I) [2](https://www.youtube.com/watch?v=LQM--KPJjD0) 
	> Note! The standard location of the origin (x=0, y=0) for images is in the top left corner with y values increasing downward and x increasing to the right. This might seem weird at first, but if you think about an image as a matrix, it makes sense that the "00" element is in the upper left.
1. Canny to Detect Lane Lines
```
#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 1
high_threshold = 10
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
```
	>As far as a ratio of low_threshold to high_threshold, John Canny himself recommended a low to high ratio of 1:2 or 1:3.
	> The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, and reject pixels below the low_threshold. Next, pixels with values between the low_threshold and high_threshold will be included as long as they are connected to strong edges. The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else.
1. Quiz: Canny Edges
	> I chose a kernelSize of 5 for Gaussian smoothing, a lowThreshold of 50 and a highThreshold of 150.
1. [Hough Transform](https://www.youtube.com/watch?v=JFwj5UtKmPY) [sine Hough](https://www.youtube.com/watch?time_continue=3&v=XQf7FOhwOVk)https://www.youtube.com/watch?time_continue=3&v=XQf7FOhwOVk
	* [Quiz1](images/quiz14_1.png)
	* [Quiz2](images/quiz14_2.png)
	* [Quiz3](images/quiz14_3.png)
	* [Quiz4](images/quiz14_4.png)
	* [Quiz5](images/quiz14_5.png)
1. Quiz: Hough Transform
```
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255   

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(0, 0), (imshape[1], 0), (imshape[1],imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 1     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 5 #minimum number of pixels making up a line
max_line_gap = 1    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)

```
	> Here's how I did it: I went with a low_threshold of 50 and high_threshold of 150 for Canny edge detection.

	> For region selection, I defined vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)

	> I chose parameters for my Hough space grid to be a rho of 2 pixels and theta of 1 degree (pi/180 radians). I chose a threshold of 15, meaning at least 15 points in image space need to be associated with each line segment. I imposed a min_line_length of 40 pixels, and max_line_gap of 20 pixels