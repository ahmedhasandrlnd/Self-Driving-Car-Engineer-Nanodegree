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