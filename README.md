# Algorithm-designed-for-detecting-and-matching-keypoints-between-two-images

Library Imports: 

Code imports necessary libraries and modules such as argparse for command-line argument parsing, colorsys for color system conversions, numpy for numerical operations, cv2 for OpenCV functions, PIL.Image for image loading and saving, and copy for object copying. It also imports math, numpy, and the Enum class for various utility functions and data structures.

Constants Definition: The script defines several constants related to the image processing algorithms, such as parameters for Gaussian blur, keypoint detection thresholds, and descriptor generation parameters.

Class Definitions: 

- Interpolacja: An enumeration for interpolation methods.

- Obraz: A class representing an image with methods for loading, displaying, copying, saving, and manipulating pixels and pixel coordinates.

- PunktKluczowy: Represents a keypoint in an image.

- PiramidaSkal: Manages a pyramid of images scaled for multi-resolution analysis.

- Image Processing Functions: These include functions for converting RGB images to grayscale, Gaussian blurring, generating scale-space pyramids, detecting keypoints, computing keypoint descriptors, matching keypoints between two images, and drawing keypoints and matches on images.

- Utility Functions: Functions for color conversion, Euclidean distance computation, and drawing matches between images using color coding.

Main Script Flow: 

The script uses argparse to parse command-line arguments for two image paths. It then proceeds to load these images, convert them to grayscale (if necessary), detect keypoints and their descriptors, match keypoints between the two images based on descriptor similarity, and draw these matches on a combined image. Finally, the resulting image is saved.

This script encapsulates a complex process of feature detection, description, and matching, commonly used in computer vision applications such as image stitching, object recognition, and motion tracking. It specifically seems to implement algorithms similar to the Scale-Invariant Feature Transform (SIFT) or its variants for keypoint detection and matching.

Example results are shown below:

![rezultat](https://github.com/kacdro/Algorithm-designed-for-detecting-and-matching-keypoints-between-two-images/assets/100469610/8eb11e82-eb3d-4ec0-b531-d5580d39bab9)


![Bodo](https://github.com/kacdro/Algorithm-designed-for-detecting-and-matching-keypoints-between-two-images/assets/100469610/0e2e402a-a063-4a42-98bd-484a6df41fd3)

![Budapest](https://github.com/kacdro/Algorithm-designed-for-detecting-and-matching-keypoints-between-two-images/assets/100469610/5bc60040-561d-4d6a-87c8-de825c50594f)


![rezultat](https://github.com/kacdro/Algorithm-designed-for-detecting-and-matching-keypoints-between-two-images/assets/100469610/d7e679bb-271c-4441-b861-d0a6255d352d)

