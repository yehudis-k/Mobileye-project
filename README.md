# Mobileye (~120h challenge) :
Detecting traffic lights and the distance to them on runtime within given video using image
processing and machine learning, in a long project separated into 4 parts using Python:
1. Detection of source lights in an image using convolution with customized high- and lowpass filters.
2. Generating and training CNN using the products of the previous stage as input, to
conclude all the traffic lights in the image (using tensorflow).
3. Estimating the distance to each detected traffic light from the camera picturing the
images of interest, involving geometric and linear algebra calculations.
4. Integrating all previous parts into a functional and intuitive SW product
