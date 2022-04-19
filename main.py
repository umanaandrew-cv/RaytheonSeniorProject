import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import time
import pixellib
from pixellib.instance import instance_segmentation

#function that calls the camera
def test_with_cam():
    # reading the input using the camera
    result, test_image = cam.read()

    # If image will detected without any error,
    # show result
    if result:
        # output
        # cv2.imshow("Recognition", test_image)

        # saving image in local storage
        cv2.imwrite("Webcam.jpg", test_image)

        # If keyboard interrupt occurs, destroy image
        # window
        # cv2.waitKey(0)
        # cv2.destroyWindow("GeeksForGeeks")

    # Load image
    test_image = "Webcam.jpg"
    return test_image



#keeping track of iterations
i=0
#Capturing images from the webcam
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

while 1:
    test_image="test3_usf.jpg"
    #test_image = test_with_cam()


    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Opening image that was taken with camera
    image = Image.open(test_image)

    #resize the image to a 224x224 that was used in the model.h5 file
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    #image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)

    value = prediction[0][0]
    largest_index=0

    for x in range(len(prediction[0])):
        if(prediction[0][x]>value):
            value=prediction[0][x]
            largest_index=x

    if(value<.75):
        print("Prediction is too inaccurate to be certain")
    elif(largest_index==0):
        print("Prediction is No logo")
        if(value<.9):
            print("Not super confident")
    elif(largest_index==1):
        print("Prediction is USF logo")
        if (value < .9):
            print("Not super confident")

    i=i+1
    print(str(i)+" iteration")

    time.sleep(5)

print("Finished")
