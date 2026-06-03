# Hand Gesture PPT Controller Using CNN

## Project Overview

I developed a Hand Gesture PPT Controller that allows users to control PowerPoint presentations using hand gestures captured through a webcam. The main goal of this project was to create a touchless presentation control system that provides a more interactive and convenient way of delivering presentations.

The system uses MediaPipe for hand detection and a Convolutional Neural Network (CNN) for gesture classification. Once a gesture is recognized, it is mapped to a PowerPoint action such as starting the presentation, ending the presentation, navigating slides, or controlling the pointer.

## Motivation

Traditional presentation control relies on keyboards, mice, or presentation remotes. I wanted to explore how computer vision and deep learning could be used to create a more natural human-computer interaction system. This project demonstrates how hand gestures can be used as an alternative input method for controlling presentations.

## Features

* Real-time hand gesture recognition
* Touchless PowerPoint control
* Start and end slideshow using gestures
* Navigate between slides
* Pointer control mode
* Stable prediction mechanism to reduce false detections
* CNN-based gesture classification

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy
* PyAutoGUI

## Dataset

For training the model, I used the HaGRID (Hand Gesture Recognition Image Dataset). The dataset contains images of different hand gestures collected under various lighting conditions and backgrounds.

The gestures used in this project are:

* Like
* Dislike
* Palm
* Fist
* One Finger
* Two Finger
* OK

The dataset was preprocessed, split into training and validation sets, and then used to train the CNN model.

## Model Architecture

The gesture classification model is based on a Convolutional Neural Network (CNN). The network consists of multiple convolutional layers, batch normalization layers, pooling layers, and fully connected layers. Data augmentation techniques were also applied during training to improve the model's generalization ability.

## Project Workflow

1. Capture hand images using a webcam.
2. Detect the hand region using MediaPipe.
3. Crop and preprocess the hand image.
4. Pass the image to the trained CNN model.
5. Predict the gesture class.
6. Map the gesture to a PowerPoint action.
7. Execute the action in real time.

## Results

The trained CNN model achieved an overall test accuracy of 91.92%.

The system was successfully tested in real-time and was able to perform presentation control actions with minimal delay. A stability mechanism was implemented to ensure actions are triggered only after consistent predictions across multiple frames.

## Future Enhancements

* Support for dynamic gestures
* Voice command integration
* Improved performance under low-light conditions
* Mobile and embedded system deployment
* Integration with other applications beyond PowerPoint

## Conclusion

This project successfully demonstrates the use of computer vision and deep learning for touchless presentation control. By combining MediaPipe and CNN-based gesture recognition, the system provides an intuitive and practical way to interact with PowerPoint presentations without using traditional input devices.
