# Celebrity Look-Alike App

## Overview
The Celebrity Look-Alike App allows users to upload an image and find their celebrity look-alikes using a machine learning model. The app utilizes the ResNet50 model to extract features from images and compare them with a predefined set of celebrity images.

## Features
- **Image Upload**: Users can upload their own images.
- **Image Preview**: Users can preview the uploaded image before submission.
- **Celebrity Matching**: The app finds and displays the top 5 celebrity look-alikes based on similarity.
- **Social Sharing**: Users can share their results on Facebook and Twitter.

## Technologies Used
- **Flask**: A lightweight WSGI web application framework for Python.
- **TensorFlow**: For implementing the ResNet50 model for image processing.
- **HTML/CSS**: For front-end development.

## Installation

### Prerequisites
- Python 3.x
- Flask
- TensorFlow
- NumPy

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd celebrity_app
### Install the required packages:
```bash
pip install Flask tensorflow numpy
