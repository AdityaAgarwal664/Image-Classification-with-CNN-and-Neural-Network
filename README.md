# Image Classification with CNN to identify Objects [An A<sup>2</sup> Product]
# Image Classifier

This project is an image classification application built using Flask and PyTorch. It allows users to upload an image and get a prediction of the object in the image. The classifier is trained on the CIFAR-10 dataset and can identify the following classes: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck.

## Features

- Upload an image and get a prediction for its class.
- The model is trained using a Convolutional Neural Network (CNN) with three convolutional layers and two fully connected layers.
- The application provides a user-friendly web interface for uploading images and displaying results.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/image-classifier.git
    cd image-classifier
    ```

2. Create a virtual environment (optional but recommended):

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Download the CIFAR-10 dataset (if not already downloaded) and train the model:

    ```sh
    python main.py
    ```

5. Start the Flask application:

    ```sh
    python app.py
    ```

6. Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage

1. Open the web application.
2. Upload an image by clicking the "Choose File" button.
3. Click the "SUBMIT" button.
4. See the prediction result displayed on the page along with the uploaded image.

## Directory Structure

```
image-classifier/
│
├── app.py                  # Flask application file
├── main.py                 # Model training and evaluation script
├── image_classification_model.pth  # Saved model weights
├── requirements.txt        # Required packages
├── static/
│   ├── styles.css          # CSS for the web application
│   └── uploads/            # Directory for uploaded images
├── templates/
│   └── index.html          # HTML template for the web application
├── README.md               # Project documentation
└── data/                   # Directory for the CIFAR-10 dataset
```

## Model Details

The model is a Convolutional Neural Network (CNN) with the following architecture:

- Three convolutional layers with ReLU activations and max pooling.
- Two fully connected layers with ReLU activations.
- Dropout layer for regularization.
- The model is trained using the Adam optimizer and cross-entropy loss.

## Acknowledgements

- The CIFAR-10 dataset used in this project is publicly available and provided by the Canadian Institute For Advanced Research.
- The project is built using [PyTorch](https://pytorch.org/) for the deep learning model and [Flask](https://flask.palletsprojects.com/en/2.0.x/) for the web application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
