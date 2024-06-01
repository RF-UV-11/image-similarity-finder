
# 🖼️ Image Similarity Finder

This project is designed to find the most **similar image** to a given **input image** from a directory containing multiple subdirectories of images. It uses the **VGG16 model** pre-trained on **ImageNet** to extract features from the images and calculates similarity using **cosine similarity**. The most similar image is then plotted alongside the input image for visual comparison.

## 🌟 Features

- 📥 **Loads and preprocesses images** for the **VGG16 model**.
- 🧠 **Extracts high-level features** using the **VGG16 model**.
- 📊 **Computes cosine similarity** between the input image and images in the specified directory.
- 🔍 **Identifies and returns the most similar image**.
- 🖼️ **Plots the input image and the most similar image side by side**.

## 🛠️ Requirements

- ![Python](https://img.shields.io/badge/Python-3.x-blue.svg?logo=python&logoColor=white) **Python 3.x**
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white) **TensorFlow**
- ![Keras](https://img.shields.io/badge/Keras-2.x-red.svg?logo=keras&logoColor=white) **Keras**
- ![NumPy](https://img.shields.io/badge/NumPy-1.x-blue.svg?logo=numpy&logoColor=white) **NumPy**
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.x-blue.svg?logo=scikit-learn&logoColor=white) **scikit-learn**
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blue.svg?logo=matplotlib&logoColor=white) **Matplotlib**

## 📦 Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/image-similarity-finder.git
    cd image-similarity-finder
    ```

2. **Install the required Python packages**:
    ```sh
    pip install tensorflow keras numpy scikit-learn matplotlib
    ```

## 📂 Data Preparation

1. **Download the dataset** from [Google Drive](https://drive.google.com/file/d/1UVSJ6h7r8pmOWYZkqWeAZ_YvwbFr1wV3/view) and extract it to your local machine.

2. **Organize the dataset**:
    - Place your **input image** in a directory, e.g., `input/`.
    - Ensure the directory structure for images to be compared is in place, e.g., `check/` with subdirectories containing images.

3. **Example directory structure**:
    ```
    image-similarity-finder/
    ├── input/
    │   └── test.png
    └── check/
        ├── subdir1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── subdir2/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
    ```

## 🚀 Usage

1. **Edit the script with your paths**:
    - Set the `input_image_path` and `base_directory` variables to your specific paths in the `find_and_plot_most_similar_image` function call.

2. **Run the script**:
    ```sh
    python image_similarity_finder.py
    ```

## 📝 Example

Here's an example of how to use the script. Assume you have an input image at `input/test.png` and a directory of images to check at `check/`:

```python
# Example input image path
input_image_path = 'input/test.png'
base_directory = 'check'

find_and_plot_most_similar_image(input_image_path, base_directory)
```

After running the script, you should see an output indicating the most similar image and a plot displaying both the input image and the matched image.

![Output](https://github.com/RF-UV-11/image-similarity-finder/blob/main/output/output.png)
## 📜 Code Explanation

### 🔄 `load_and_preprocess_image(img_path, target_size=(224, 224))`

Loads and preprocesses an image for the **VGG16 model**, resizing it to the target size and applying necessary preprocessing steps.

### 🧬 `extract_features(img_array, model)`

Extracts features from an image array using the **VGG16 model**.

### 📊 `plot_images(input_image_path, matched_image_path, similarity)`

Plots the input image and the matched image side by side, displaying their similarity score.

### 🏋️‍♂️ `load_model()`

Loads the pre-trained **VGG16 model** with higher-level layers.

### 🔍 `find_and_plot_most_similar_image(input_image_path, base_directory)`

Main function to find and plot the most similar image to the input image. It loads and preprocesses the input image, traverses the directory to extract features from other images, computes similarities, and identifies the most similar image.

## 📑 Logging

The script uses Python's `logging` module to provide informational messages and error handling. This helps in understanding the process flow and troubleshooting issues.

## License

[MIT](https://choosealicense.com/licenses/mit/)
