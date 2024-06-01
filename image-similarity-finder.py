# import cv2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics.pairwise import cosine_similarity

# # Load and preprocess an image
# def load_and_preprocess_image(img_path, target_size=(224, 224)):
#     img = image.load_img(img_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)

# # Extract features using VGG16
# def extract_features(img_array, model):
#     features = model.predict(img_array)
#     return features.flatten()

# # Plotting function
# def plot_images(input_image_path, similar_images):
#     plt.figure(figsize=(20, 10))

#     # Plot input image
#     ax = plt.subplot(3, 4, 1)
#     input_img = image.load_img(input_image_path, target_size=(224, 224))
#     plt.imshow(input_img)
#     plt.title("Input Image")
#     plt.axis("off")

#     # Plot similar images
#     for i, (img_path, similarity) in enumerate(similar_images):
#         ax = plt.subplot(3, 4, i + 2)
#         img = image.load_img(img_path, target_size=(224, 224))
#         plt.imshow(img)
#         plt.title(f"Sim: {similarity:.2f}")
#         plt.axis("off")

#     plt.show()

# # Load the pre-trained VGG16 model + higher level layers
# base_model = VGG16(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# # Example input image path
# input_image_path = 'input/test1.jpg'
# input_image = load_and_preprocess_image(input_image_path)
# input_image_features = extract_features(input_image, model)

# # Directory containing subdirectories with images
# base_directory = 'check'

# # Dictionary to hold image paths and their corresponding features
# image_features = {}

# # Traverse the directory structure
# for root, _, files in os.walk(base_directory):
#     for file in files:
#         try:
#             if file.lower().endswith(('png', 'jpg', 'jpeg')):
#                 img_path = os.path.join(root, file)
#                 img_array = load_and_preprocess_image(img_path)
#                 features = extract_features(img_array, model)
#                 image_features[img_path] = features
#             else:
#                 continue

#         except:
#             img_path = os.path.join(root, file)
#             print(img_path)

# # Compute similarities
# similarities = {}
# for img_path, features in image_features.items():
#     similarity = cosine_similarity([input_image_features], [features])[0][0]
#     similarities[img_path] = similarity

# # Get top 10 most similar images
# top_10_similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]

# # Print the results
# for img_path, similarity in top_10_similar_images:
#     print(f"Image: {img_path}, Similarity: {similarity}")

# # Plot the input image and the top 10 most similar images
# plot_images(input_image_path, top_10_similar_images)



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess an image
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Extract features using VGG16
def extract_features(img_array, model):
    features = model.predict(img_array)
    return features.flatten()

# Plotting function
def plot_images(input_image_path, matched_image_path, similarity):
    plt.figure(figsize=(10, 5))

    # Plot input image
    ax = plt.subplot(1, 2, 1)
    input_img = image.load_img(input_image_path, target_size=(224, 224))
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.axis("off")

    # Plot matched image
    ax = plt.subplot(1, 2, 2)
    matched_img = image.load_img(matched_image_path, target_size=(224, 224))
    plt.imshow(matched_img)
    plt.title(f"Matched Image\nSimilarity: {similarity:.2f}")
    plt.axis("off")

    plt.show()

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Example input image path
input_image_path = 'input/test.png'
input_image = load_and_preprocess_image(input_image_path)
input_image_features = extract_features(input_image, model)

# Directory containing subdirectories with images
base_directory = 'check'

# Dictionary to hold image paths and their corresponding features
image_features = {}

# Traverse the directory structure
for root, _, files in os.walk(base_directory):
    for file in files:
        try:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img_array = load_and_preprocess_image(img_path)
                features = extract_features(img_array, model)
                image_features[img_path] = features
            else:
                continue
        except:
            img_path = os.path.join(root, file)
            print(f"Error processing image: {img_path}")

# Compute similarities
similarities = {}
for img_path, features in image_features.items():
    similarity = cosine_similarity([input_image_features], [features])[0][0]
    similarities[img_path] = similarity

# Get the most similar image
most_similar_image_path, highest_similarity = max(similarities.items(), key=lambda x: x[1])

# Print the result
print(f"Most similar image: {most_similar_image_path}, Similarity: {highest_similarity}")

# Plot the input image and the most similar image
plot_images(input_image_path, most_similar_image_path, highest_similarity)
