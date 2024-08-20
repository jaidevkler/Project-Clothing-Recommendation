import cv2
import pandas as pd
import numpy as np

from PIL import Image

def get_class_bounding_boxes(mask, num_classes):
    """
    Find bounding boxes for each class in the segmentation mask.
    
    Args:
    mask: np.array, shape (height, width), contains class labels for each pixel.
    num_classes: int, number of classes.
    
    Returns:
    bounding_boxes: dict, where keys are class indices and values are bounding boxes (x_min, y_min, x_max, y_max).
    """
    bounding_boxes = {}
    
    for class_idx in range(num_classes):
        # Find all pixels belonging to the current class
        class_mask = (mask == class_idx).astype(np.uint8)

        # Find contours for the class mask
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the bounding box around the largest contour
            x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x + w), max(y_max, y + h)
            
            bounding_boxes[class_idx] = (x_min, y_min, x_max, y_max)
    # Return bounding boxes
    return bounding_boxes

def import_lables():
    # Import labels from csv file
    labels = pd.read_csv('Resources/labels/labels.csv')
    # Drop unamed column
    labels.drop('Unnamed: 0', axis=1, inplace=True)
    # Return labels dataframe
    return labels

def create_bounding_images(bounding_boxes, labels, image):
    images=[]
    pixels = []
    # Loop through bounding boxes, crop images and save them to disk
    for class_idx, box in bounding_boxes.items():
        class_name = labels.iloc[class_idx]['label_list']
        temp_img = image.crop(box)
        images.append(temp_img)
        pixels.append(temp_img.size[0]*temp_img.size[1])
    # Return
    return images, pixels

def get_middle_index(df):
    if len(df)%2 == 0:
        return int(len(df)/2)-1
    else:
        return int((len(df) - 1)/2)

def get_bounding_images(predicted_mask, image):
    # Number of classes
    num_classes = predicted_mask.max() + 1  
    # Create bounding boxes
    bounding_boxes = get_class_bounding_boxes(predicted_mask, num_classes)
    # Import labels
    labels = import_lables()
    # Get label names from bounding_boxes keys
    image_labels = labels.iloc[list(bounding_boxes.keys())].reset_index(drop=True)
    # Create bounding boxes for all the classes
    images, pixels = create_bounding_images(bounding_boxes, labels, image)
    # Add pixels column to the dataframe
    image_labels['pixels'] = pixels
    # Sort values
    image_labels = image_labels.sort_values(by='pixels', ascending=False)
    # List of chosen categories to search for
    bounding_categories = ['upper', 'lower', 'shoes']
    # Empty list to hold the images
    bounding_images = []
    # Loop through the images and save them to the list
    for category in bounding_categories:
        # Filter labels based on the category
        filtered_labels = image_labels[(image_labels['category'] == category) & (image_labels['pixels'] > 3000)]
        # If labels exist for the category
        if len(filtered_labels) >= 1:
            # Set label as the middle index
            filtered_labels = filtered_labels.iloc[get_middle_index(filtered_labels)]
            # Save a copy of the image
            images[filtered_labels.name].save(f'Output/images/{category}.png')
            # Add image to list
            bounding_images.append(images[filtered_labels.name])
        else:
            Image.new('RGB', (10, 10)).save(f'Output/images/{category}.png')
    return bounding_categories
