import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
from collections import defaultdict

import numpy as np

def calculate_class_weights(mask_files, colors, input_folder, num_classes=7):
    class_counts = defaultdict(int)

    for mask_file in mask_files:
        mask = cv2.cvtColor(cv2.imread(os.path.join(input_folder, mask_file), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        
        for color_idx, color in enumerate(colors):
            class_counts[color_idx] += np.sum(np.all(mask == color, axis=-1))

    total_pixels = sum(class_counts.values())

    class_weights = {class_id: total_pixels/(num_classes * count) 
                     for class_id, count in class_counts.items()}

    return class_weights

def one_hot_to_rgb(one_hot_mask, colors):
    single_channel_mask = np.argmax(one_hot_mask, axis=-1)

    rgb_image = np.zeros((*single_channel_mask.shape, 3), dtype=np.uint8)

    for class_id, color in enumerate(colors):
        rgb_image[single_channel_mask == class_id] = color

    return rgb_image

def data_generator(input_folder, colors, image_files, mask_files, batch_size, num_classes, spatial_augmentation, color_augmentation):
    color_tolerance = 10  # Allow some variation in color

    while True:
        for i in range(0, len(image_files), batch_size):
            batch_images = []
            batch_masks = []

            for j in range(len(image_files[i:i+batch_size])):
                image = cv2.cvtColor(cv2.imread(os.path.join(input_folder, image_files[i+j])), cv2.COLOR_BGR2RGB)
                mask = cv2.cvtColor(cv2.imread(os.path.join(input_folder, mask_files[i+j])), cv2.COLOR_BGR2RGB)

                augmented = spatial_augmentation(image=image, mask=mask)
                image_aug, mask_aug = augmented['image'], augmented['mask']

                image_aug = color_augmentation(image=image_aug)['image']

                batch_images.append(image_aug)
                batch_masks.append(mask_aug)

            images = np.array(batch_images) / 255.0

            one_hot_masks = np.zeros((len(batch_masks), *mask_aug.shape[:2], num_classes), dtype=np.int_)
            for c in range(num_classes):
                color = np.array(colors[c])
                lower_bound = np.maximum(color - color_tolerance, 0)
                upper_bound = np.minimum(color + color_tolerance, 255)
                for k, mask in enumerate(batch_masks):
                    class_mask = np.all(np.logical_and(mask >= lower_bound, mask <= upper_bound), axis=-1)
                    one_hot_masks[k, ..., c] = class_mask.astype(np.int_)

            yield images, one_hot_masks

def test_generator(input_folder, image_files, batch_size, spatial_augmentation, color_augmentation):
    while True:
        for i in range(0, len(image_files), batch_size):
            batch_images = []

            for j in range(len(image_files[i:i+batch_size])):
                image = cv2.cvtColor(cv2.imread(os.path.join(input_folder, image_files[i+j])), cv2.COLOR_BGR2RGB)

                augmented = spatial_augmentation(image=image)
                image_aug = augmented['image']

                image_aug = color_augmentation(image=image_aug)['image']

                batch_images.append(image_aug)

            images = np.array(batch_images) / 255.0

            yield images