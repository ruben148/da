import matplotlib.pyplot as plt
import itertools
import numpy as np
import cv2
import os

def display_color_classes(colors, class_names, num_classes=7):
    plt.figure(figsize=(10, 2))
    for i in range(len(colors)):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow([[colors[i]]])
        plt.title(class_names[i], fontsize=8)
        plt.axis('off')
    plt.show()

def overlay_masks_on_image(image, one_hot_masks, colors, alpha=0.3):
    img = image.copy()
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)

    for c in range(one_hot_masks.shape[-1]):
        mask = one_hot_masks[..., c]
        color = np.array(colors[c]) / 255

        mask_indices = mask.astype(bool)

        img[mask_indices] = (1 - alpha) * img[mask_indices] + alpha * color * 255

    return img

def plot_images_with_overlaid_masks(images, one_hot_masks, colors, num_rows=4, images_per_row=4):
    total_images = min(len(images), num_rows * images_per_row)
    plt.figure(figsize=(5 * images_per_row, 5 * num_rows))

    for i in range(total_images):
        plt.subplot(num_rows, images_per_row, i + 1)
        overlaid_image = overlay_masks_on_image(images[i], one_hot_masks[i], colors)
        plt.title(f'Image {i+1}')
        plt.imshow(overlaid_image.astype(np.uint8))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_images_and_masks(input_folder, images, masks, predicted_masks = None, load = False):

    if load:
        images = [cv2.cvtColor(cv2.imread(os.path.join(input_folder, image_file)), cv2.COLOR_BGR2RGB) for image_file in images]
        masks = [cv2.cvtColor(cv2.imread(os.path.join(input_folder, mask_file)), cv2.COLOR_BGR2RGB) for mask_file in masks]

    for i in range(len(images)):
        plt.subplot(1,(2 if predicted_masks is None else 3),1)
        plt.imshow(images[i])
        plt.axis('off')

        plt.subplot(1,(2 if predicted_masks is None else 3),2)
        plt.imshow(masks[i])
        plt.axis('off')
        
        if predicted_masks is not None:
            plt.subplot(1,3,3)
            plt.imshow(predicted_masks[i])
            plt.axis('off')

        plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if cm.dtype == 'float' else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()