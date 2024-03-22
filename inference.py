from ultralytics import YOLO
from PIL import Image
import cv2
import os
# import pgmagick as pg
import rembg
import colorthief
from colorthief import ColorThief
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm


def plot_color(color, label):
    plt.figure(figsize=(2, 2))
    plt.imshow([[color]])
    plt.title(label)
    plt.axis('off')
    plt.show()


def plot_palette(palette):
    plt.figure(figsize=(len(palette) * 2, 2))
    for i, color in enumerate(palette):
        plt.subplot(1, len(palette), i+1)
        plt.imshow([[color]])
        plt.title(f"Color {i+1}")
        plt.axis('off')
    plt.show()


def classify_color(color):
    if color is None:
        return "Unknown"

    # Should be adjusted!!!
    white_threshold = 150
    black_threshold = 100

    r, g, b = color

    luminance = (0.299 * r + 0.587 * g + 0.114 * b)

    if luminance >= white_threshold:
        return "White"
    elif luminance <= black_threshold:
        return "Black"
    else:
        return "Other"


def writing_the_number_of_guinea_pigs(image, number):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    text = f"Number of Guinea Pigs: {number}"
    size = cv2.getTextSize(text, font, scale, thickness)
    text_width = size[0][0]
    text_height = size[0][1]
    text_x = image.shape[1] - text_width - 10
    text_y = text_height + 10
    cv2.putText(image, text, (text_x, text_y), font, scale, (226, 135, 67), thickness)


def get_dominant_color(image):
    try:
        color_thief = ColorThief(image)
        dominant_color = color_thief.get_color(quality=1)

        return dominant_color

    except Exception as e:
        print(f"Error getting dominant color from image: {e}")
        return None


def inference(model, image_path, destination_dir):
    image_name = os.path.basename(image_path)

    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)

    results = model.predict(source=image_path, conf=0.5)

    boxes = results[0].boxes.xyxy.cpu().tolist()
    number_of_guinea_pigs = len(boxes)
    print(number_of_guinea_pigs)

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_original = image.copy()

    writing_the_number_of_guinea_pigs(image_original, number_of_guinea_pigs)

    if boxes is not None:
        for box in boxes:
            crop_obj = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            output_array = rembg.remove(crop_obj)
            output_image = Image.fromarray(output_array)

            temp_file_path = "temp_image.png"
            output_image.save(temp_file_path)

            dominant_color = get_dominant_color(temp_file_path)
            # plot_color(dominant_color, "Dominant color")

            color_of_the_guinea_pig = classify_color(dominant_color)

            cv2.putText(image_original, color_of_the_guinea_pig, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (226, 135, 67), 2)

            cv2.rectangle(image_original, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (37, 150, 190), 2)

            # palette = color_thief.get_palette(color_count=6)
            # plot_palette(palette)

    cv2.imwrite(os.path.join(destination_dir, str(image_name)), image_original)

if __name__ == "__main__":

    model = YOLO('trained_models/YOLOv8_larger_own_dataset.pt')

    # Single file
    # image_path = './test images/lujzika.jpg'
    # inference(model, image_path)

    destination_directory = "analyzed_test_images"
    test_dir = 'data/test/images'
    image_pattern = os.path.join(test_dir, "*.jpg")
    image_paths = glob.glob(image_pattern)
    print(len(image_paths))

    for image_path in tqdm(image_paths):
        inference(model, image_path, destination_directory)



