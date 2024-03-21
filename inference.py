from ultralytics import YOLO
from PIL import Image
import cv2
import os
import pgmagick as pg
import rembg
import colorthief
from colorthief import ColorThief
import matplotlib.pyplot as plt


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


model = YOLO('trained_models/YOLOv8_larger_own_dataset.pt')
image_path = './test images/received_419200997452539.jpeg'

crop_dir_name = "cropped_images"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

results = model.predict(source=image_path, conf=0.5)

boxes = results[0].boxes.xyxy.cpu().tolist()
number_of_guinea_pigs = len(boxes)
print(number_of_guinea_pigs)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if boxes is not None:
    for box in boxes:
        crop_obj = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        # Apply background removal using rembg
        output_array = rembg.remove(crop_obj)

        # Create a PIL Image from the output array
        output_image = Image.fromarray(output_array)

        # # Save the output image
        # output_image.save('output_image.png')

        average_color = cv2.mean(crop_obj)[:3]

        cv2.imwrite(os.path.join(crop_dir_name, str(average_color) + ".png"), output_array)

color_thief = ColorThief('cropped_images/(46.82660792943524, 36.43322378493874, 27.365472736281692).png')
dominant_color = color_thief.get_color(quality=1)
plot_color(dominant_color, "Dominant color")

palette = color_thief.get_palette(color_count=6)
plot_palette(palette)
