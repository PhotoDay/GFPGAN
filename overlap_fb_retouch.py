import os
import cv2
import torch
import argparse
import numpy as np
from gfpgan import GFPGANer
from ultralytics import YOLO
from basicsr.utils import imwrite
from skimage.util.shape import view_as_windows


def retouch(image_slice, model_path):
    restorer = GFPGANer(
        model_path=model_path,
        arch="original",
        channel_multiplier=1,
        bg_upsampler=None)
    retouch_image = restorer.enhance_part(image_slice)
    return retouch_image


def retouch_bbox(image, bbox, restorer, window_size=512, reverse_pixel=12, debug=False, debug_folder=""):
    x1, y1, x2, y2 = map(int, bbox)
    person_image = image[y1:y2, x1:x2]
    pad_height = (window_size - person_image.shape[0] % window_size) % window_size
    pad_width = (window_size - person_image.shape[1] % window_size) % window_size
    padded_person_image = np.pad(person_image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    num_windows_x = (padded_person_image.shape[1] - window_size) // (window_size - reverse_pixel) + 1
    num_windows_y = (padded_person_image.shape[0] - window_size) // (window_size - reverse_pixel) + 1
    for i in range(num_windows_y):
        for j in range(num_windows_x):
            if i == 0 and j == 0:
                start_y = 0
                start_x = 0
            else:
                start_y = i * (window_size - reverse_pixel)
                start_x = j * (window_size - reverse_pixel)
            person_window = padded_person_image[start_y:start_y + window_size, start_x:start_x + window_size, :]
            if debug:
                window_filename = f"{start_x}_{start_y}.jpg"
                window_path = os.path.join(debug_folder, window_filename)
                cv2.imwrite(window_path, person_window)        
            retouched_window = restorer.enhance_part(person_window)
            padded_person_image[start_y:start_y + window_size, start_x:start_x + window_size, :] = retouched_window
    retouched_bbox = padded_person_image[:person_image.shape[0], :person_image.shape[1], :]
    return retouched_bbox


def process_image_person(original_image_path, output_retouch_path, output_stacked_path, restorer, stacked=False, reverse_pixel=12, debug=False):
    yolo_model = YOLO("yolov8x.pt")
    original_image = cv2.imread(original_image_path)
    gt = original_image.copy()
    if original_image is None:
        raise ValueError(f"Error loading image: {original_image_path}")

    image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    image_debug_folder = ""
    if debug:
        debug_dir = "debug"
        image_debug_folder = os.path.join(debug_dir, image_name)
        os.makedirs(image_debug_folder, exist_ok=True)
    
    results = yolo_model.predict(original_image_path, imgsz=1024, conf=0.35, classes=[0])
    detected_boxes = []
    for result in results:
        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
        print(boxes_xyxy)
        detected_boxes.extend(boxes_xyxy)
    
    window_size = 512
    
    for box in detected_boxes:
        retouched_region = retouch_bbox(original_image, box, restorer, window_size, reverse_pixel, debug, image_debug_folder)
        x1, y1, x2, y2 = map(int, box)
        original_image[y1:y2, x1:x2] = retouched_region
    
    cv2.imwrite(output_retouch_path, original_image)
    
    if stacked:
        hstack_image = np.hstack((gt, original_image))
        cv2.imwrite(output_stacked_path, hstack_image)


def main(original_images_dir, output_retouch_dir, output_stack_dir, model_path, stacked=False, reverse_pixel=12, debug=False):
    if not os.path.exists(output_retouch_dir):
        os.makedirs(output_retouch_dir)
    if stacked:
        if not os.path.exists(output_stack_dir):
            os.makedirs(output_stack_dir)    
    original_images = sorted(os.listdir(original_images_dir))
    restorer = GFPGANer(
        model_path=model_path,
        arch="original",
        channel_multiplier=1,
        bg_upsampler=None)
    for original_image_name in original_images:
        original_image_path = os.path.join(original_images_dir, original_image_name)
        output_retouch_path = os.path.join(output_retouch_dir, original_image_name)
        output_stacked_path = ''
        if stacked:
            output_stacked_path = os.path.join(output_stack_dir, original_image_name)
        try:
            process_image_person(original_image_path, output_retouch_path, output_stacked_path, restorer, stacked=stacked, reverse_pixel=reverse_pixel, debug=debug)
        except Exception as e:
            print("Image Failed: ",original_image_name)
            print("Error: ", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retouch skin areas in detected persons.")
    parser.add_argument("--original_images_dir", type=str, default="july1_test", help="Path to the directory of original images.")
    parser.add_argument("--output_retouch_dir", type=str, default="full_body_retouched", help="Path to save the retouched images.")
    parser.add_argument("--output_stack_dir", type=str, default="full_body_retouched_stacked", help="Path to save the stack retouched images.")
    parser.add_argument("--retouch_model_path", type=str, default="experiments/pretrained_models/retouch_20231211.pth", help="Path to the retouch model.")
    parser.add_argument('--stacked', action='store_true', help='Stack the images')
    parser.add_argument('--reverse_pixel', type=int, default=12, help='Pixel offset for the sliding window.')
    parser.add_argument('--debug', action='store_true', help='Save sliding windows for debugging.')
    args = parser.parse_args()
    main(args.original_images_dir, args.output_retouch_dir, args.output_stack_dir, args.retouch_model_path, stacked=args.stacked, reverse_pixel=args.reverse_pixel, debug=args.debug)

