import os
import cv2
import torch
import argparse
import numpy as np
from gfpgan import GFPGANer
from ultralytics import YOLO

'''
export BASICSR_JIT='True'
'''


def process_image_person(image_name, in_dir, out_dir, out_stack_dir, gfpgan_model, yolo_model, gt_dir, class_id):

    original_image_path = os.path.join(in_dir, image_name)
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise ValueError(f"Error loading original image: {original_image_path}")

    gt_image_path = os.path.join(gt_dir, image_name)
    gt_image = cv2.imread(gt_image_path)
    if gt_image is None:
        raise ValueError(f"Error loading gt image: {gt_image_path}")

    base_image = original_image.copy()

    # Detect glasses on face
    results = yolo_model.predict(original_image_path, imgsz=1024, conf=0.35, classes=[class_id])
    detected_boxes = []
    for result in results:
        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
        # print(boxes_xyxy)
        detected_boxes.extend(boxes_xyxy)

    if len(detected_boxes) == 0:
        original_image = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_CUBIC)
        original_image = gfpgan_model.enhance_part(original_image)
        h, w = base_image.shape[:2]
        original_image = cv2.resize(original_image, (w,h), interpolation=cv2.INTER_CUBIC)

    for box in detected_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        glasses_region = original_image[y1:y2, x1:x2]

        ### TODO - instead of glass region resizing to 512x512, crop glass region and padding horizontally or vertically to make it square then resize to 512 x 512
        glasses_region_resized = cv2.resize(glasses_region, (512, 512), interpolation=cv2.INTER_CUBIC)

        retouched_glasses_region = gfpgan_model.enhance_part(glasses_region_resized)
        retouched_glasses_region_resized = cv2.resize(retouched_glasses_region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
        original_image[y1:y2, x1:x2] = retouched_glasses_region_resized

    if len(out_dir) > 5:
        cv2.imwrite(os.path.join(out_dir, image_name), original_image)

    # Stack original and retouched images side by side
    if len(out_stack_dir) > 5:
        cv2.imwrite(os.path.join(out_stack_dir, image_name), np.hstack((base_image, original_image, gt_image)))


def main(in_dir, out_dir, out_stack_dir, retouch_model_path, yolo_model_path, gt_dir, class_id):

    if len(out_dir) > 5:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    if len(out_stack_dir) > 5:
        if not os.path.exists(out_stack_dir):
            os.makedirs(out_stack_dir)

    original_images = sorted(os.listdir(in_dir))

    gfpgan_model = GFPGANer(
        model_path=retouch_model_path,
        arch="original",
        channel_multiplier=1,
        bg_upsampler=None)

    yolo_model = YOLO(yolo_model_path)

    for image_name in original_images:
        try:
            process_image_person(image_name, in_dir, out_dir, out_stack_dir, gfpgan_model, yolo_model, gt_dir, class_id)
        except Exception as e:
            print("Image Failed: ", image_name)
            print("Error: ", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glass glare removal.")
    parser.add_argument("-id", "--in-dir", type=str, default="/home/humayunirshad/Desktop/glare_removal/trainset_2/or", help="Path to the directory of original images.")
    parser.add_argument("-od", "--out-dir", type=str, default="", help="Path to save the glare removed images images.")
    parser.add_argument("-osd", "--out-stack-dir", type=str, default="/home/humayunirshad/Desktop/glare_removal/___results/GFPGAN/trainset", help="Path to save the stack retouched images.")
    parser.add_argument("-mp", "--retouch-model-path", type=str, default="/data/pd/saved_models_danial/gfpgan/20240814_glare/net_g_740000.pth", help="Path to the retouch model.")
    parser.add_argument("-yp", "--yolo-model-path", type=str, default="/data/pd/saved_models_danial/Yolo/8_July_Glasses_teeth_Detector/best.pt")
    parser.add_argument("-gd", "--gt_dir", type=str, default="/home/humayunirshad/Desktop/glare_removal/trainset_2/gt", help="Path to the directory of ground truth images.")
    parser.add_argument("-c", "--class_id", default=0, type=int, help="Class ID for Yolo model object[0: glass, 1:teeth, 2:braces]")
    args = parser.parse_args()

    main(args.in_dir, args.out_dir, args.out_stack_dir, args.retouch_model_path, args.yolo_model_path, args.gt_dir, class_id=args.class_id)

