import os
import cv2
import dlib
import argparse
import numpy as np
from gfpgan import GFPGANer
from ultralytics import YOLO


def get_face_parts(model, faced, class_id):
    results = model.predict(faced, imgsz=512, conf=0.10, classes=[class_id])
    detected_boxes = []
    for result in results:
        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
        if not boxes_xyxy:
            return None
        detected_boxes.extend(boxes_xyxy)
    return detected_boxes


def dlib_face(img, detector, padding=0):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    cropped_faces = []
    faces_cords = []
    for det in dets:
        x1, y1, w, h = det.left(), det.top(), det.width(), det.height()
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img.shape[1], x1 + w + padding)
        y2_padded = min(img.shape[0], y1 + h + padding)
        crp_fce = img[y1_padded:y2_padded, x1_padded:x2_padded]
        cropped_faces.append(crp_fce)
        faces_cords.append((x1_padded, y1_padded, x2_padded, y2_padded))
    return cropped_faces, faces_cords


def load_gfpgan(model_path):
    restorer = GFPGANer(
        model_path=model_path,
        arch="original",
        channel_multiplier=1,
        bg_upsampler=None)
    return restorer


def square_cropped_gtb(image_array, bbox, target_size=512):
    left, top, right, bottom = map(int, bbox[:4])
    bbox_width, bbox_height = right - left, bottom - top
    padding_top = padding_bottom = 0
    if bbox_width > bbox_height:
        padding_needed = bbox_width - bbox_height
        padding_top = padding_needed // 2
        padding_bottom = padding_needed - padding_top
    new_top = top - padding_top
    new_bottom = bottom + padding_bottom
    cropped_image = image_array[max(new_top, 0):min(new_bottom, image_array.shape[0]), left:right]
    pad_top = abs(min(new_top, 0))
    pad_bottom = abs(max(new_bottom - image_array.shape[0], 0))
    if pad_top > 0 or pad_bottom > 0:
        cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
    cropped_width = cropped_image.shape[1]
    cropped_height = cropped_image.shape[0]
    resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return {
        'resized_image': resized_image,
        'padding_top': padding_top,
        'padding_bottom': padding_bottom,
        'cropped_width': cropped_width,
        'cropped_height': cropped_height
    }


def restore_cropped_gtb(resized_image, original_image, bbox, padding_info):
    padding_top = padding_info['padding_top']
    padding_bottom = padding_info['padding_bottom']
    cropped_width = padding_info['cropped_width']
    cropped_height = padding_info['cropped_height']
    resized_padded_image = cv2.resize(resized_image, (cropped_width, cropped_height), interpolation=cv2.INTER_LANCZOS4)
    if padding_top > 0:
        resized_padded_image = resized_padded_image[padding_top:, :]
    if padding_bottom > 0:
        resized_padded_image = resized_padded_image[:-padding_bottom, :]
    left, top, right, bottom = map(int, bbox[:4])
    original_image[top:top + resized_padded_image.shape[0], left:left \
                        + resized_padded_image.shape[1]] = resized_padded_image
    return original_image


def perform_teeth_whitening(args):
    img = cv2.imread(args.img_path)
    teeth_whitener = load_gfpgan(args.teeth_whitening)
    gtb_detector = YOLO(args.gtb_model)
    face_detector = dlib.get_frontal_face_detector()
    faces, faces_cords = dlib_face(img, face_detector, padding=0)
    if faces is not None:
        for i, face in enumerate(faces):
            # teeth whitening
            mouth_bbox = get_face_parts(gtb_detector, face, class_id=1)
            print(mouth_bbox)
            if mouth_bbox is not None:
                padding_info = square_cropped_gtb(face, mouth_bbox[0])
                white_teeth = teeth_whitener.enhance_part(padding_info['resized_image'])
                restored_white_teeth_face = restore_cropped_gtb(white_teeth, face.copy(), mouth_bbox[0], padding_info)
                face = restored_white_teeth_face.copy()
            else:
                print("Teeth not detected")
            # paste back face
            fx1, fy1, fx2, fy2 = faces_cords[i]
            img[fy1:fy2, fx1:fx2] = face
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teeth Whitening")

    parser.add_argument("-gtb","--gtb-model", type=str, default="/data/trained_models/yolo/yolov8x_face_parts_20240708.pt", help="Path to the  glasses, teeth and braces detector model")
    parser.add_argument("-tw","--teeth-whitening", type=str, default="/data/trained_models/gfpgan/20240831_teeth/net_g_760000.pth", help="Path to the teeth whitening model")

    parser.add_argument("-imp","--img-path", type=str, default="3.png", help="Path to the directory of original images")
    args = parser.parse_args()
    img = perform_teeth_whitening(args)
    cv2.imwrite("test_tw.png", img)

