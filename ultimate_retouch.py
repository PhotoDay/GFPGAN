import os
import cv2
import dlib
import argparse
import numpy as np
from gfpgan import GFPGANer
from ultralytics import YOLO

"""
python ultimate_retouch.py  -stk -dbg
"""
def get_face_parts(model, faced, class_id):
    results = model.predict(faced, imgsz=512, conf=0.25, classes=[class_id])
    detected_boxes = []
    for result in results:
        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
        if not boxes_xyxy:
            return None
        detected_boxes.extend(boxes_xyxy)
    return detected_boxes


def divide_and_resize_face(face_img, output_size=(512, 512)):
    h, w, _ = face_img.shape
    half_h1, half_h2 = h // 2, h - h // 2
    half_w1, half_w2 = w // 2, w - w // 2
    parts = [
        face_img[:half_h1, :half_w1],
        face_img[:half_h1, half_w1:],
        face_img[half_h1:, :half_w1],
        face_img[half_h1:, half_w1:]
    ]
    resized_parts = [cv2.resize(part, output_size) for part in parts]
    return resized_parts, (half_h1, half_h2, half_w1, half_w2)


def reassemble_face_from_parts(parts, split_sizes):
    half_h1, half_h2, half_w1, half_w2 = split_sizes
    resized_parts = [
        cv2.resize(parts[0], (half_w1, half_h1)),
        cv2.resize(parts[1], (half_w2, half_h1)),
        cv2.resize(parts[2], (half_w1, half_h2)),
        cv2.resize(parts[3], (half_w2, half_h2))
    ]
    top_row = np.hstack((resized_parts[0], resized_parts[1]))
    bottom_row = np.hstack((resized_parts[2], resized_parts[3]))
    reassembled_face = np.vstack((top_row, bottom_row))
    return reassembled_face


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




def detect_face_yunet(img, detector, padding=0):
    img_width = int(img.shape[1])
    img_height = int(img.shape[0])
    detector.setInputSize((img_width, img_height))
    faces = detector.detect(img)
    if faces[1] is None:
        print("Cannot find a face")
        return None, None
    cropped_faces = []
    faces_cords = []
    #print("detect_face_yunet: faces = ", faces)
    for _, face in enumerate(faces[1]):
        x1, y1, w, h = map(int, face[0:4])
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(img.shape[1], x1 + w + padding)
        y2_padded = min(img.shape[0], y1 + h + padding)
        crp_fce = img[y1_padded:y2_padded, x1_padded:x2_padded]
        cropped_faces.append(crp_fce)
        faces_cords.append((x1_padded, y1_padded, x2_padded, y2_padded))
    return cropped_faces, faces_cords


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
    cropped_image = image_array[max(new_top, 0):min(new_bottom,
                               image_array.shape[0]), left:right]
    pad_top = abs(min(new_top, 0))
    pad_bottom = abs(max(new_bottom - image_array.shape[0], 0))
    if pad_top > 0 or pad_bottom > 0:
        cropped_image = np.pad(cropped_image, ((pad_top, pad_bottom),
                              (0, 0), (0, 0)), mode='constant',
                              constant_values=0)
    cropped_width = cropped_image.shape[1]
    cropped_height = cropped_image.shape[0]
    resized_image = cv2.resize(cropped_image, (target_size, target_size),
                              interpolation=cv2.INTER_LANCZOS4)
    return {
        'resized_image': resized_image,
        'padding_top': padding_top,
        'padding_bottom': padding_bottom,
        'cropped_width': cropped_width,
        'cropped_height': cropped_height
    }


def restore_cropped_gtb(resized_image, original_image, bbox, padding_info):
    """
    Restore a cropped and resized image to its original location in the original image.
    
    Args:
        resized_image (np.array): The image resized to target size (512x512).
        original_image (np.array): The original image from which cropping was done.
        bbox (tuple): The bounding box (left, top, right, bottom) used for cropping.
        padding_info (dict): Dictionary containing padding and dimension information from
                             the `square_cropped_gtb` function.

    Returns:
        np.array: The original image with the restored cropped area placed correctly.
    """

    # Extract padding and dimension information
    padding_top = padding_info['padding_top']
    padding_bottom = padding_info['padding_bottom']
    cropped_width = padding_info['cropped_width']
    cropped_height = padding_info['cropped_height']

    # Resize the image back to its original cropped size with padding
    resized_padded_image = cv2.resize(resized_image, (cropped_width, cropped_height), interpolation=cv2.INTER_LANCZOS4)

    # Remove padding added during resizing
    if padding_top > 0:
        resized_padded_image = resized_padded_image[padding_top:, :]
    if padding_bottom > 0:
        resized_padded_image = resized_padded_image[:-padding_bottom, :]

    # Original height of the cropped region
    #original_image_height = cropped_height - (padding_top + padding_bottom)

    #if padding_top > 0 or padding_bottom > 0:
    #    restored_image = resized_padded_image[padding_top:padding_top + original_image_height, :]
    #else:
    #    restored_image = resized_padded_image

    # Place the restored image back into the original image
    left, top, right, bottom = map(int, bbox[:4])
    original_image[top:top + resized_padded_image.shape[0], left:left + resized_padded_image.shape[1]] = resized_padded_image

    return original_image


def load_gfpgan(model_path):
    restorer = GFPGANer(
        model_path=model_path,
        arch="original",
        channel_multiplier=1,
        bg_upsampler=None)
    return restorer


def full_body_retouch(image, bbox, restorer, window_size=512, reverse_pixel=12,
                      debug=False, debug_folder=""):
    print("Debug Folder from full_body_retouch = ", debug_folder)
    x1, y1, x2, y2 = map(int, bbox)
    person_image = image[y1:y2, x1:x2]
    pad_height = (window_size - person_image.shape[0] % window_size) % window_size
    pad_width = (window_size - person_image.shape[1] % window_size) % window_size
    padded_person_image = np.pad(person_image, ((0, pad_height), (0, pad_width),
                                 (0, 0)), mode='constant')
    num_windows_x = (padded_person_image.shape[1] - window_size) // \
                     (window_size - reverse_pixel) + 1
    num_windows_y = (padded_person_image.shape[0] - window_size) // \
                     (window_size - reverse_pixel) + 1
    for i in range(num_windows_y):
        for j in range(num_windows_x):
            if i == 0 and j == 0:
                start_y = 0
                start_x = 0
            else:
                start_y = i * (window_size - reverse_pixel)
                start_x = j * (window_size - reverse_pixel)
            person_window = padded_person_image[start_y:start_y + window_size,
                                               start_x:start_x + window_size, :]
            if debug:
                window_filename = f"{start_x}_{start_y}.jpg"
                window_path = os.path.join(debug_folder, window_filename)
                cv2.imwrite(window_path, person_window)
            retouched_window = restorer.enhance_part(person_window)
            padded_person_image[start_y:start_y + window_size,
                            start_x:start_x + window_size, :] = retouched_window
    retouched_bbox = padded_person_image[:person_image.shape[0],
                                        :person_image.shape[1], :]
    return retouched_bbox


def process_image_person(args, original_image_path, output_retouch_path,
                        output_stacked_path, models_dict):
    original_image = cv2.imread(original_image_path)
    
    if original_image is None:
        print(f"Error loading image: {original_image_path}")
        return
    gt = original_image.copy()
    image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    image_debug_folder = ""
    if args.debug:
        stage = "full_body_retouch"
        image_debug_folder = os.path.join(args.debug_dir, stage, image_name)
        os.makedirs(image_debug_folder, exist_ok=True)
    body_retoucher = models_dict.get("body_retoucher")
    teeth_whitener = models_dict.get("teeth_whitener")
    glare_remover = models_dict.get("glare_removal")
    face_retoucher = models_dict.get("face_retoucher")
    gtb_detector = models_dict.get("gtb_detector")
    person_detector = models_dict.get("person_detector")
    face_detector = models_dict.get("face_detector")
    results = person_detector.predict(original_image_path, imgsz=640, conf=0.5,
                                classes=[0])
    detected_boxes = []
    for result in results:
        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
        print("Person detected = ", len(boxes_xyxy))
        detected_boxes.extend(boxes_xyxy)
    for box in detected_boxes:
        # body retouching
        retouched_region = full_body_retouch(original_image, box, body_retoucher,
                                       args.window_size, args.reverse_pixel,
                                       args.debug, image_debug_folder)
        x1, y1, x2, y2 = map(int, box)
        # face detection
        faces, faces_cords = dlib_face(retouched_region, face_detector, padding=0)
        #detect_face_yunet(retouched_region, face_detector, padding=0)
        print("faces = ", len(faces))
        if faces is not None:
            for i, face in enumerate(faces):
                # teeth whitening
                mouth_bbox = get_face_parts(gtb_detector, face, class_id=1)
                print(mouth_bbox)
                if mouth_bbox is not None:
                    padding_info = square_cropped_gtb(face, mouth_bbox[0])
                    white_teeth = teeth_whitener.enhance_part(padding_info['resized_image'])
                    restored_white_teeth_face = restore_cropped_gtb(white_teeth,
                                                              face.copy(),
                                                              mouth_bbox[0],
                                                              padding_info)
                    if args.debug:
                        stage = "teeth_whitening"
                        image_debug_folder_t = os.path.join(args.debug_dir,
                                                           stage, image_name)
                        os.makedirs(image_debug_folder_t, exist_ok=True)
                        face_name = image_debug_folder_t + '/face_' + str(i) + \
                                    '.png'
                        model_input = image_debug_folder_t + '/model_input_' \
                                      + str(i) + '.png'
                        model_output = image_debug_folder_t + '/model_output' + \
                                       str(i) + '.png'
                        restored = image_debug_folder_t + '/restored' + str(i) +\
                                   '.png' 
                        stacked = image_debug_folder_t + '/stacked' + str(i) +\
                                 '.png'
                        cv2.imwrite(face_name, face)
                        cv2.imwrite(model_input, padding_info['resized_image'])
                        cv2.imwrite(model_output, white_teeth)
                        cv2.imwrite(restored, restored_white_teeth_face)
                        cv2.imwrite(stacked, np.hstack((face,
                                                    restored_white_teeth_face)))
                    print('restored_white_teeth_face shape = ', restored_white_teeth_face.shape)
                    print('face shape = ', face.shape)
                    face = restored_white_teeth_face.copy()
                else:
                    print("Teeth not detected")
                # glass glare removal
                image_dbg = os.path.join(args.debug_dir, 'glare_removal', image_name)
                #cv2.imwrite(image_dbg + 'before.png', face)
                glasses_bbox = get_face_parts(gtb_detector, face, class_id=0)
                if glasses_bbox is not None:
                    padding_info_1 = square_cropped_gtb(face, glasses_bbox[0])
                    #cv2.imwrite(image_dbg + 'before.png', face)
                    glare_removed = glare_remover.enhance_part(padding_info_1['resized_image'])
                    restored_gr_face = restore_cropped_gtb(glare_removed, face.copy(),
                                                           glasses_bbox[0],
                                                           padding_info_1)
                    cv2.imwrite(image_dbg + 'before.png', face)
                    if args.debug:
                        stage = "glare_removal"
                        image_debug_folder_g = os.path.join(args.debug_dir, stage,
                                                         image_name)
                        os.makedirs(image_debug_folder_g, exist_ok=True)
                        face_name = image_debug_folder_g + '/face_' + str(i) + \
                                    '.png'
                        model_input = image_debug_folder_g + '/model_input_' \
                                      + str(i) + '.png'
                        model_output = image_debug_folder_g + '/model_output' + \
                                       str(i) + '.png'
                        restored = image_debug_folder_g + '/restored' + str(i) +\
                                   '.png' 
                        stacked = image_debug_folder_g + '/stacked' + str(i) +\
                                 '.png'
                        cv2.imwrite(face_name, face)
                        cv2.imwrite(model_input, padding_info_1['resized_image'])
                        cv2.imwrite(model_output, glare_removed)
                        cv2.imwrite(restored, restored_gr_face)
                        cv2.imwrite(stacked, np.hstack((face,
                                                    restored_gr_face)))
                    face = restored_gr_face
                else:
                    print("Glasses not detected")
                # face retouching
                parts, split_sizes = divide_and_resize_face(face)
                retouch_parts = [face_retoucher.enhance_part(part) for part in parts]
                retouched_face = reassemble_face_from_parts(retouch_parts,
                                                           split_sizes)
                cv2.imwrite(output_retouch_path, original_image)
                cv2.imwrite(output_stacked_path, np.hstack((face,
                                                retouched_face)))
                if args.debug:
                    stage = "face_retouching"
                    image_debug_folder_fr = os.path.join(args.debug_dir, stage,
                                                     image_name)
                    os.makedirs(image_debug_folder_fr, exist_ok=True)
                    face_name = image_debug_folder_fr + '/face_' + str(i) + '.png'
                    restored = image_debug_folder_fr + '/restored' + str(i) + '.png'
                    stacked = image_debug_folder_fr + '/stacked' + str(i) + '.png'
                    cv2.imwrite(face_name, face)
                    cv2.imwrite(restored, retouched_face)
                    cv2.imwrite(stacked, np.hstack((face,
                                                retouched_face)))
                face = retouched_face
                # paste back face
                fx1, fy1, fx2, fy2 = faces_cords[i]
                retouched_region[fy1:fy2, fx1:fx2] = face
        # paste back region
        original_image[y1:y2, x1:x2] = retouched_region
    cv2.imwrite(output_retouch_path, original_image)
    if args.stacked:
        hstack_image = np.hstack((gt, original_image))
        cv2.imwrite(output_stacked_path, hstack_image)


def main(args):
    if not os.path.exists(args.output_retouch_dir):
        os.makedirs(args.output_retouch_dir)
    if args.stacked:
        if not os.path.exists(args.output_stack_dir):
            os.makedirs(args.output_stack_dir)
    original_images = sorted(os.listdir(args.original_images_dir))
    # load models
    '''
    body_retoucher = load_gfpgan(args.full_body_retouch)
    teeth_whitener = load_gfpgan(args.teeth_whitening)
    glare_removal = load_gfpgan(args.glare_removal)
    face_retoucher = load_gfpgan(args.face_retouching)
    gtb_detector = YOLO(args.gtb_model)
    person_detector = YOLO(args.yolo_model)
    face_detector = cv2.FaceDetectorYN.create(args.yunet_model, "",
                                             (320, 320), 0.3, 0.3, 5000)
    '''
    models_dict = {
    "body_retoucher": load_gfpgan(args.full_body_retouch),
    "teeth_whitener": load_gfpgan(args.teeth_whitening),
    "glare_removal": load_gfpgan(args.glare_removal),
    "face_retoucher": load_gfpgan(args.face_retouching),
    "gtb_detector": YOLO(args.gtb_model),
    "person_detector": YOLO(args.yolo_model),
    "face_detector": dlib.get_frontal_face_detector()}
    #cv2.FaceDetectorYN.create(args.yunet_model, "",(320, 320), 0.30, 0.2, 5000)}
    for original_image_name in original_images:
        original_image_path = os.path.join(args.original_images_dir,
                                          original_image_name)
        output_retouch_path = os.path.join(args.output_retouch_dir,
                                          original_image_name)
        output_stacked_path = ''
        if args.stacked:
            output_stacked_path = os.path.join(args.output_stack_dir,
                                              original_image_name)
        process_image_person(args, original_image_path, output_retouch_path,
                                output_stacked_path, models_dict)
        '''
        try:
            process_image_person(args, original_image_path, output_retouch_path,
                                output_stacked_path, models_dict)
        except Exception as e:
            print("Image Failed: ",original_image_name)
            print("Error: ", e)
        '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retouch skin areas in \
                detected persons")
    ##########################  I/O Directory Paths   ##########################
    # original images dir
    parser.add_argument("-oid","--original_images_dir", type=str,
        default="/data/pd/datasets/colorcorr/Selected/ground_truth/gt", help="Path to the directory of original images")
    # output retouch dir
    parser.add_argument("-ord","--output_retouch_dir", type=str,
        default="/data/pd/datasets/colorcorr/Selected/ground_truth/Results/Retouched", help="Path to save the retouched images")
    # output stack dir
    parser.add_argument("-osd", "--output_stack_dir", type=str,
        default="/data/pd/datasets/colorcorr/Selected/ground_truth/Results/Stacked",
        help="Path to save the stack retouched images")
    # output debug dir
    parser.add_argument("-dbgd","--debug_dir", type=str,
        default="/data/pd/datasets/colorcorr/Selected/ground_truth/Results/Debug", help="Path to save the retouched images")
    ##########################  I/O Directory Paths   ##########################

    ##########################   Model Paths          ##########################
    # full body retouch
    parser.add_argument("-fbr","--full_body_retouch", type=str,
        default="/data/pd/trained_models/pd_retouch_subface_20231214.pth",
        help="Path to the full body retouch model")
    # teeth whitening
    parser.add_argument("-tw","--teeth_whitening", type=str,
        default="/data/pd/trained_models/gfpgan/20240831_teeth/net_g_760000.pth",
        help="Path to the teeth whitening model")
    # glass glare
    parser.add_argument("-gr","--glare_removal", type=str,
        default="/data/pd/trained_models/gfpgan/20240831_glare/net_g_620000.pth",
        help="Path to the glass glare model")
    # face retouching
    parser.add_argument("-fr","--face_retouching", type=str,
        default="/data/pd/trained_models/pd_retouch_subface_20231214.pth",
        help="Path to the face retouching model")
    # glasses, teeth and braces detector
    parser.add_argument("-gtb","--gtb_model", type=str,
        default="/data/pd/trained_models/Yolo/8_July_Glasses_teeth_Detector/best.pt",
        help="Path to the  glasses, teeth and braces detector model")
    # yolo person detection model
    parser.add_argument("-yolo","--yolo_model", type=str,
        default="/data/pd/trained_models/Yolo/yolo_person_detector/yolov8x.pt",
        help="Path to the YOLOv8x model")
    # yunet face detection model
    parser.add_argument("-yu","--yunet_model", type=str,
        default="/data/pd/trained_models/Face_DETECTION/yunet.onnx",
        help="Path to the yunet model")
    ##########################   Model Paths       #############################

    ##########################   Settings          #############################
    # reverse pixel (body retouch overlapping)
    parser.add_argument("-rvpx",'--reverse_pixel', type=int, default=12,
        help="Pixel offset for the sliding window")
    # window size
    parser.add_argument("-ws",'--window_size', type=int, default=512,
        help="Size of the sliding window")
    # boolean stacked (saving stacked images)
    parser.add_argument("-stk","--stacked", action="store_true",
        help="Stack the images")
    # boolean debug
    parser.add_argument("-dbg",'--debug', action="store_true",
        help="Saving images at each stage")
    args = parser.parse_args()
    main(args)
