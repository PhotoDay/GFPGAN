import argparse
import cv2
import glob
import numpy as np
import os
import torch
from ultralytics import YOLO
from basicsr.utils import imwrite
from gfpgan import GFPGANer



def main(args):
    img_list = sorted(glob.glob(os.path.join(args.input, '*')))
    os.makedirs(args.output, exist_ok=True)

    yolo_model = YOLO(args.yolo_model_path)

    restorer = GFPGANer(
        model_path=args.retouch_model_path,
        upscale=2,
        arch='original',
        channel_multiplier=1)

    # restorer = GFPGANer(model_path=args.retouch_model_path)

    for img_path in img_list:
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)

        output_stacked_path = os.path.join(args.output, 'stacked', img_name)
        if os.path.isfile(output_stacked_path):
            continue

        print(f'Processing {img_name} ...')
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        restored_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if args.mode == 1 or args.mode == 3:
            _, _, restored_img = restorer.enhance(
                restored_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight)

        if args. mode == 2 or args.mode == 3:
            results = yolo_model.predict(restored_img, imgsz=512, conf=0.35,
                                        classes=[args.class_id])
            detected_boxes = []
            for result in results:
                boxes = result.boxes
                boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
                print(boxes_xyxy)
                detected_boxes.extend(boxes_xyxy)

            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                glasses_region = restored_img[y1:y2, x1:x2]
                glasses_region_rsz = cv2.resize(glasses_region, (512, 512),
                                                    interpolation=cv2.INTER_CUBIC)
                retouched_glasses_region = restorer.enhance_part(glasses_region_rsz)
                retouched_glasses_region_rsz = cv2.resize(retouched_glasses_region,
                                                    (x2 - x1, y2 - y1),
                                                    interpolation=cv2.INTER_CUBIC)
                restored_img[y1:y2, x1:x2] = retouched_glasses_region_rsz

        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, 'restored_imgs',
                                        f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.output, 'restored_imgs',
                                        f'{basename}.{extension}')

            imwrite(restored_img, save_restore_path)

            stacked_image = np.hstack((input_img, restored_img))
            imwrite( stacked_image, output_stacked_path)

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/data/pd/datasets/teeth_whitening/set_2/original/', help='Input image or folder.')
    parser.add_argument('-rmp', "--retouch_model_path", type=str, default="/data/pd/trained_models/gfpgan/20240826_teeth/net_g_425000.pth", help="Path to the retouch model.")
    parser.add_argument('-ymp', "--yolo_model_path", type=str, default="/data/pd/trained_models/Yolo/8_July_Glasses_teeth_Detector/best.pt", help="Path to the yolo model.")
    parser.add_argument('-o', '--output', type=str, default='/data/pd/datasets/teeth_whitening/set_2_20240826_m', help='Output folder.')
    parser.add_argument("-c", "--class_id", default=1, type=int, help="Class ID for Yolo model object[0: glass, 1:teeth, 2:braces]")
    parser.add_argument("-m", "--mode", default=2, type=int, help="mode 1 is only face level retouching, mode 2 is object (glass/mouth), mode 3 is both")
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument('--ext',type=str,default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs.')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    args = parser.parse_args()

    main(args)
