import argparse

import cv2
import numpy as np

from attack import Attack
from dct_watermark import DCT_Watermark
from dwt_watermark import DWT_Watermark


def main(args):
    img = cv2.imread(args.origin)
    wm = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)

    if args.wm_type == 'DCT':
        model = DCT_Watermark()
    elif args.wm_type == 'DWT':
        model = DWT_Watermark()
    if args.action_type == "embedding":
        emb_img = model.embed(img, wm)
        cv2.imwrite(args.output, emb_img)
        print("Embedded to {}".format(args.output))
    elif args.action_type == 'extracting':
        signature = model.extract(img)
        #cv2.imshow("signature", signature)
        #cv2.waitKey(0)
        cv2.imwrite(args.output, signature)
        print("Extracted to {}".format(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="compare", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--origin", default="./images/cover.jpg", help="origin image file")
    parser.add_argument("--watermark", default="./images/watermark.jpg", help="watermark image file")
    parser.add_argument("--output", default="./images/watermarked.jpg", help="embedding image file")
    parser.add_argument("--wm_type", default="DCT", type=str, choices=["DCT, DWT"])
    parser.add_argument("--action_type", default="embedding", type=str, choices=["embedding", "extracting"])
    args = parser.parse_args()
    main(parser.parse_args())
