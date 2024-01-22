import os
import uuid
import numpy as np
import tensorflow
from PIL import Image, ImageDraw
from ultralytics import YOLO


def predict(img_path):
    model = YOLO('best.pt')
    results = model(img_path, device='cpu')
    img = Image.open(img_path)
    mask_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_img)
    for r in results:
        mask = r.masks
        labels = mask.xy
        boxes = r.boxes.data
        clss = boxes[:, 5]
        for _id, label in enumerate(labels):
            color = (255, 250, 250, 128)
            line_color = (255, 250, 250)
            if clss[_id] == 0.:
                color = (100, 149, 237, 128)
                line_color = (100, 149, 237)
            if clss[_id] == 1:
                color = (167, 252, 0, 158)
                line_color = (167, 252, 0)
            elif clss[_id] == 2:
                color = (255, 71, 202, 128)
                line_color = (255, 71, 202)
            draw.polygon(label, fill=color, outline=line_color)
    img.paste(mask_img, (0, 0), mask_img)
    file_name = 'test.png'
    img.save(file_name)


if __name__ == '__main__':
    predict('C_0259_D5_1.png')