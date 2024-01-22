import math
import uuid

import cv2
import numpy as np
import tensorflow
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_UNET = load_model('models/model_unet_final.h5')


def segm_unet(img_path, pixel_spacing):
    img = Image.open(img_path)
    img_array = tensorflow.keras.utils.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = expanded_img_array / 255.
    prediction = MODEL_UNET.predict(preprocessed_img)

    inter_disc_color = (0, 204, 0, 80)
    inter_disc_line_color = (0, 255, 0)
    tectal_canal_color = (204, 0, 204, 100)
    tectal_canal_line_color = (255, 0, 255)
    posterior_element_color = (0, 0, 255, 120)
    posterior_element_line_color = (255, 0, 0)
    anterior_posterior_region_color = (0, 255, 255, 100)
    anterior_posterior_region_line_color = (255, 255, 0)

    mask1 = get_mask(np.round(prediction[0, :, :, 0] * 255), inter_disc_color)
    img.paste(mask1, (0, 0), mask1)
    mask2 = get_mask(np.round(prediction[0, :, :, 1] * 255), tectal_canal_color)
    img.paste(mask2, (0, 0), mask2)
    mask3 = get_mask(np.round(prediction[0, :, :, 2] * 255), posterior_element_color)
    img.paste(mask3, (0, 0), mask3)
    mask4 = get_mask(np.round(prediction[0, :, :, 3] * 255), anterior_posterior_region_color)
    img.paste(mask4, (0, 0), mask4)

    vis1 = cv2.cvtColor(np.array(mask1), cv2.COLOR_RGBA2GRAY)
    _, binary_image1 = cv2.threshold(vis1, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    cv2.drawContours(image, contours, -1, inter_disc_line_color, 1)

    vis2 = cv2.cvtColor(np.array(mask2), cv2.COLOR_RGBA2GRAY)
    _, binary_image2 = cv2.threshold(vis2, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, tectal_canal_line_color, 1)

    vis3 = cv2.cvtColor(np.array(mask3), cv2.COLOR_RGBA2GRAY)
    _, binary_image3 = cv2.threshold(vis3, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, posterior_element_line_color, 1)

    vis4 = cv2.cvtColor(np.array(mask4), cv2.COLOR_RGBA2GRAY)
    _, binary_image4 = cv2.threshold(vis4, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, anterior_posterior_region_line_color, 1)

    image2 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    image_segm = Image.fromarray(image2)
    file_title_segm = str(uuid.uuid4())
    file_name_semg = 'output/' + file_title_segm + '.png'
    image_segm.save(file_name_semg)

    disk_area = cv2.countNonZero(binary_image1)
    disk_area = float(disk_area) * (pixel_spacing ** 2)
    canal_area = cv2.countNonZero(binary_image2)
    canal_area = float(canal_area) * (pixel_spacing ** 2)
    div_area = -1
    if disk_area > 0 and canal_area > 0:
        div_area = math.sqrt(float(canal_area) / float(disk_area))

    bbox = mask1.getbbox()
    x1 = bbox[0]
    x2 = bbox[2]
    bbox2 = mask2.getbbox()
    x21 = bbox2[0]
    x22 = bbox2[2]
    bbox3 = mask3.getbbox()
    x31 = bbox3[0]
    x32 = bbox3[2]
    bbox4 = mask4.getbbox()
    x41 = bbox4[0]
    x42 = bbox4[2]

    xa1 = [x1, x21, x31, x41]
    xa2 = [x2, x22, x32, x42]

    s = (float(x2) - float(x1)) / 2
    s = int(s)
    s = x1 + s

    disk_dist = 0
    canal_dist = 0
    array1 = np.array(mask1)
    array2 = np.array(mask2)
    start_disk_point = [0, 0]
    end_disk_dist_point = [0, 0]
    start_canal_point = [0, 0]
    end_dist_canal_point = [0, 0]

    for x in range(mask1.height):
        if array1[x, s, 3] > 70:
            disk_dist = disk_dist + 1
            end_disk_dist_point = [s, x]
            if start_disk_point[0] == 0:
                start_disk_point = [s, x]
        if array2[x, s, 3] > 70:
            canal_dist = canal_dist + 1
            end_dist_canal_point = [s, x]
            if start_canal_point[0] == 0:
                start_canal_point = [s, x]
    disk_dist = float(disk_dist)
    canal_dist = float(canal_dist)
    div_dist = -1.
    if disk_dist == 0 and canal_dist == 0:
        bbox = mask2.getbbox()
        x1 = bbox[0]
        x2 = bbox[2]
        s = (float(x2) - float(x1)) / 2
        s = int(s)
        s = x1 + s
        for x in range(mask1.height):
            if array1[x, s, 3] > 70:
                disk_dist = disk_dist + 1
                end_disk_dist_point = [s, x]
                if start_disk_point[0] == 0:
                    start_disk_point = [s, x]
            if array2[x, s, 3] > 70:
                canal_dist = canal_dist + 1
                end_dist_canal_point = [s, x]
                if start_canal_point[0] == 0:
                    start_canal_point = [s, x]

    disk_dist = float(disk_dist) * pixel_spacing
    canal_dist = float(canal_dist) * pixel_spacing
    if disk_dist > 0 and canal_dist > 0:
        div_dist = canal_dist / disk_dist

    text_height = 12
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    text_thickness = 1

    if start_canal_point[0] > 0:
        color = (255, 255, 255)
        x2 = max(xa2)
        height = max(start_canal_point[1], end_dist_canal_point[1]) - min(start_canal_point[1], end_dist_canal_point[1])
        if text_height > height:
            text_height = height
        image = cv2.line(image, start_canal_point, end_dist_canal_point, color, 1)
        image = cv2.line(image, (x2 + 25, start_canal_point[1]), start_canal_point, color, 1)
        image = cv2.line(image, (x2 + 25, end_dist_canal_point[1]), end_dist_canal_point, color, 1)
        image = cv2.line(image, (x2 + 12, start_canal_point[1]), (x2 + 12, end_dist_canal_point[1]), color, 1)

        text = str(canal_dist)
        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        while text_size[1] > text_height - 2 and font_scale > 0.1:
            font_scale = font_scale - 0.05
            text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        w1 = x2 + 25
        h1 = int(end_dist_canal_point[1] - (text_height - text_size[1]) / 2) - 1
        text_origin = (w1, h1)
        cv2.putText(image, text, text_origin, font, font_scale, text_color, text_thickness) \
 \
        text = "{:.4f}".format(canal_area)
        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        h2 = int(h1 / 2)
        text_origin = (w1, h2)
        cv2.putText(image, text, text_origin, font, font_scale, text_color, text_thickness)
        image = cv2.line(image, (w1 - 3, h2 + 2), (w1 + text_size[0] + 3, h2 + 2), color, 1)

        x1 = bbox2[0]
        x2 = bbox2[2]
        y1 = bbox2[1]
        y2 = bbox2[3]
        dx = int((max(x1, x2) - min(x1, x2)) / 4)
        dy = int((max(y1, y2) - min(y1, y2)) / 2)

        image = cv2.line(image, (w1 - 3, h2 + 2), (x2 - dx, y1 + dy), color, 1)

    if start_disk_point[0] > 0:
        color = (255, 255, 255)
        x1 = min(xa1)
        image = cv2.line(image, start_disk_point, end_disk_dist_point, color, 1)
        image = cv2.line(image, (x1 - 25, start_disk_point[1]), start_disk_point, color, 1)
        image = cv2.line(image, (x1 - 25, end_disk_dist_point[1]), end_disk_dist_point, color, 1)
        image = cv2.line(image, (x1 - 12, start_disk_point[1]), (x1 - 12, end_disk_dist_point[1]), color, 1)

        text = str(disk_dist)
        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        y_avg = max(start_disk_point[1], end_disk_dist_point[1]) - min(start_disk_point[1], end_disk_dist_point[1])
        w1 = x1 - 25 - text_size[0]
        h1 = int(end_disk_dist_point[1] - y_avg / 2 + text_size[1] / 2)
        text_origin = (w1, h1)
        cv2.putText(image, text, text_origin, font, font_scale, text_color, text_thickness)

        text = "{:.4f}".format(disk_area)
        text_size, _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        h2 = int(h1 / 2)
        text_origin = (w1, h2)
        cv2.putText(image, text, text_origin, font, font_scale, text_color, text_thickness)
        image = cv2.line(image, (w1 - 3, h2 + 2), (w1 + text_size[0] + 3, h2 + 2), color, 1)

        x1 = bbox[0]
        x2 = bbox[2]
        y1 = bbox[1]
        y2 = bbox[3]
        dx = int((max(x1, x2) - min(x1, x2)) / 4)
        dy = int((max(y1, y2) - min(y1, y2)) / 2)

        image = cv2.line(image, (w1 + text_size[0] + 3, h2 + 2), (x1 + dx, y1 + dy), color, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_params = Image.fromarray(image)
    file_title_params = str(uuid.uuid4())
    file_name_params = 'output/' + file_title_params + '.png'
    image_params.save(file_name_params)

    disk_area = max(0, disk_area)
    canal_area = max(0, canal_area)
    div_area = max(0, div_area)
    disk_dist = max(0, disk_dist)
    canal_dist = max(0, canal_dist)
    div_dist = max(0, div_dist)
    return {"disk_area": disk_area,
            "canal_area": canal_area,
            "div_area": div_area,
            "disk_dist": disk_dist,
            "canal_dist": canal_dist,
            "div_dist": div_dist,
            "image_segm": file_title_segm,
            "image_params": file_title_params}


def get_mask(prediction_img, color):
    img_array = prediction_img.astype(np.uint8)

    mask = np.zeros((320, 320, 4), dtype=np.uint8)

    mask[img_array < 128] = [0, 0, 0, 0]
    mask[img_array >= 128] = color

    return Image.fromarray(mask, mode='RGBA')


if __name__ == '__main__':
    # print(segm_unet('C1_0050_D4.png', 0.6875))
    pass
