import os
import csv
import json
import numpy as np
from tqdm import tqdm
import imageio
import skimage


def csv_to_masks(dataset_dir="", annot_fn=""):

    _image_ids = []
    image_info = []
    # Background is always the first class
    class_info = [{"source": "", "id": 0, "name": "BG"}]
    source_class_ids = {}
    
    # annotations = json.load(annot_fn)
    # annotations = list(annotations.values())  # don't need the dict keys

    # # The VIA tool saves images in the JSON even if they don't have any
    # # annotations. Skip unannotated images.
    # annotations = [a for a in annotations if a['regions']]
    
    masks = {}
    prev_image_path = None    # for cacheing

    with open(annot_fn, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
         
        # Convert each row into a dictionary
        # and add it to data
        for a in csvReader:

            print(a)
        
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            shape_attr = a['region_shape_attributes']
            if shape_attr is not None:
                # shape_attr = shape_attr.replace('""', '"')    # happens with some of the files
                shape_attr = eval(shape_attr)    # dict -> str
            # print('regions:', shape_attr)

            image_id = a['filename']

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. 
            image_path = os.path.join(dataset_dir, image_id)
            if image_path != prev_image_path:
                image = imageio.imread(image_path)
                height, width = image.shape[:2]
                # print('image_path', image_path, height, width)
            prev_image_path = image_path

            if image_id not in masks.keys():
                mask = np.zeros([height, width], dtype=np.bool)
                # print('creating a mask', mask.shape)
            else:
                # print('image_id:', image_id)
                # print('masks keys:', masks.keys())
                mask = masks[image_id]
                # print('saved mask res:', mask.shape)
            
            # print(shape_attr)
            if len(shape_attr) > 0 and 'phyllodoce' in a['region_attributes'].lower():
                if shape_attr['name'] == 'polygon':
                    
                    # Get indexes of pixels inside the polygon and set them to 1
                    rr, cc = skimage.draw.polygon(shape_attr['all_points_y'], shape_attr['all_points_x'])
                    # print('rr.min(), rr.max()', rr.min(), rr.max())
                    mask[rr - 1, cc - 1] = True    # "- 1" since VIA apparently has [1, H] not [0, H-1] notation
                
                elif shape_attr['name'] == 'rect':
                    
                    mask[shape_attr['y'] - 1:shape_attr['y'] - 1 + shape_attr['height'],
                         shape_attr['x'] - 1:shape_attr['x'] - 1 + shape_attr['width']] = True
                
                masks[image_id] = mask

    return masks


if __name__ == '__main__':
    # masks = csv_to_masks(
    #     dataset_dir="/home/artem/Загрузки/Phyllodoce_Annotation/iNaturalist",
    #     annot_fn="/home/artem/Загрузки/Phyllodoce_Annotation/iNaturalist/from_iNaturalist.csv"
    # )
    # masks = csv_to_masks(
    #     dataset_dir="/home/artem/Загрузки/Phyllodoce_Annotation/LA_17A",
    #     annot_fn="/home/artem/Загрузки/Phyllodoce_Annotation/LA_17A/via_project_28Dec2022_19h0m_csv.csv"
    # )
    # masks = csv_to_masks(
    #     dataset_dir="/home/artem/Загрузки/Phyllodoce_Annotation/LA_21C",
    #     annot_fn="/home/artem/Загрузки/Phyllodoce_Annotation/LA_21C/via_project_2Jan2023_8h28m_csv.csv"
    # )
    masks = csv_to_masks(
        dataset_dir="/home/artem/Загрузки/Phyllodoce_Annotation/LA_39B",
        annot_fn="/home/artem/Загрузки/Phyllodoce_Annotation/LA_39B/via_project_2Jan2023_10h26m_csv.csv"
    )

    # out_dir = "/home/artem/Загрузки/Phyllodoce_Annotation/iNaturalist/masks"
    # out_dir = "/home/artem/Загрузки/Phyllodoce_Annotation/LA_17A/masks"
    # out_dir = "/home/artem/Загрузки/Phyllodoce_Annotation/LA_21C/masks"
    out_dir = "/home/artem/Загрузки/Phyllodoce_Annotation/LA_39B/masks"
    os.makedirs(out_dir, exist_ok=True)
    for image_id, mask in tqdm(masks.items()):
        imageio.imwrite(os.path.join(out_dir, image_id), mask.astype(np.uint8) * 255)
