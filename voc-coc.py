import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from pathlib import Path
import yaml


with open("datasets\\tables\\data.yaml", 'r') as file:
    data = yaml.load(file, Loader=yaml.FullLoader)


def convert_label(path, lb_path, image_id,image_set):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(f"datasets/tables/anno/{image_set}/{image_id}.xml")
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    names = list(data['names'].values()) 
    for obj in root.iter('object'): 
        cls = obj.find('name').text
        if cls in names and int(obj.find('difficult').text) != 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)
            out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + '\n')

dir = Path(data['path'])

path = dir / 'images'
for image_set in 'train', 'val', 'test':
    imgs_path = dir / 'images' / f'{image_set}'
    lbs_path = dir / 'labels' / f'{image_set}'

    lbs_path.mkdir(exist_ok=True, parents=True)

    image_ids = []
    for filename in os.listdir(f"datasets/tables/images/{image_set}"):
        if filename.endswith(".jpg"):
            image_ids.append(os.path.splitext(filename)[0])

    for image_id in tqdm(image_ids, desc=f'{image_set}'):
        f = f'datasets/tables/images/{image_set}/{image_id}.jpg'  
        f_name = os.path.basename(f)
        lb_path = (lbs_path / f_name).with_suffix('.txt')  
        #os.rename(f, imgs_path / f_name)  # move image
        convert_label(path, lb_path, image_id,image_set)  
