import h5py
import cv2


def get_name(index, hdf5_data):
    name_ref = hdf5_data['/digitStruct/name'][index].item()
    return ''.join([chr(v[0]) for v in hdf5_data[name_ref]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item_ref = hdf5_data['/digitStruct/bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item_ref][key]
        values = [hdf5_data[attr[i].item()][0][0].astype(int)
                  for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
        attrs[key] = values
    return attrs


def SVHN2YOLO(path_to_digitStruct, path_to_YOLO_labels):
    with h5py.File(path_to_digitStruct, 'r') as hdf5_data:
        for idx in range(12668, len(hdf5_data['/digitStruct/bbox'])):
            # for idx in range(12667, 12668):
            img_path = get_name(idx, hdf5_data)
            img_id = img_path.split('.')[0]
            img_meta = get_bbox(idx, hdf5_data)
            img = cv2.imread('train/' + img_path)
            img_height, img_width, _ = img.shape
            with open('{}/{}.txt'.format(path_to_YOLO_labels, img_id), 'w') as fd:
                obj_labels = []
                for obj_idx in range(len(img_meta['label'])):
                    label = img_meta['label'][obj_idx]
                    left = img_meta['left'][obj_idx]
                    top = img_meta['top'][obj_idx]
                    width = img_meta['width'][obj_idx]
                    height = 11  # img_meta['height'][obj_idx]
                    c_x = (left + width/2) / img_width
                    c_y = (top + height/2) / img_height
                    width_n = width / img_width
                    height_n = height / img_height
                    assert(c_y <= 1 and c_y >= 0)
                    assert(width_n <= 1 and width_n >= 0)
                    assert(height_n <= 1 and height_n >= 0)
                    obj_labels.append('{:.0f} {:f} {:f} {:f} {:f}'.format(
                        label % 10, c_x, c_y, width_n, height_n))
                fd.write('\n'.join(obj_labels))


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--svhn_meta_file', type=str, dest='in_file',
                        default='digitStruct.mat', help='path to digitStruct.mat')
    parser.add_argument('--yolov5_meta_dir', type=str, dest='out_dir',
                        default='metaa', help="path to yolov5 labels' dir")
    opt = parser.parse_args()

    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir, exist_ok=False)
    SVHN2YOLO(opt.in_file, opt.out_dir)
