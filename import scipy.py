import scipy.io
mat = scipy.io.loadmat("/home/mahsa/gmm_dae/dataset/shanghaitech/ShanghaiTech/part_A/train_data/ground-truth/GT_IMG_1.mat")

dict1 = mat.items()
dict2 = mat.keys()

if 'bounding_box_data' in mat:
    bounding_boxes = mat['bounding_box_data']
    # Now, you can work with the bounding box coordinates.
else:
    print('Bounding box data not found in the MAT file.')

print(mat['image_info'])