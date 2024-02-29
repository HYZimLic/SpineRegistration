import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config

class LITS_preprocess:
    def __init__(self, raw_dataset_path,fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice
        self.size = args.min_slices
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale

        self.valid_rate = args.valid_rate

    def fix_data(self):
        if not os.path.exists(self.fixed_path):
            os.makedirs(join(self.fixed_path,'ct'))
            os.makedirs(join(self.fixed_path, 'label'))
        file_list = os.listdir(join(self.raw_root_path,'ct'))
        Numbers = len(file_list)
        print('Total numbers of samples is :',Numbers)
        for ct_file,i in zip(file_list,range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i+1,Numbers))
            ct_path = os.path.join(self.raw_root_path, 'ct', ct_file)
            seg_path = os.path.join(self.raw_root_path, 'label', ct_file)
            new_ct, new_seg = self.process(ct_path, seg_path, classes = self.classes)
            if new_ct != None and new_seg != None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label', ct_file))


    def process(self, ct_path, seg_path, classes=None):
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)
        ct_array = sitk.GetArrayFromImage(ct)
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        print("Ori shape:",ct_array.shape, seg_array.shape)
        if classes==2:

            seg_array[seg_array > 0] = 1

        # ct_array[ct_array > self.upper] = self.upper
        # ct_array[ct_array < self.lower] = self.lower


        ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=3)
        seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale), order=0)
        

        z = np.any(seg_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]


        # if start_slice - self.expand_slice < 0:
        #     start_slice = 0
        # else:
        #     start_slice -= self.expand_slice
        #
        # if end_slice + self.expand_slice >= seg_array.shape[0]:
        #     end_slice = seg_array.shape[0] - 1
        # else:
        #     end_slice += self.expand_slice

        print("Cut out range:",str(start_slice) + '--' + str(end_slice))

        if end_slice - start_slice + 1 < self.size:
            #print('Too little slice，give up the sample:', ct_file)
            return None,None

        ct_array = ct_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]
        print("Preprocessed shape:",ct_array.shape,seg_array.shape)

        new_ct = sitk.GetImageFromArray(ct_array)
        new_ct.SetDirection(ct.GetDirection())
        new_ct.SetOrigin(ct.GetOrigin())
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        
        new_seg = sitk.GetImageFromArray(seg_array)
        new_seg.SetDirection(ct.GetDirection())
        new_seg.SetOrigin(ct.GetOrigin())
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale), ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))
        return new_ct, new_seg


    # def write_train_val_name_list(self):
    #     data_name_list = os.listdir(join(self.fixed_path, "ct"))
    #     data_num = len(data_name_list)
    #     print('the fixed dataset total numbers of samples is :', data_num)
    #     random.shuffle(data_name_list)
    #
    #     assert self.valid_rate < 1.0
    #     train_name_list = data_name_list[0:int(data_num*(1-self.valid_rate))]
    #     val_name_list = data_name_list[int(data_num*(1-self.valid_rate)):int(data_num*((1-self.valid_rate) + self.valid_rate))]
    #
    #     self.write_name_list(train_name_list, "train_path_list.txt")
    #     self.write_name_list(val_name_list, "val_path_list.txt")


    def write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)
            seg_path = os.path.join(self.fixed_path, 'label', name)
            f.write(ct_path + ' ' + seg_path + "\n")
        f.close()

if __name__ == '__main__':
    raw_dataset_path = ''
    fixed_dataset_path = ''

    args = config.args 
    tool = LITS_preprocess(raw_dataset_path,fixed_dataset_path, args)
    tool.fix_data()

