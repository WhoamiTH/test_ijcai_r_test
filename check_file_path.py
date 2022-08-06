
# # -*- coding:utf-8 -*-
import os



# ---------------------  检查 test -----------------------------------
import sys
import os
import math
# dataset_list = ['abalone19', 'ecoli1', 'glass0', 'glass5', 'pageblocks1', 'pima', 'vehicle0', 'yeast3', 'yeast5', 'yeast6']
dataset_list = ['glass5']

data_range = 5
record_index = 1

train_infor_method_list = ['normal', 'bm', 'im', 'im2', 'im3', 'both', 'both2', 'both3']
# early_stop_type_list = [ '20000', '15000', '10000', '8000', '5000', '2000']
early_stop_type_list = ['15000', '8000']
test_infor_method_list = [ 'normal', 'bm', 'im', 'both']

ref_num_type_list = ['num']
ref_times_list = ['10']
boundary_type_list = ['half']

device_id = 0

command_list = []
for dataset in dataset_list:
    for sample_method in train_infor_method_list:
        for early_stop_type in early_stop_type_list:
            for test_infor_method in test_infor_method_list:
                for ref_num_type in ref_num_type_list:
                    for ref_times in ref_times_list:
                        for boundary_type in boundary_type_list:
                            cur_command_list = []
                            path_flag = False
                            train_method = 'MLP_{0}_{1}'.format(sample_method, early_stop_type)
                            test_method = '{0}_{1}_{2}_{3}'.format(test_infor_method, ref_num_type, ref_times, boundary_type)
                            
                            cur_path = './test_{0}/result_{1}_{2}/'.format(dataset, train_method, test_method)
                            if os.path.exists(cur_path):
                                cur_command_list.append('rm -rf ./test_{0}/model_{2}/\n'.format(dataset, record_index, train_method)) 

                            cur_path = './test_{0}/model_{1}/'.format(dataset, train_method)
                            if os.path.exists(cur_path):
                                cur_command_list.append('rm -rf ./test_{0}/result_{1}_{2}\n'.format(dataset, train_method, test_method))
                                
                            if len(cur_command_list) != 0:
                                cur_command_list.append('\n')
                                command_list.append(cur_command_list)

with open('del_file.sh', 'w') as fsh:
    fsh.write('#!/bin/bash\n')
    fsh.write('set -e\n\n\n')
    for cur_command_list in command_list:
        for line in cur_command_list:
            fsh.write(line)

