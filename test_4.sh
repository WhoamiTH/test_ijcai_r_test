# 选择资源

#!/bin/bash
set -e



mkdir -p ./test_yeast5/result_MLP_both3_20000_bm_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_20000 test_method=bm_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_20000 test_method=bm_num_10_half device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_2000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_2000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_2000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_2000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_2000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_2000 test_method=im_num_10_half device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_5000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_5000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_5000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_5000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_5000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_5000 test_method=im_num_10_half device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_8000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_8000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_8000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_8000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_8000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_8000 test_method=im_num_10_half device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_10000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_10000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_10000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_10000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_10000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_10000 test_method=im_num_10_half device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_15000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_15000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_15000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_15000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_15000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_15000 test_method=im_num_10_half device_id=1



mkdir -p ./test_yeast5/result_MLP_both3_20000_im_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_both3_20000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_both3_20000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_both3_20000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_both3_20000 test_method=im_num_10_half device_id=1
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_both3_20000 test_method=im_num_10_half device_id=1



