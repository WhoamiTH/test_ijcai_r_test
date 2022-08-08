# 选择资源

!/bin/bash\n
set -e


mkdir -p ./test_yeast5/result_MLP_im2_2000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im2_2000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im2_2000 test_method=both_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_im2_5000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_im2_5000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_im2_5000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_im2_5000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im2_5000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im2_5000 test_method=both_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_im2_8000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_im2_8000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_im2_8000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_im2_8000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im2_8000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im2_8000 test_method=both_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_im2_10000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_im2_10000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_im2_10000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_im2_10000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im2_10000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im2_10000 test_method=both_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_im2_15000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_im2_15000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_im2_15000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_im2_15000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im2_15000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im2_15000 test_method=both_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_im2_20000_both_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_im2_20000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_im2_20000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_im2_20000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im2_20000 test_method=both_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im2_20000 test_method=both_num_10_half device_id=0



mkdir -p ./test_yeast5/result_MLP_im3_5000_normal_num_10_half/record_1/
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=1 record_index=1 train_method=MLP_im3_5000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=2 record_index=1 train_method=MLP_im3_5000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=3 record_index=1 train_method=MLP_im3_5000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=4 record_index=1 train_method=MLP_im3_5000 test_method=normal_num_10_half device_id=0
python3 ./classifier_MLP/test.py dataset_name=yeast5 dataset_index=5 record_index=1 train_method=MLP_im3_5000 test_method=normal_num_10_half device_id=0



