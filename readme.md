## prepare data
# use the same dataset as Spoof_ResNet_TF


# train the net, first change ./config/net_config_ld_conv.py accordingly
python -m train_net

## test data with checkpoints
python -m spoof_eval_video -c 7 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96
python -m spoof_eval_pic -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -f /media/macul/black/spoof_db/MSU_MFSD_collected/original/MSU_MFSD_negative/ -t 0.9 -tf 0.95 -c 7
python -m spoof_test_ld_cm -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2  -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0 -tfl 1.0 -c 7



