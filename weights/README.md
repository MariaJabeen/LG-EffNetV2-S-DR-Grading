# weights/

Not tracked by git (~570 MB). Download the trained LG-EffNetV2-S weights from
https://www.kaggle.com/datasets/mariajabeen01/effnetv2s-v3-dr-weights

    kaggle datasets download -d mariajabeen01/effnetv2s-v3-dr-weights -p weights --unzip

This folder must then contain `effnetv2s_v3_tf_dr.weights.h5` and `history.csv`
(or set DRXAI_WEIGHTS to the folder that does).
