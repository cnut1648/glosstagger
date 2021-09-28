source activate wsd;

python esc/predict.py --dataset-paths ../esc/data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.data.xml \
    --prediction-types probabilistic \
    --ckpt /home/jiashu/WSD/esc/escher_semcor_best.ckpt \
    --evaluate

