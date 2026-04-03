export workspace=/nas2/home/yuhao/code/InfiniDepth/experiments
export commonspace=/nas2/home/yuhao/code/InfiniDepth/common_space
export CUDA_VISIBLE_DEVICES=0

python3 main.py \
    --c training/exp_configs/exps/infinidepth_depthsensor.yaml \
    --i training/exp_configs/components/data/train/infinidepth_train_hypersim.yaml \
    exp_name=train_infinidepth_depthsensor_on_hypersim \
    model.compute_abs_metric=True \
    model.save_orig_pred=True \
    model.save_metrics=True \
    pl_trainer.devices=1 \
    