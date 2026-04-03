export workspace=/nas2/home/yuhao/code/InfiniDepth/experiments
export commonspace=/nas2/home/yuhao/code/InfiniDepth/common_space
export CUDA_VISIBLE_DEVICES=0

python3 main.py \
    --c training/exp_configs/exps/infinidepth_depthsensor.yaml \
    --i training/exp_configs/components/data/test/infinidepth_mix_data.yaml \
    --e val \
    ckpt_path=checkpoints/depth/infinidepth_depthsensor.ckpt \
    exp_name=eval_infinidepth_depthsensor \
    model.compute_abs_metric=True \
    model.save_orig_pred=True \
    model.save_metrics=True \
    pl_trainer.devices=1 \
    