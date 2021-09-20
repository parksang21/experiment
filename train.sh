model_name=noisyFeature

gpus="0,"
lr=0.1
max_epoch=100


python main.py --model $model_name \
               --gpus $gpus \
               --accelerator ddp \
               --lr $lr \
               --max_epoch $max_epoch \
