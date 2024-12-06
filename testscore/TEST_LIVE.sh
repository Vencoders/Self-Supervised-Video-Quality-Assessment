j=5

## Fine-tuning

for ((i=0; i<j; i++))
do
    echo `CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com python3 testSY.py --epoch=100 --batch-size=8 --batch-test=4 --dataset=LIVE_VQA --frame=64 --base_lr=3e-4 --fine_tune=True --loss=plcc --best=0.9 --idx=$i`;
done

#for ((i=0; i<j; i++))
#do
#    echo `CUDA_VISIBLE_DEVICES=2 python3 testSY.py --epoch=100 --batch-size=4 --batch-test=2 --dataset=CSIQ_VQA --frame=32 --base_lr=2e-4 --loss=mix --fine_tune=True --best=0.92 --idx=$i`;
#done

#for ((i=0; i<j; i++))
#do
#    echo `python3 outTESTSY.py --batch-size=16 --batch-test=8 --dataset=CSIQ_VQA --frame=32 --base_lr=2e-4 --loss=plcc --fine_tune=True --best=0.92 --idx=$i`;
#done

#CUDA_VISIBLE_DEVICES=1
