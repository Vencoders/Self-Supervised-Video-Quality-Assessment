j=5

## Fine-tuning

for ((i=0; i<j; i++))
do
    echo `CUDA_VISIBLE_DEVICES=1 python3 testSY.py --epoch=100 --batch-size=16 --batch-test=8 --dataset=CSIQ_VQA --frame=64 --base_lr=3e-4 --loss=plcc --fine_tune=True --best=0.92 --idx=$i`;
done

#for ((i=0; i<j; i++))
#do
#    echo `python3 outTESTSY.py --batch-size=16 --batch-test=8 --dataset=CSIQ_VQA --frame=32 --base_lr=2e-4 --loss=plcc --fine_tune=True --best=0.92 --idx=$i`;
#done

#CUDA_VISIBLE_DEVICES=1