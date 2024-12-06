j=5

## Fine-tuning
#for ((i=0; i<j; i++))
#do
#    echo `python3 ourTESTAU.py --batch-size=16 --batch-test=1 --dataset=VQC_VQA --frame=16 --base_lr=3e-4 --fine_tune=True --loss=plcc --best=0.74 --idx=$i`;
#done

for ((i=0; i<j; i++))
do
    echo `CUDA_VISIBLE_DEVICES=1 python3 testAU.py --batch-size=16 --batch-test=8 --dataset=VQC_VQA --frame=64 --base_lr=3e-4 --fine_tune=True --loss=plcc --best=0.74 --idx=$i`;
done


#for ((i=0; i<j; i++))
#do
#    echo `CUDA_VISIBLE_DEVICES=1 python3 testAU.py --batch-size=16 --batch-test=8 --dataset=KON_VQA --frame=64 --base_lr=2e-4 --loss=plcc --fine_tune=True --best=0.84 --idx=$i`;
#done