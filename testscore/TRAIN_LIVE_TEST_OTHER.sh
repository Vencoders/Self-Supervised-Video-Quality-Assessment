#datasets=("CSIQ_VQA" "KON_VQA" "VQC_VQA")
#frames=("64" "64" "64")
#best=("0.8" "0.7" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i ++ ))
#do
#    echo `CUDA_VISIBLE_DEVICES=3 python3 cross_train_test.py --batch-size=4 --batch-test=2 --dataset=LIVE_VQA --dataset_test=${datasets[$i]} --frame=${frames[$i]} --fine_tune=True --base_lr=2e-5 --loss=mix --best=${best[$i]} --epoch=20`;
#done

#datasets=("LIVE_VQA" "KON_VQA" "VQC_VQA")
#frames=("64" "64" "64")
#best=("0.8" "0.7" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=3 python3 cross_train_test.py --batch-size=8 --batch-test=8 --dataset=CSIQ_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=2e-4 --loss=mix --best=${best[i]} --epoch=30
#done

#datasets=("LIVE_VQA" "CSIQ_VQA" "VQC_VQA")
#frames=("64" "64" "64")
#best=("0.8" "0.8" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=3 python3 cross_train_test.py --batch-size=16 --batch-test=8 --dataset=KON_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=1e-4 --loss=mix --best=${best[i]} --epoch=30
#done

#datasets=("LIVE_VQA" "CSIQ_VQA" "KON_VQA")
#frames=("64" "64" "64")
#best=("0.8" "0.8" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=1 python3 cross_train_test.py --batch-size=16 --batch-test=8 --dataset=VQC_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=2e-4 --loss=mix --best=${best[i]} --epoch=30
#done
#datasets=("CSIQ_VQA" "KON_VQA" "VQC_VQA")
#frames=("64" "64" "64")
#best=("0.8" "0.7" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i ++ ))
#do
#    echo `CUDA_VISIBLE_DEVICES=3 python3 cross_train_test.py --batch-size=8 --batch-test=4 --dataset=LIVE_VQA --dataset_test=${datasets[$i]} --frame=${frames[$i]} --fine_tune=True --base_lr=2e-5 --loss=mix --best=${best[$i]} --epoch=20`;
#done

#datasets=("LIVE_VQA" "KON_VQA" "VQC_VQA")
#frames=("64" "64" "64")
#best=("0.8" "0.7" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=0 python3 cross_train_test.py --batch-size=8 --batch-test=8 --dataset=CSIQ_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=3e-4 --loss=mix --best=${best[i]} --epoch=30
#done

#datasets=("LIVE_VQA" "CSIQ_VQA" "VQC_VQA")
#frames=("16" "16" "16")
#best=("0.8" "0.8" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=1 python3 cross_train_test.py --batch-size=16 --batch-test=8 --dataset=KON_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=8e-5 --loss=mix --best=${best[i]} --epoch=20
#done


#datasets=("VQC_VQA" "VQC_VQA" "VQC_VQA")
#frames=("32" "32" "32")
#best=("0.7" "0.7" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=2 python3 cross_train_test.py --batch-size=16 --batch-test=16 --dataset=CSIQ_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=3e-4 --loss=plcc --best=${best[i]} --epoch=80
#done

datasets=("CSIQ_VQA" "CSIQ_VQA" "CSIQ_VQA")
frames=("64" "64" "64")
best=("0.8" "0.8" "0.8")

for ((i = 0; i < ${#datasets[@]}; i ++ ))
do
    echo `CUDA_VISIBLE_DEVICES=2 python3 cross_train_test.py --batch-size=8 --batch-test=8 --dataset=LIVE_VQA --dataset_test=${datasets[$i]} --frame=${frames[$i]} --fine_tune=True --base_lr=1e-4 --loss=plcc --best=${best[$i]} --epoch=100`;
done

#datasets=("KON_VQA" "KON_VQA")
#frames=("64" "64")
#best=("0.7" "0.7")
#
#for ((i = 0; i < ${#datasets[@]}; i++ ))
#do
#    CUDA_VISIBLE_DEVICES=3 python3 cross_train_test.py --batch-size=16 --batch-test=16 --dataset=VQC_VQA --dataset_test=${datasets[i]} --frame=${frames[i]} --fine_tune=True --base_lr=3e-4 --loss=plcc --best=${best[i]} --epoch=100
#done