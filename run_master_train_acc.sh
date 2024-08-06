
declare -a arr=( "RUN22" "RUN33" "RUN44" "RUN55")
for i in "${arr[@]}"
do
	cd global



	python main.py --config ./config/kinetics-skeleton_global/train_acceleration.yaml  --device 1 --batch-size 32 --base-lr .1 --num-epoch 200 --step 80 160 --warm_up_epoch 10 --weight-decay .001 --optimizer Adam




	


	
	

	mv work_dir work_dir_$i
	mv runs run_$i
	cd ..
	
done
