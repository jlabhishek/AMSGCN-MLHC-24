
declare -a arr=( "RUN2" "RUN3" "RUN4" "RUN5")
for i in "${arr[@]}"
do
	cd global

	data_name=('joint' 'velocity' 'acceleration' 'boneL' 'boneA')


	for str in ${data_name[@]}; do
		echo $str
		#python main.py --config ./config/kinetics-skeleton_global/train_${str}.yaml  --device 1 --batch-size 32 --base-lr .1 --num-epoch 1200 --step 400 800 1000  --warm_up_epoch 10 --weight-decay .001 --optimizer Adam --save-interval 50
		python main.py --config ./config/kinetics-skeleton_global/train_${str}.yaml  --device 1 --batch-size 32 --base-lr .1 --num-epoch 200 --step 80 160 --warm_up_epoch 10 --weight-decay .001 --optimizer Adam

	done






	data_folders=("val" "test")
	for data_folder in ${data_folders[@]}; do
		for str in ${data_name[@]}; do
			echo $str
			python main.py --config ./config/kinetics-skeleton_global/${data_folder}_$str.yaml --device 1 --weights runs/ki_agcn_${str}_global-best.pt  --test-batch-size 32
		done
	done


	
	cd ..
	cd local

	data_name=('joint' 'velocity' 'acceleration' 'boneL' 'boneA')


	for str in ${data_name[@]}; do
		echo $str
		#python main.py --config ./config/kinetics-skeleton/train_${str}.yaml  --device 1 --batch-size 32 --base-lr .1 --num-epoch 1200 --step 400 800 1000  --warm_up_epoch 10 --weight-decay .001 --optimizer Adam --save-interval 50
		python main.py --config ./config/kinetics-skeleton/train_${str}.yaml  --device 1 --batch-size 32 --base-lr .1 --num-epoch 200 --step 80 160 --warm_up_epoch 10 --weight-decay .001 --optimizer Adam

	done


	data_folders=("val" "test")
	for data_folder in ${data_folders[@]}; do
		for str in ${data_name[@]}; do
			echo $str
			python main.py --config ./config/kinetics-skeleton/${data_folder}_$str.yaml --device 1 --weights runs/ki_agcn_${str}-best.pt  --test-batch-size 32
		done
	done
	

	
	cd ..
	python ensemble_local_global_add.py

	sleep 2

	python ensemble_test_add.py --test 0 >> $i
	python ensemble_test_add.py --test 1 >> $i
	
	cd local
	mv work_dir work_dir_$i
	mv runs run_$i
	cd ..
	
	cd global
	mv work_dir work_dir_$i
	mv runs run_$i
	cd ..
	
done
