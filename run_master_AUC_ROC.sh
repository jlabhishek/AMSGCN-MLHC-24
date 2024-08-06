rm compare_AUC_mean.csv
declare -a arr=("RUN1"  "RUN2" "RUN3" "RUN4" "RUN5" )
for i in "${arr[@]}"
do

	cd local

	data_name=('joint' 'velocity' 'acceleration' 'boneL' 'boneA')

	mv work_dir_$i work_dir
	mv run_$i run
	
	cd ..
	
	
	cd global

	data_name=('joint' 'velocity' 'acceleration' 'boneL' 'boneA')


	mv work_dir_$i work_dir
	mv run_$i run


	


	
	cd ..
	


	python ensemble_ruroc_2sAGCNFeat.py --test 0 --run $i
	python ensemble_ruroc_2sAGCNFeat.py --test 1 --run $i
	
	cd local
	mv work_dir work_dir_$i
	mv run run_$i
	cd ..
	
	cd global
	mv work_dir work_dir_$i
	mv run run_$i
	cd ..
	
	
done
python mean_compare.py
