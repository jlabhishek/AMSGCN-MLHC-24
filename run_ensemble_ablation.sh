
declare -a arr=( "RUN1" "RUN2" "RUN3" "RUN4" "RUN5")
for i in "${arr[@]}"
do
	python ensemble_ablation.py --run $i
	#python ensemble_ablation.py --run $i --test 0 >> ${i}_ensemble_ablation
	
	#python ensemble_ablation.py --run $i --test 1 >> ${i}_ensemble_ablation
	
	
done
