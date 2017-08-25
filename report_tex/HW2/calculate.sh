#!/bin/bash
# Jerry Sun ys7va 2017-03-07
# calculate.sh
# used for automatically submit slurm.sh to SLURM and record the timing

# the first parameter given specified the number of iterations to run
iterate=$1
# the second parameter given specified the number of frame to render in blender
# 1 - 250 used for testing (use smaller value)
frameCounts=$2
# list of number of tasks that is going to be tested
numOfTasksList=(40 20 10 8 4 2 1)

# for loop to submit all tasks "iterate" times
for i in `seq 1 $iterate`;
do
# loop around all different number of tasks
for numOfTasks in "${numOfTasksList[@]}"
do
	# determine the task distribution
	# if numOfTasks is larger then 20, then there are in total 20 nodes, and each is assigned
	# with corresponding number of tasks, namely numOfTasks/20.
	if [ $numOfTasks -gt 20 ]
	then
		let "tasksPerNodes=$numOfTasks/20"
		numOfNodes=20
	else
		# if numOfTasks is smaller or equal to 20, they are all distributed on different nodes
		# and each node has only one task
		tasksPerNodes=1
		numOfNodes=$numOfTasks
	fi
	# endif
	# record the hyperparameter specified above into outputfile
	echo "Total Tasks: $numOfTasks, Nodes: $numOfNodes, Tasks per node: $tasksPerNodes" >> outputfile
	# submit the shell script to slurm specifying the required hyperparameters(detail see slurm.sh)
	sbatch --nodes=$numOfNodes --ntasks=$numOfTasks --ntasks-per-node=$tasksPerNodes slurm.sh $frameCounts
	# check if output.avi has been produced, if not keep waiting, and recheck every 10 seconds
	while [ ! -f output.avi ];
	do
		sleep 10
	done
	# grep the recorded timing in output of the slurm.sh into the outputfile
	grep 'Durations' script-output >> outputfile
	# remove the output.avi and ready for the next task
	rm -f output.avi
done
# endfor for
done
