#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=00:15:00
#SBATCH --partition=parallel
#SBATCH --account=parallelcomputing
#SBATCH --output=script-output
#SBATCH --exclusive

#Jerry Sun ys7va 2017-03-07
#slurm.sh
#used for submit parallel tasks onto slurm
#example submit format
#sbatch --nodes=1 --ntasks=1 --ntasks-per-node=1 slurm.sh 250
#three flags specify the number of nodes/tasks, and the task distribution
#the first parameter specify the number of frame to render

#time counter start
STARTTIME=$(date +%s)

#the total number of task = numOfNodes * numOfTasksPerNode
let "multi=$SLURM_NNODES * $SLURM_NTASKS_PER_NODE"

# the number of frames to be used to generate the video
frame_count=$1

#load the renderer engine that will generate frames for the video
module load blender

# -b option is for command line access, i.e., without an output console
# -s option is the starting frame
# -e option is the ending frame# -a option indicates that all frames should be rendered
# -j option is the number of frames to skip from the last generated frame

# run multi number of tasks all starting from different number of frames
# if multi=4 then 4 tasks will generate corresponding frames
# 1, 5, 9, 13 ...
# 2, 6, 10, 14 ...
# 3, 7, 11, 15 ...
# 4, 8, 12, 16 ...
for m in $(seq 1 $multi)
do
	# -n 1, -N 1 make sure each task take one whole process
	# & option push the last task to the backend and open a new process for the next task
	srun -n 1 -N 1 blender -b Star-collapse-ntsc.blend -s $m -j $multi -e $frame_count -a &
done

# wait to make sure all processes finish before proceeding
wait

# need to give the generated frames some extension; otherwise the video encoder will not worki
ls star-collapse-* | xargs -I % mv % %.jpg

# load the video encoder engine
module load ffmpeg

# start number should be 1 as by default the encoder starts looking from file ending with 0
# frame rate and start number options are set before the input files are specified so that the
# configuration is applied for all files going to the output
ffmpeg -framerate 25 -start_number 1 -i star-collapse-%04d.jpg -vcodec mpeg4 output.avi

# end clock
ENDTIME=$(date +%s)

# calculate the time spent for the whole process
let "DURATION=$ENDTIME-$STARTTIME"

# output the time spent into script-output
echo "Total Durations: $DURATION seconds"

# remove all the generated jpg files for future work
rm -f *.jpg
