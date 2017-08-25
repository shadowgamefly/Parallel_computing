#!/bin/bash
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3000
#SBATCH --time=00:15:00
#SBATCH --partition=parallel
#SBATCH --account=parallelcomputing
#SBATCH --output=script-output
#SBATCH --exclusive

#Jerry Sun ys7va 2017-03-07
#seq.sh
#used for submit parallel tasks onto slurm for linear partition strategy
#example submit format
#sbatch slurm.sh 250
#the first parameter specify the number of frame to render

#start clock 
STARTTIME=$(date +%s)
# the number of tasks to be used to generate the video
let "multi=$SLURM_NNODES*$SLURM_NTASKS_PER_NODE"

# the number of frames to be used to generate the video
frame_count=$1
# determine how many frames each task need to generate
let "step=$frame_count/$multi"

#load the renderer engine that will generate frames for the video
module load blender

# -b option is for command line access, i.e., without an output console
# -s option is the starting frame
# -e option is the ending frame# -a option indicates that all frames should be rendered

# number of iterations need to be run except last one
let "iterations=$multi-1"
# the startFrame for the first process
startFrame=1
# the endFrame for the first process
let "endFrame=1+$step"

# submit (iterations - 1) number of tasks
for m in $(seq 1 $iterations)
do
  # echo the start and the end frame for debugging
  echo "from $startFrame to $endFrame"

  # flags are similar to slurm.sh, but only involves -s and -e
  # used to specify the range of the frame to generate for this process
	srun -n 1 -N 1 blender -b Star-collapse-ntsc.blend -s $startFrame -e $endFrame -a &
  # determine the next startFrame is just the one after current endFrame
  let "startFrame=$endFrame+1"
  # the next endFrame should be the number of startFrame plus number of frames
  # each process is assigned
  let "endFrame=$startFrame+$step"
done

# Final process to make sure it stops at 250
let "startFrame=$endFrame+1"
echo "from $startFrame to 250"
srun -n 1 -N 1 blender -b Star-collapse-ntsc.blend -s $startFrame -e 250 -a &

# wait untill all processes are finished
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
