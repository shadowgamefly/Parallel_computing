#include <stdio.h>
#include <time.h>

int main() {
	system("module load blender")
	clock_t begin = clock();
	int i;
	for (i = 0 ; i < 20; i++) {
		command = "srun -n 1 -N 1 blender -b Star-collapse-ntsc.blend -s "+ i +" -j 20 -e 250 -a &"
		system(command);
	}
	system("wait");
	clock_t end = clock();
	double time = (double) (end - begin);
	printf("cpu time: %f\n", time);
}
