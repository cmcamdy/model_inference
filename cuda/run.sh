dirname=$1
filename=$2

nvcc ./${dirname}/${filename}.cu -o ./output/${dirname}_${filename}

./output/${dirname}_${filename}