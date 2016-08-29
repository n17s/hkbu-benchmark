
all: cntk torch tf

cntk:
	cd CNTK/lstm && \
	cntk configFile=rnn.cntk configName=lstm deviceId=0 minibatchSize=256 epochSize=8192 maxEpochs=1 makeMode=false && \
	cd ../..

torch:
	cd Torch/lstm && \
	th lstm.lua --seqlen 32 --batchsize 256 --iters 10 --hiddensize '{256,256}' --lstm --startlr 1 --cutoff 5 --maxepoch 1 --cuda --device 1 && \
	cd ../.. 

tf: ~/data/PennTreeBank/ptb.train.txt
	cd TF/lstm && \
	LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		lstm.py --batchsize 256 --iters 10 --seqlen 32 --numlayer 2 \
		--hiddensize 256 --device 0 --data_path ~/data/PennTreeBank/ && \
	cd ../..
