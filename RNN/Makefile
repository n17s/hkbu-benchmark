
all: cntk torch tf
CNTK17=~/CNTK/bin/cntk
CNTK15=~/cntk/cntk/bin/cntk
CNTK=${CNTK17}

#128 sequences * 32 = 4096
#128 sequences * 64 = 8192 

#256 sequences * 32 = 8192
#256 sequences * 64 = 16384

#512 sequences * 32 = 16384
#512 sequences * 64 = 32768
cntk:
	cd CNTK/lstm && \
	${CNTK} configFile=rnn.cntk configName=lstm deviceId=1 minibatch=16384 epochSize=4096 maxEpochs=3 makeMode=false trainFile="ptb.train.32.ctf" && \
	${CNTK} configFile=rnn.cntk configName=lstm deviceId=1 minibatch=32768 epochSize=8192 maxEpochs=3 makeMode=false trainFile="ptb.train.64.ctf" && \
	cd ../..

tf:
	cd TF/lstm && \
	LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		lstm.py --batchsize 128 --iters 10 --seqlen 32 --numlayer 2 \
		--hiddensize 256 --device 1 --data_path ~/data/PennTreeBank/ && \
	LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		lstm.py --batchsize 128 --iters 10 --seqlen 64 --numlayer 2 \
		--hiddensize 256 --device 1 --data_path ~/data/PennTreeBank/ && \
	cd ../..
