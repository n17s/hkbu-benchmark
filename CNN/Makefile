all: tf cntk

CNTK/imagenet_data.txt:
	cd CNTK && \
	python createFakeImageNetDataForCNTKCNN.py && \
	cd ..

CNTK/labelmap.txt:
	cd CNTK && \
	python createLabelMapForCNTKCNN.py && \
	cd ..

cntk: CNTK/imagenet_data.txt CNTK/labelmap.txt
	cd CNTK/alexnet && \
	cntk configFile=alexnet.cntk configName=alexnet deviceId=1 minibatchSize=16 epochSize=2560 maxEpochs=4 && \
	cd ../resnet && \
	cntk configFile=resnet.cntk configName=resnet deviceId=1 minibatchSize=16 epochSize=2560 maxEpochs=4 && \
	cd ../..

tf:
	cd TF/alexnet && \
	LD_LIBRARY_PATH=${HOME}/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		alexnet.py -e 4 -b 16 -i 2 -d 1 && \
	cd ../resnet && \
	LD_LIBRARY_PATH=${HOME}/cuda/lib64:/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		resnet.py -e 4 -b 16 -i 2 -d 1 && \
	cd ../..
