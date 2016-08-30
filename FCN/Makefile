all: tf cntk

CNTK/data26752_4k.txt:
	cd CNTK && \
	python createDataForCNTKFCN.py && \
	cd ..

CNTK/labelmap26752.txt:
	cd CNTK && \
	python createLabelMapForCNTKFCN.py && \
	cd ..

cntk: CNTK/labelmap26752.txt CNTK/data26752_4k.txt
	cd CNTK/fcn5 && \
	cntk configFile=fcn5.cntk configName=fcn5 deviceId=0 minibatchSize=64 && \
	cd ../fcn8 && \
	cntk configFile=fcn8.cntk configName=fcn8 deviceId=0 minibatchSize=64 && \
	cd ../..

tf:
	cd TF/fcn5 && \
	LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		fcn5.py -e 2 -b 64 -i 4 -d 1 && \
	cd ../fcn8 && \
	LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/usr/local/cuda-7.5/extras/CUPTI/lib64 python \
		fcn8.py -e 2 -b 64 -i 4 -d 1 && \
	cd ../..
