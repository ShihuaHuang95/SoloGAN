## SoloGAN
* Unsupervised Multimodal Image Translation using a Single Generative Adversarial Network

# Example Usage
Testing
-
First, download the pretrained models and put them in ./checkpoints/

Runing the following command to translate edges to shoes&handbags

python ./test.py --name edges_shoes&handbags --d_num 2 

Then the results are stored in ./checkpoints/edges_shoes&handbags/edges_shoes&handbags_results directory. By default, it produce 10 random translation outputs.

Training
1. Download the dataset you want to use and move to ./datasets. For example, you can use the horse2zebra dataset provided by CycleGAN.
2. Start training:
	python ./train.py --name horse2zebra --d_num 2

3. Intermediate image outputs and model binary files are stored in ./checkpoints/horse2zebra/web	
