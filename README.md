# SoloGAN
[Unsupervised Multimodal Image Translation using a Single Generative Adversarial Network](https://arxiv.org/pdf/1901.03353.pdf)

### Results

Cat ↔ Dog:
<p align="center">
<img src='images/cat.jpg'  width='18%' /><img src='images/cat2dog.gif'   width='18%' />
<img src='images/dog.jpg'  width='18%'/><img src='images/dog2cat.gif'   width='18%'/>
</p>
Label ↔ Facade:
<p align="center">
<img src='images/label.jpg'  width='18%' /><img src='images/label2facade.gif'   width='18%' />
<img src='images/facade.jpg'  width='18%'/><img src='images/facade2label.gif'   width='18%'/>
</p>
Edge ↔ Shoes:
<p align="center">
<img src='images/edge.jpg'  width='18%' /><img src='images/edge2shoe.gif'   width='18%' />
<img src='images/shoe.jpg'  width='18%'/><img src='images/shoe2edge.gif'   width='18%'/>
</p>

## Usage Guidance
### Testing

* Downloading the pretrained models and put them in ./checkpoints/ from
[Google Drive](https://drive.google.com/drive/u/1/folders/1ipVSrr-0dAJKqHbqFw7Y8sqfF3GXq5XN) or
[Baidu Yun](https://pan.baidu.com/s/1HixWmTob0uU0TjwZAWqKFg)

* Runing the following command to translate edges to shoes&handbags:
> python ./test.py --name edges_shoes&handbags --d_num 2

Then the translated samples are stored in ./checkpoints/edges_shoes&handbags/edges_shoes&handbags_results directory.
By default, it produce 5 random translation outputs.

### Training

* Download the dataset you want to use and move to ./datasets. For example, you can use the horse2zebra dataset provided by [CycleGAN][1].
* Start training with the following command:
> python ./train.py --name horse2zebra --d_num 2

Intermediate image outputs and model binary files are stored in ./checkpoints/horse2zebra/web

#### bibtex
If this work help to easy your research, please cite the corresponding paper :
```
@inproceedings{huangsologan,
	title={[Unsupervised Multimodal Image Translation using a Single Generative Adversarial Network},
	author={Shihua, Huang and Cheng, He and Yuli, Zhang and Ran Cheng},
	booktitle={},
	year={2019}
 }
 ```

### Acknowledgment

The code used in this research is based on [SingleGAN](https://github.com/Xiaoming-Yu/SingleGAN)

### Concat

Feeling free to reach me if there is any questions (huangsh6@mail.sustc.edu.cn)


[1]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix "CycleGAN"
