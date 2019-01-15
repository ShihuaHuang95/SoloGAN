import os
from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from util.visualizer import save_images
from itertools import islice
from models.solver import SoloGAN
from util import html, util
import torch

opt = TestOptions().parse()
opt.n_samples = 5
opt.how_many = 200
opt.isTrain = False

opt.dataroot = '{}/{}'.format(opt.dataroot, opt.name)

opt.no_flip = True
opt.batchSize = 1

data_loader = CreateDataLoader(opt)

model = SoloGAN()
model.initialize(opt)

img_dir = '{}/images'.format(opt.results_dir)
if os.path.isdir(img_dir):
    for file in os.listdir(img_dir):
        os.remove("{}/{}".format(img_dir, file))

web_dir = os.path.join(opt.results_dir)

webpage = html.HTML(web_dir, 'task {}'.format(opt.name))

if opt.name == 'cat_dog_tiger':
    domain_names = ['cat', 'dog', 'tiger']

elif opt.name == 'photo2arts':
    domain_names = ['photo', 'Vangh', 'Monet']
else:
    domain_names = opt.name.split('_')


def test():
    for i, data in enumerate(islice(data_loader, opt.how_many)):
        print('process input image %3.3d/%3.3d' % (i, opt.how_many))
        with torch.no_grad():
            all_images, all_names = model.translation(data, domain_names)
        img_path = 'image%3.3i' % i
        save_images(webpage, all_images, all_names, img_path, None, width=opt.img_size)
    webpage.save()


if __name__ == '__main__':
    test()
