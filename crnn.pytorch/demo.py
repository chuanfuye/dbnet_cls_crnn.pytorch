import argparse
import time

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch.autograd import Variable

import lib.config.alphabets as alphabets
import lib.models.crnn as crnn
import lib.utils.utils as utils


def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/icdar2015_config.yaml')
    parser.add_argument('--image_path', type=str, default='/home/wt_yechuanfu/work_code/pytorchOCR/icdar2015/test/word_1.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/icdar2015/crnn/checkpoints/checkpoint_best.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w, c = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w, c = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    input = torch.ones(1, 3, 32, 160).cuda(0)
    traced_scipt_model = torch.jit.trace(model, input)
    traced_scipt_model.save("crnn.pt")
    preds = model(img)
    print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))

if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()

    img = cv2.imread(args.image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    recognition(config, img, model, converter, device)

    finished = time.time()
    print('elapsed time: {0}'.format(finished - started))

