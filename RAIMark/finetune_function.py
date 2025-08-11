import datetime
import json
import logging
import os
from pathlib import Path
import pytz
import random
import timm.scheduler.cosine_lr

from module import *


def initial_config():
    global formatter
    global sh

    timezone = pytz.timezone('Asia/Shanghai')

    logging.Formatter.converter = lambda *args: datetime.datetime.fromtimestamp(args[1], timezone).timetuple()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    for i in [FTFUNC, FTLOG]:
        os.makedirs("./{}".format(i), exist_ok=True)


def img_loss(encoded_imgs, imgs):
    return torch.nn.functional.mse_loss(encoded_imgs, imgs, reduction="mean")


def msg_loss(decoded_msgs, msgs):
    return torch.nn.functional.binary_cross_entropy(torch.sigmoid(decoded_msgs), 0.5 * (msgs + 1), reduction="mean")


def train_one_epoch(logger, epoch, model, sampler, imgdata, watermark, optimizer, scheduler):
    logger.info("Epoch {}, {}: Train".format(epoch, sampler))

    scheduler.step(epoch)

    model.funcimg.train()

    for i in range(len(noise_layers)):
        decoded_msg, sampled_img, noised_img, noise = model(sampler)

        loss1 = img_loss(sampled_img, imgdata)
        loss2 = msg_loss(decoded_msg, watermark)

        ber = torch.sum((decoded_msg > 0) != (watermark > 0)) / msg_length

        loss = loss1 * loss_img + loss2 * loss_msg 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info("Epoch {}, image loss {:.6f}".format(epoch, loss1))
        logger.info("Noise {}, message loss {:.6f}, ber {:.6f}".format(repr(noise), loss2, ber))


def eval_one_epoch(logger, epoch, model, sampler, imgdata, watermark, imgname):
    model.funcimg.eval()

    decoded_msgs, sampled_img, noised_imgs, noises = model(sampler, eval=True)

    loss1 = img_loss(sampled_img, imgdata)
    logger.info("Epoch {}, {}: Eval, image loss {:.6f}".format(epoch, sampler, loss1))

    aber = 0

    for i in range(len(noise_layers)):

        loss2 = msg_loss(decoded_msgs[i], watermark)

        ber = torch.sum((decoded_msgs[i] > 0) != (watermark > 0)) / msg_length

        aber += ber

        logger.info("Noise {}, message loss {:.6f}, ber {:.6f}".format(repr(noises[i]), loss2, ber))

    aber /= len(noise_layers)

    logger.info("Average bit error rate {:.6f}".format(aber))

    return aber


def finetune_img(img, samplers, decoder):
    imgname = Path(img).stem

    loader = ImgLoader(imgname, loadwm=False, msg_length=msg_length)

    logger = logging.getLogger(img)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./{}/{}.log".format(FTLOG, imgname), "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Start finetuning {}".format(img))

    funcimg = loader.funcimg

    watermark = loader.watermark.to(device)

    model = Model(funcimg, noise_layers, decoder, msg_length)

    model = model.to(device)

    sampler_and_samples = [(s, s(model.funcimg).clone().detach()) for s in samplers]

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    t_initial = 1
    lr_min = 0

    scheduler = timm.scheduler.cosine_lr.CosineLRScheduler(optimizer, t_initial=t_initial, lr_min=lr_min, warmup_lr_init=lr, warmup_t=5)

    bber, bep = 100, 0

    for epoch in range(1, max_epoch + 1):
        sss = [x for x in sampler_and_samples]
        random.shuffle(sss)

        for _ in range(len(sampler_and_samples)):
            sampler, imgdata = sss.pop()
            train_one_epoch(logger, epoch, model, sampler, imgdata, watermark, optimizer, scheduler)

        if epoch % eval_epoch == 0:
            res = []

            aber = 0, 0

            for sampler, imgdata in sampler_and_samples:
                ber = eval_one_epoch(logger, epoch, model, sampler, imgdata, watermark, imgname)

                aber += ber

            state_dict = funcimg.state_dict()

            save_args = {
                "watermark": loader.watermark,
                "epoch": epoch,
                "state_dict": state_dict
            }

            torch.save(save_args, "{}/{}-{}.pth".format(FTFUNC, imgname, epoch))

            aber /= len(sampler_and_samples)

            if aber < bber:
                bber = aber
                bep = epoch

    logger.info("Finish finetuning {}".format(img))
    logger.info("Best epoch {}, average ber {:.6f}".format(bep, bber / len(noise_layers)))


def finetune_folder(folder):
    decoder = Decoder(decoder_max_msg)
    decoder.load_state_dict(torch.load("./{}/{}.pth".format(DMODEL, decoder_name), map_location="cpu"))

    decoder.eval()

    for param in decoder.parameters():
        param.requires_grad = False

    samplers = [Sampler(H, W).to(device) for H, W in samples]

    for i in os.listdir(folder):
        finetune_img(i, samplers, decoder)


if __name__ == '__main__':
    global samples, decoder_max_msg
    global msg_length, lr
    global loss_img, loss_msg
    global decoder_name, eval_epoch
    global max_epoch, noise_layers, device

    initial_config()

    os.chdir(os.path.dirname(__file__))

    config = json.load(open("./config/finetune_function.json", "r"))
    samples = config["samples"]
    decoder_max_msg = config["decoder_max_msg"]
    msg_length = config["msg_length"]
    lr = config["lr"]
    loss_img, loss_msg = config["loss_weight"]
    decoder_name = config["decoder_name"]
    eval_epoch = config["eval_epoch"]
    max_epoch = config["max_epoch"]
    noise_layers = config["noise_layers"]
    device = config["device"]

    finetune_folder(FSIMG)
