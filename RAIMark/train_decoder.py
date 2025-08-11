import datetime
import json
import logging
import os
import pytz
from torch.utils.data import DataLoader
import torch_optimizer
import timm.scheduler.cosine_lr

from module import *


def initial_config():
    for i in [DLOG, DMODEL]:
        os.makedirs("./{}".format(i), exist_ok=True)

    timezone = pytz.timezone('Asia/Shanghai')
    logging.Formatter.converter = lambda *args: datetime.datetime.fromtimestamp(args[1], timezone).timetuple()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("{}/{}.log".format(DLOG, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]), "w"),
            logging.StreamHandler()
        ])
    

def img_loss(encoded_imgs, imgs):
    return torch.nn.functional.mse_loss(encoded_imgs, imgs, reduction="mean")


def msg_loss(decoded_msgs, msgs):
    return torch.nn.functional.binary_cross_entropy(torch.sigmoid(decoded_msgs), 0.5 * (msgs + 1), reduction="mean")
    

def train_one_epoch(epoch, model, loader, optimizer, scheduler):
    logging.info("Epoch {}: Train".format(epoch))

    scheduler.step(epoch)

    model.train()

    total = 0
    timgloss = 0
    tmsgloss = 0

    for imgs in loader:
        model.module.noise.sample()

        imgs = imgs.to(devices[0])
        msgs = torch.Tensor(np.random.choice([-1, 1], (imgs.shape[0], msg_length))).to(imgs.device)

        decoded_msgs, encoded_imgs, noised_imgs = model(imgs, msgs)

        loss1 = img_loss(encoded_imgs, imgs)
        loss2 = msg_loss(decoded_msgs, msgs)

        loss = loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += imgs.shape[0]
        timgloss += loss1.item() * imgs.shape[0]
        tmsgloss += loss2.item() * imgs.shape[0]

    logging.info("LR: {:.6f}".format(optimizer.param_groups[0]["lr"]))
    logging.info("Img Loss: {:.6f}".format(timgloss / total))
    logging.info("Msg Loss: {:.6f}".format(tmsgloss / total))


def eval_one_epoch(epoch, model, loader):
    logging.info("Epoch {}: Eval".format(epoch))

    model.eval()

    with torch.no_grad():
        for noise in model.module.noise.noises:
            total = 0
            timgloss = 0
            tmsgloss = 0
            tber = 0

            model.module.noise.noise = noise

            for imgs in loader:
                imgs = imgs.to(devices[0])
                msgs = torch.Tensor(np.random.choice([-1, 1], (imgs.shape[0], msg_length))).to(imgs.device)

                decoded_msgs, encoded_imgs, noised_imgs = model(imgs, msgs)

                loss1 = img_loss(encoded_imgs, imgs)
                loss2 = msg_loss(decoded_msgs, msgs)

                ber = torch.sum((decoded_msgs > 0) != (msgs > 0)) / torch.numel(msgs)

                total += imgs.shape[0]
                timgloss += loss1.item() * imgs.shape[0]
                tmsgloss += loss2.item() * imgs.shape[0]
                tber += ber * imgs.shape[0]

            logging.info("Img Loss: {:.6f}".format(timgloss / total))
            logging.info("Msg Loss: {:.6f}, BER for {}: {:.6f}".format(tmsgloss / total, repr(noise), tber / total))


def train():
    global msg_length
    model = EncoderDecoder(msg_length, weight, noises)
    model = torch.nn.DataParallel(model, device_ids=devices).to(devices[0])

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = 0.01

    if train_continue_epoch:
        model.module.encoder.load_state_dict(torch.load("{}/E_{}.pth".format(DMODEL, train_continue_epoch), map_location="cpu"))
        model.module.decoder.load_state_dict(torch.load("{}/D_{}.pth".format(DMODEL, train_continue_epoch), map_location="cpu"))

        for param in model.parameters():
            param.requires_grad = False
        
        model = model.eval()

    optimizer = torch_optimizer.Lamb(model.parameters(), lr=1e-6 if train_continue_epoch else lr * batch_size / 512)
    scheduler = timm.scheduler.cosine_lr.CosineLRScheduler(optimizer, t_initial=30000, lr_min=1e-6, warmup_lr_init=1e-6, warmup_t=5)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((H, W)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = Dataset(DTPATH, train_transform, H, W)
    valset = Dataset(DVPATH, val_transform, H, W)

    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size * 4, num_workers=8, shuffle=False, pin_memory=True)
    
    for epoch in range(train_continue_epoch + 1, train_continue_epoch + max_epoch + 1):
        train_one_epoch(epoch, model, trainloader, optimizer, scheduler)

        if epoch % eval_epoch == 0:
            eval_one_epoch(epoch, model, valloader)
            
            torch.save(model.module.encoder.state_dict(), "{}/E_{}.pth".format(DMODEL, epoch))
            torch.save(model.module.decoder.state_dict(), "{}/D_{}.pth".format(DMODEL, epoch))



if __name__ == '__main__':
    global H, W
    global msg_length, lr, weight
    global train_continue_epoch, eval_epoch, max_epoch
    global noises, devices

    initial_config()

    os.chdir(os.path.dirname(__file__))

    config = json.load(open("./config/train_decoder.json", "r"))
    H = config["H"]
    W = config["W"]
    msg_length = config["msg_length"]
    weight = config["encode_weight"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    train_continue_epoch = config["train_continue_epoch"]
    eval_epoch = config["eval_epoch"]
    max_epoch = config["max_epoch"]
    noises = config["noise_layers"]
    devices = config["devices"]

    train()
