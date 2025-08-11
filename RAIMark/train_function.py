import datetime
import json
import logging
import os
from pathlib import Path
import pytz

from module import *


def initial_config():
    global formatter
    global sh

    timezone = pytz.timezone('Asia/Shanghai')

    logging.Formatter.converter = lambda *args: datetime.datetime.fromtimestamp(args[1], timezone).timetuple()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    for i in [FLOG, FIMG, FSIMG]:
        os.makedirs("./{}".format(i), exist_ok=True)


def train(logger, dataset, model, imgname):
    _lr = this_lr

    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)

    sample = Sampler(dataset.H, dataset.W).to(device)

    imgdata = dataset.imgdata.to(device)

    loss_save = []

    for epoch in range(1, max_epoch + 1):
        logger.info("Epoch {}".format(epoch))

        optimizer.zero_grad()

        output = sample(model)

        loss = torch.nn.functional.mse_loss(output, imgdata, reduction="sum")

        loss.backward()

        optimizer.step()

        logger.info("Epoch {}, loss {:.10f}".format(epoch, loss))

        if loss.item() < cutdown:
            save_func_img(output, "./{}/{}.jpg".format(FIMG, imgname))
            break

        if len(loss_save) > 50:
            if loss.item() > max(loss_save):
                _lr *= lr_decay

                for param_group in optimizer.param_groups:
                    param_group['lr'] = _lr

                logger.info("Change lr to {}".format(_lr))

                loss_save = []

            else:
                loss_save.pop(0)

        loss_save.append(loss.item())


def save_func_img(image, path):
    output = image.squeeze(0).cpu().detach().numpy()

    output = output.transpose(1, 2, 0)

    output = (output + 1) / 2 * 255

    output = np.clip(output, 0, 255)

    output = output.astype(np.uint8)

    if image.shape[1] == 1:
        output = Image.fromarray(output.squeeze(-1), mode="L")

    if image.shape[1] == 3:
        output = Image.fromarray(output, mode="RGB")

    output.save(path)


def train_img(img):
    global cutdown
    global this_lr, this_step

    imgname = Path(img).stem

    logger = logging.getLogger(img)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./{}/{}.log".format(FLOG, imgname), "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    dataset = ImgDataset(H, W, img)
    cutdown = 4 / (10 ** (PSNR / 10)) * dataset.H * dataset.W * dataset.C

    logger.info("Start training {}".format(img))

    hidden_layers = 1
    this_lr = lr / hidden_layers
    this_step = step_lr / hidden_layers
    model = FuncImg(out_channels=dataset.C, hidden_layers=hidden_layers).to(device)

    train(logger, dataset, model, imgname)

    torch.save(model.cpu().state_dict(), "./{}/{}.pth".format(FSIMG, imgname))

    logger.info("Finish training {}".format(img))


def train_folder(folder):
    for i in os.listdir(folder):
        train_img(i)


if __name__ == '__main__':
    global H, W
    global lr, lr_decay, step_lr
    global max_epoch, PSNR, device

    initial_config()

    os.chdir(os.path.dirname(__file__))

    config = json.load(open("./config/train_function.json", "r"))
    H = config["H"]
    W = config["W"]
    lr = config["lr"]
    lr_decay = config["lr_decay"]
    step_lr = config["step_lr"]
    max_epoch = config["max_epoch"]
    PSNR = config["PSNR"]
    device = config["device"]

    train_folder(SIMG)
