import datetime
import json
import logging
import os
import pytz
import kornia

from module import *



def initial_config():
    global formatter
    global sh

    timezone = pytz.timezone('Asia/Shanghai')

    logging.Formatter.converter = lambda *args: datetime.datetime.fromtimestamp(args[1], timezone).timetuple()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    for i in [TIMG, TLOG]:
        os.makedirs("./{}".format(i), exist_ok=True)


def img_loss(encoded_imgs, imgs):
    return torch.nn.functional.mse_loss(encoded_imgs, imgs, reduction="mean")


def msg_loss(decoded_msgs, msgs):
    return torch.nn.functional.binary_cross_entropy(torch.sigmoid(decoded_msgs), 0.5 * (msgs + 1), reduction="mean")


def eval_one_epoch(logger, model, sampler, imgdata, watermark, imgname):
    model.funcimg.eval()

    with torch.no_grad():
        decoded_msgs, sampled_img, noised_imgs, noises = model(sampler, eval=True)

    save_func_img(imgdata, "{}/{}-im-{}x{}.jpg".format(TIMG, imgname, sampler.H, sampler.W))
    save_func_img(sampled_img, "{}/{}-wm-{}x{}.jpg".format(TIMG, imgname, sampler.H, sampler.W))

    loss1 = img_loss(sampled_img, imgdata)
    psnr = -kornia.losses.psnr_loss(sampled_img, imgdata, 2)
    ssim = 1 - 2 * kornia.losses.ssim_loss(sampled_img, imgdata, 5)
    logger.info("Eval {}, image loss {:.6f}, psnr {:.6f}, ssim {:.6f}".format(sampler, loss1, psnr, ssim))

    aber = 0

    for i in range(len(noise_layers)):

        loss1 = msg_loss(decoded_msgs[i], watermark)

        ber = torch.sum((decoded_msgs[i] > 0) != (watermark > 0)) / msg_length

        aber += ber

        error_pos = torch.where((decoded_msgs[i] > 0) != (watermark > 0))[1].tolist()

        if ber > 0:
            logger.info("Noise {}, message loss {:.6f}, ber {:.6f}, error pos {}".format(repr(noises[i]), loss1, ber, error_pos))
        else:
            logger.info("Noise {}, message loss {:.6f}, ber {:.6f}".format(repr(noises[i]), loss1, ber))

    aber /= len(noise_layers)

    logger.info("Average bit error rate {:.6f}".format(aber))

    return aber
        

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


def test_img(log_name):
    imgname = log_name.split("-")[0]

    bepoch = 0
    bimgloss = 0
    bber = 1

    s_imgloss = 0
    s_ber = 0
    s_epoch = 0

    for i in open("./{}/{}.log".format(FTLOG, log_name), "r").readlines():
        if "Eval" in i:
            epoch = int(i.split()[-7][:-1])

            if epoch != s_epoch:
                if s_epoch and s_imgloss < 4 / (10 ** (35 / 10)) and s_ber < bber or (s_ber == bber and s_imgloss < bimgloss):
                    bber = s_ber
                    bepoch = epoch
                    bimgloss = s_imgloss

                s_epoch = epoch
                s_ber = 0
                s_imgloss = float(i.split()[-1])
            else:
                s_imgloss += float(i.split()[-1])

        if "Average" in i:
            s_ber += float(i.split()[-1])

    if s_epoch and s_ber < bber or (s_ber == bber and s_imgloss < bimgloss):
        bber = s_ber
        bepoch = epoch
        bimgloss = s_imgloss

    loader = ImgLoader(imgname, loadwm="{}/{}-{}.pth".format(FTFUNC, imgname, bepoch), msg_length=msg_length)

    logger = logging.getLogger(imgname)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./{}/{}.log".format(TLOG, imgname), "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("Best epoch {}".format(bepoch))

    funcimg = loader.funcimg.to(device)
    funcimg_wm = loader.funcimg_wm

    watermark = loader.watermark.to(device)

    samplers = [Sampler(H, W).to(device) for H, W in samples]

    with torch.no_grad():
        sampler_and_samples = [(s, s(funcimg).clone().detach()) for s in samplers]

    decoder = Decoder(decoder_max_msg)
    decoder.load_state_dict(torch.load("./{}/{}.pth".format(DMODEL, decoder_name), map_location="cpu"))

    model = Model(funcimg_wm, noise_layers, decoder, msg_length)

    model = model.to(device)

    model.eval()

    for sampler, imgdata in sampler_and_samples:
        eval_one_epoch(logger, model, sampler, imgdata, watermark, imgname)


def test_folder():
    for i in os.listdir(FTLOG):
        log_name = i.split(".")[0]
        test_img(log_name)


if __name__ == '__main__':
    global samples, decoder_max_msg
    global msg_length, decoder_name
    global noise_layers, device

    initial_config()

    os.chdir(os.path.dirname(__file__))

    config = json.load(open("./config/test_function.json", "r"))
    samples = config["samples"]
    decoder_max_msg = config["decoder_max_msg"]
    msg_length = config["msg_length"]
    decoder_name = config["decoder_name"]
    noise_layers = config["noise_layers"]
    device = config["device"]

    test_folder()
