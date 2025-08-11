# Achieving Resolution-Agnostic DNN-based Image Watermarking: A Novel Perspective of Implicit Neural Representation

This folder is the code for the paper. The environment of experiments are:

| Packages        | Version      |
| --------------- | ------------ |
| python          | 3.12.2       |
| pytorch         | 2.2.1        |
| torchvision     | 0.17.1       |
| pytz            | 2023.3.post1 |
| timm            | 0.9.16       |
| torch-optimizer | 0.3.0        |
| kornia          | 0.7.1        |
| augly           | 1.0.0        |


The following shows how to run the code:

1. Generate dataset

```
python generate.py
```

Create image symbolic link to certain folders in `./result`. Change the path to the dataset on your server.

2. Train INR (Stage 1)

```
python train_function.py
```

The configuration file is `./config/train_function.json`. Log files, fitted INRs and images sampled by fitted INR are in `./result/function-log`, `./result/function-space` and `./result/function-imgs`.

3. Pre-train Watermark Decoder (Stage 2)

```
python train_decoder.py
```

The configuration file is `./config/train_decoder.json`. Log files and trained decoder are in `./result/decoder-log` and `./result/decoder-model`.

4. Fine-tune INR (Stage 3)

```
python finetune_function.py
```

The configuration file is `./config/finetune_function.json`. Log files and fine-tuned INRs are in `./result/finetune-log` and `./result/finetune-function`.

5. Test INR

```
python test_function.py
```

The configuration file is `./config/test_function.json`. Log files and result images are in `./result/test-log` and `./result/test-imgs`.

The `./result` folder is auto-generated after running commands.

