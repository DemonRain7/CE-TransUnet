# CE-TransUnet

This Repository is Code for 'CE-TransUnet: A Convolutional Enhanced Model for Pulmonary Alveolus Pathology Image Segmentation' on ICIC.

You could follow the instruction below to reproduce our project:

## Dataset Configuration
Under the `data` folder, the directory format is as follows:
- The `JPEGImages` folder stores original images.
- The `mask_input` folder stores mask images.
Original and mask images share the same filenames.

## Preprocessing
Run `mask_input_trans.py` to convert pixel value 255 in mask images to 1.

## Configuration Adjustment
Open `utils.py` and modify the input image size, preferably in multiples of 224x224.
![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/73ddb586-5fc8-447b-b49f-032217e60f0d)

Open `ce_net.py` and adjust the following values if needed:
![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/c78e3b36-3a34-487b-8962-09008f6a9342)

Some reference values:

![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/87eddbb9-059f-4524-b9eb-7e6b4aa3eff0)

- `embed_dim=96` corresponds to `num_heads(3, 6, 12, 24)`.
- `embed_dim=128` corresponds to `num_heads(4, 8, 16, 32)`.
`depths` refers to the number of CE_Transformer_Blocks.

## Training
Open `train.py` and modify the following values:
![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/172e9d15-7c47-459d-9536-5947c3e57cc4)
   
`CE_TransUnet` and `CE_TransTest` are optional.
![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/90dce3aa-1a5f-453f-b9d0-114f007d8c4b)

Simply run `train.py` to initiate training.

## Testing
Open `test.py` and make necessary modifications.
![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/a6ac9f4d-bb0f-4eea-b8df-263644642396)
![image](https://github.com/DemonRain7/CE-TransUnet/assets/102237492/938880f8-9443-4399-9c10-1d0bb92b7973)

Import the weight files into the `params` folder.

Run `test.py` to execute testing.

## Post-training Transformation
Run `res_trans.py`.
