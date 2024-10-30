# CE-TransUnet

This Repository is Code for 'CE-TransUnet: A Convolutional Enhanced Model for Pulmonary Alveolus Pathology Image Segmentation' on ICIC.
Link: [http://poster-openaccess.com/files/icic2024/1949.pdf](http://poster-openaccess.com/files/icic2024/1949.pdf)

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
![](readme_img/1.png)

Open `ce_net.py` and adjust the following values if needed:
![](readme_img/2.png)


## Training
Open `train.py` and modify the following values:
![](readme_img/4.png)
   
`CE_TransUnet` and `CE_TransTest` are optional.
![](readme_img/5.png)

Simply run `train.py` to initiate training.

## Testing
Open `test.py` and make necessary modifications.
![](readme_img/6.png)
![](readme_img/7.png)

Import the weight files into the `params` folder.

Run `test.py` to execute testing.

## Post-training Transformation
Run `res_trans.py`.
