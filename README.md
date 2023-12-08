# Facial Recognition with Siamese Network
Course project for the course Deep Learning for Visual Recognition


## How to download CelebA dataset
One of the cells in Triplet_net.ipynb automatically downloads the CelebA images, labels etc., 
but in case the download limit has been hit, use the following instructions:

- Go to the Google Drive with the data:
https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing

- Download img/img_align_celeba.zip, and all txt files from Anno/ and Eval/

- Place the zip and all txt files in a folder data/celeba/

- In order to get the cleaned data identities, go to:
https://github.com/C3Imaging/Deep-Learning-Techniques/tree/clean-celebA

- Download the two text files and put them into data/celeba/

- Everything should work now. The images automatically gets extracted when the Triplet_net.ipynb notebook runs the first time
