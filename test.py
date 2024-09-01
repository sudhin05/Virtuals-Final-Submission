import dataloader_paligemma
import tensorflow as tf
ob=dataloader_paligemma.MyDataset(image_dir='trail_data/images',label_dir='trail_data/labels')
print(type(ob))
