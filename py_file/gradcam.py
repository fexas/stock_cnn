import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
# import torch.enable_grad
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
import csv

def runGradCam(model, images_scores, target_class, results_path='./'):
    """
    Utilizes the package 'pytorch-grad-cam' which can be installed via "pip install grad-cam".
    Further encapsulates this package for use within our project, enhancing functionality and integration.
    """
    i = 1
    for img_score in images_scores:
        score, img = img_score

        save_path = os.path.join(results_path, f'top{i}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Save the probability score
        with open(os.path.join(save_path, 'probability.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)   
            writer.writerow([score])
        
        # Save the original image
        plt.imshow(img.squeeze().cpu(), cmap='gray')
        savepath = os.path.join(save_path, 'origin_image.png')
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close()

        if len(img.shape) == 2:  
            img = img.unsqueeze(0)  # (C, H, W)
            
        # Prepare the image for Grad-CAM
        img_tensor = img.unsqueeze(0).to(next(model.parameters()).device)  # Ensure the image is on the same device as the model
        img_bw = img.squeeze().cpu().numpy()
        img_rgb = np.stack([img_bw, img_bw, img_bw], axis=-1) 
        img_normalized = np.float32(img_rgb) / 255

        # Specify the target layers (layer1, layer2, layer3)
        # target_layers = [model.layer1, model.layer2, model.layer3]
        target_layers = [
            model.layer1[0],
            model.layer2[0], 
            model.layer3[0]
        ]
        for idx, module in enumerate(target_layers, start=1):
            cam = GradCAM(model=model, target_layers=[module])
            targets = [ClassifierOutputTarget(target_class)]       
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
            
            # Grad-CAM visualization with the original image
            visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
            plt.imshow(visualization)
            plt.axis('off')
            savepath = os.path.join(save_path, f'gradcam_layer{idx}_with_origin_image.png')
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Grad-CAM visualization without the original image
            visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True, image_weight=0.0)
            plt.imshow(visualization)
            plt.axis('off')
            savepath = os.path.join(save_path, f'gradcam_layer{idx}.png')
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
            plt.close()
        i += 1