import torch
import os
import numpy as np
from plot import visualize

def inference(model, image, device):
    x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
    mask = model.predict(x_tensor)
    return mask.squeeze().cpu().numpy().round()

def inference_dataloader(
        model,
        test_dataset,
        device,
        ):
    for i in range(len(test_dataset)):
        name = os.path.basename(test_dataset.masks_fps[i])
        print(name)
        # image_vis = test_dataset[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        pr_mask = inference(model, image, device)
        print(image.shape)
        print(gt_mask.shape)
        print(pr_mask.shape)
        print()
        # visualize( 
            # image=image_vis, 
        #     ground_truth_mask=gt_mask.squeeze(), 
        #     # predicted_mask=pr_mask
        #     )