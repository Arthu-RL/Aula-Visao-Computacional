import torch
import numpy as np

from cv2 import cvtColor, COLOR_RGB2BGR

from general import non_max_suppression_kpt
from plots import output_to_keypoint, plot_skeleton_kpts

# By https://stackabuse.com/real-time-pose-estimation-from-video-in-python-with-yolov7/
def draw_keypoints(model,  model_output, image): 
    output = non_max_suppression_kpt(output, 
                                    0.30, # Confidence Threshold
                                    0.65, # IoU Threshold
                                    nc=model.yaml['nc'], # Number of Classes
                                    nkpt=model.yaml['nkpt'], # Number of Keypoints
                                    kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
        
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cvtColor(nimg, COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)  

    return nimg