import torch
import numpy as np

from cv2 import rectangle, cvtColor, COLOR_RGB2BGR, LINE_AA

from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

# By https://stackabuse.com/real-time-pose-estimation-from-video-in-python-with-yolov7/
def draw_keypoints(model,  model_output, image): 
    model_output = non_max_suppression_kpt(model_output, 
                                    0.30, # Confidence Threshold
                                    0.65, # IoU Threshold
                                    nc=model.yaml['nc'], # Number of Classes
                                    nkpt=model.yaml['nkpt'] , # Number of Keypoints
                                    kpt_label=True)
    
    model_output = output_to_keypoint(model_output)

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cvtColor(nimg, COLOR_RGB2BGR)
    for idx in range(model_output.shape[0]):
        plot_skeleton_kpts(nimg, model_output[idx, 7:].T, 3)  

        xmin, ymin = (model_output[idx, 2]-model_output[idx, 4]/2), (model_output[idx, 3]-model_output[idx, 5]/2)
        xmax, ymax = (model_output[idx, 2]+model_output[idx, 4]/2), (model_output[idx, 3]+model_output[idx, 5]/2)
        rectangle(
            nimg,
            (int(xmin), int(ymin)),
            (int(xmax), int(ymax)),
            color=(255, 0, 0),
            thickness=1,
            lineType=LINE_AA
        )
        
    return nimg