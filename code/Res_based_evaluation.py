import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from eval_metric.metrics_classification import *
import glob
import time
from pretrained_models.models import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--arch', default='res_50', type=str,
        choices=['res_50', 'moco_v3_res_50', 'dino_res_50' ],
        help="""Name of the pretrained architecture to evaluate """)
    parser.add_argument('--data_path', default='../storage/imagenet_1000samples/', type=str,
        help='Please specify path to the imagenet_1000samples folder.')
    parser.add_argument('--output_dir', default="../storage/quant_results/", type=str, help='Path to save the 1000 saliency maps.')
    parser.add_argument('--saliency', default=False, type=bool, help="""Wether you want as outputs the saliency maps for qualitative evaluation. If you want the saliency maps change this parameter to True""")
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':

    args = get_args()
    
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}

    if args.arch in models_dict.keys():
        model = models_dict[args.arch]
    
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    target_layers = [model.layer4[-1]]


    klen = 11
    ksig = 5
    kern = gkern(klen, ksig) #parameters we need for Insertion metric
    
    label, cor_pred, cor_pred2, un_prob, ma_prob, av_dr, f, t, de, ina  = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 # initialize parameters needed for quantitative metrics
    samples = 1000 # the number of samples 
    
    
    
    for img in sorted(glob.glob(args.data_path + '*.JPEG')):  
        start = time.time()
    # image load and preprocess
        rgb_img = cv2.imread(img)
        rgb_img = cv2.resize(rgb_img, (224, 224)) 
    
        rgb_img = np.float32(rgb_img) / 255 
        input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])           
        
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
        target_category = label
    
    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:

            grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)
            
            grayscale_cam = grayscale_cam[0, :]
        
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        if args.saliency == True:
           cv2.imwrite(f'{args.output_dir}{args.method}_category'+ str(label) +'.jpg', cam_image)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        
        exp_map = np.multiply(rgb_img,cam_mask) #creating the explanation map
        masked_tensor = preprocess_image(exp_map,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        if args.use_cuda:
            device = 'cuda'
            input_tensor=input_tensor.cuda()
            blur = lambda x: nn.functional.conv2d(x.to("cpu"), kern, padding=klen//2).to("cuda")
        else:
            device = 'cpu'
            blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    
        # Calculation of model accuracy for for masked and unmasked input
        outputs, outputs_mask = return_probs(model, input_tensor, masked_tensor, device)
        if outputs[1][0][0].item() == label:
            cor_pred = cor_pred + 1
        if outputs_mask[1][0][0].item() == label:
            cor_pred2 = cor_pred2 + 1
        # Calculation of average conf for unmasked and masked input given the target class 
        prob1, prob2 = return_probs2(model, input_tensor, torch.tensor(target_category), masked_tensor, device)
        un_prob = un_prob + prob1
        ma_prob = ma_prob + prob2
        # Calculation of average drop and increase metric
        avg_drop, avg_inc = averageDropIncrease(model, input_tensor, torch.tensor(target_category), masked_tensor, device)
        av_dr = av_dr + avg_drop
        if avg_inc == False:
            f=f+1
        else:
            t=t+1 
        # Calculation of deletion score
        deletion = CausalMetric(model, 'del', 224, substrate_fn=torch.zeros_like)
        del_scores = deletion.single_run(input_tensor, grayscale_cam, input_tensor.shape[2]*input_tensor.shape[3], verbose=0)
        del_auc = auc(del_scores)
        de = de + del_auc
        # Calculation of insertion score
        insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
        in_scores = insertion.single_run(input_tensor, grayscale_cam, input_tensor.shape[2]*input_tensor.shape[3], verbose=0)
        in_auc = auc(in_scores)
        ina = ina + in_auc
        
        #print("We just end with the", label, "sample")
        #print(f"Device = {device}; Time for this sample was: {(time.time() - start)/3:.3f} seconds")
        label = label + 1
    print(103 * '*')   
    print('                                         QUANTITATIVE EVALUATION                                       ')
    print("The model accuracy for unmasked input is", round(cor_pred/samples,2))
    print("The model accuracy for masked input is", cor_pred2/samples)
    print("The average conf for unmasked input and given class is", np.round(un_prob/samples,2)[0])
    print("The average conf for masked input and given class is", np.round(ma_prob/samples,2)[0])
    print("The average drop for", samples,"samples is", np.round(av_dr/samples,2)[0])
    if f == samples:
        print("The average increase is 0")
    else:
        print("The average increase is", np.round(t/samples,2))
    
    print("The AUC score for deletion is", round(de/samples,2),"and for insertion", round(ina/samples,2))
    print(103 * '*')
    print(f"Device = {device}; Time for this experiment was: {(time.time() - start)/3:.3f} seconds")
    
    