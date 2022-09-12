import argparse
import cv2
import numpy as np
import torch

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
    preprocess_image
from eval_metric.metrics_classification import *
from pretrained_models.models import *
from deit_attention.deit_attn import *
import glob
import time

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
    parser.add_argument('--arch', default='deit_base', type=str,
        choices=['deit_base', 'moco_v3_deit_base'],
        help="""Name of the pretrained architecture to evaluate """)
    parser.add_argument('--data_path', default='../storage/imagenet_1000samples/', type=str,
        help='Please specify path to the imagenet_1000samples folder.')
    parser.add_argument('--output_dir', default="../storage/quant_transf_res/", type=str, help='Path to save the 1000 saliency maps.')
    parser.add_argument('--saliency', default=False, type=bool, help="""Wether you want as outputs the saliency maps for qualitative evaluation. If you want the saliency maps change this parameter to True""")
    parser.add_argument('--method', type=str, default='raw', choices=['raw', 'rollout', 'gradcam', 'gradcam++', 'scorecam', 'xgradcam'], help='Can be raw/rollout/gradcam/gradcam++/scorecam/xgradcam')
    parser.add_argument("--power", default=1.0, type=float, help="By changing this parameter you can reveal the next more salient regions (power < 1.0).The more you reduce the more regions you uncover.")
    parser.add_argument("--layer", default=11, type=int, help="This is the layer from which we generate the saliency map. It can be an integer from 0 to 11 in order to choose one of the 12 layers. When choosing rollout method this argument is not taken into account.")
    parser.add_argument("--roll_from", default=0, type=int, help="The combination --roll_from 0 --roll_to 11 means the standard rollout but if you want for instance rollout for the last 3 layers of the model you choose --roll_from 9 --roll_to 11. When using raw method this argument is not taken into account.")
    parser.add_argument("--roll_to", default=11, type=int, help="The combination --roll_from 0 --roll_to 11 means the standard rollout but if you want for instance rollout for the last 3 layers of the model you choose --roll_from 9 --roll_to 11. When using raw method this argument is not taken into account.")
    args = parser.parse_args()
    
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=14, width=14): 
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':

    args = get_args()
    # The Cam-based methods you can use
    cam_methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "xgradcam": XGradCAM}


    if args.arch in models_dict.keys():
        model = models_dict[args.arch]  
    
    target_layers = [model.blocks[-1].norm1] 

    
    if args.method in list(cam_methods.keys()) :
       cam = cam_methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    klen = 11
    ksig = 5
    kern = gkern(klen, ksig) #parameters we need for Insertion metric
    
    label, cor_pred, cor_pred2, un_prob, ma_prob, av_dr, f, t, de, ina  = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 # initialize parameters needed for quantitative metrics
    samples = 1000 # number of samples

    for img in sorted(glob.glob(args.data_path + '*.JPEG')):
        start = time.time()
        rgb_img = cv2.imread(img)
        rgb_img = cv2.resize(rgb_img, (224, 224)) 
    
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        if args.use_cuda:
            model = model.cuda()
            input_tensor = input_tensor.cuda()
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
        target_category = label
        if args.method in list(cam_methods.keys()) :
           saliency_map = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=args.aug_smooth, eigen_smooth=args.eigen_smooth)
           saliency_map = saliency_map[0, :]
        else:  
           saliency_map = final_deit_attention(model, input_tensor, method=args.method, layer=args.layer, power=args.power, roll_from=args.roll_from, roll_to=args.roll_to)
        
        sal_image = show_cam_on_image(rgb_img, saliency_map, use_rgb=True)
        
        # sal_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        sal_image = cv2.cvtColor(sal_image, cv2.COLOR_RGB2BGR)
        if args.saliency == True:
               cv2.imwrite(f'{args.output_dir}{args.method}_category'+ str(label) +'.jpg', sal_image)
        cam_mask = cv2.merge([saliency_map, saliency_map, saliency_map])
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
        del_scores = deletion.single_run(input_tensor, saliency_map, input_tensor.shape[2]*input_tensor.shape[3], verbose=0)
        del_auc = auc(del_scores)
        de = de + del_auc
        # Calculation of insertion score
        insertion = CausalMetric(model, 'ins', 224, substrate_fn=blur)
        in_scores = insertion.single_run(input_tensor, saliency_map, input_tensor.shape[2]*input_tensor.shape[3], verbose=0)
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
    
    #print("The AUC score for deletion is", round(de/samples,2),"and for insertion", round(ina/samples,2))
    print(103 * '*')
    print(f"Device = {device}; Time for this experiment was: {(time.time() - start)/3:.3f} seconds")