import os
from io import BytesIO
from pathlib import Path
from random import shuffle

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from mini_resnet import CustomResNet
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms as T

mean = (0.49139968, 0.48215841, 0.44653091)
std = (0.24703223, 0.24348513, 0.26158784)
transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
softmax = torch.nn.Softmax(dim=0)

model = CustomResNet()
model.load_state_dict(torch.load("weights/weights.pt", map_location=torch.device("cpu")))
model.eval()

misclf_path = "images/miss_classified"
mis_classified_imgs = list(Path(misclf_path).glob("*"))


def get_traget_layer(block: str, layer: int):
    layer_num = 0 if layer == 0 else -1
    if block == "block1":
        return model.layer1[layer_num]
    if block == "block2":
        return model.layer2[layer_num]
    if block == "block3":
        return model.layer3[layer_num]


default_cam = GradCAM(model=model, target_layers=[get_traget_layer("block3", -1)])


def make_image(p: Path | str, pred: str, label: str):
    im = cv2.imread(str(p))
    im = cv2.resize(im, (64, 64))

    plt.imshow(im)
    plt.title(f"{pred} / {label}")
    plt.axis("off")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    img_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()

    # Decode the image array using OpenCV
    im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return im


@torch.inference_mode()
def predict_img(img: np.ndarray, top_k: int = 10):
    preds = model(img)
    preds = softmax(preds.flatten())
    preds = {classes[i]: float(preds[i]) for i in range(10)}
    preds = {
        k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)[:top_k]
    }

    return preds


def display_cam(cam: GradCAM, org_img: np.ndarray, img: torch.Tensor, transparency: float):
    grayscale_cam = cam(input_tensor=img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        org_img / 255, grayscale_cam, use_rgb=True, image_weight=transparency
    )
    return visualization


def inference(
    org_img: np.ndarray,
    top_k: int,
    show_cam: str,
    num_cam_imgs: int,
    cam_block: str,
    target_layer_num: int,
    transparency: float,
    show_misclf: str,
    num_misclf: int,
):
    input_img = transforms(org_img)
    input_img = input_img.unsqueeze(0)

    preds = predict_img(input_img, top_k)
    org_img = display_cam(default_cam, org_img, input_img, transparency)

    shuffle(mis_classified_imgs)
    cam_outputs = []
    if show_cam:
        img_list = []

        target_layers = [get_traget_layer(cam_block, target_layer_num)]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        for p in mis_classified_imgs[:num_cam_imgs]:
            im = cv2.imread(str(p))
            inp_im = transforms(im)
            inp_im = inp_im.unsqueeze(0)

            grayscale_cam = cam(input_tensor=inp_im, targets=None)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(
                im / 255, grayscale_cam, use_rgb=True, image_weight=transparency
            )
            cam_outputs.append(visualization)

        del cam, img_list

    misclf_images_output = []
    if show_misclf:
        img_list = []
        gt = []
        for p in mis_classified_imgs[:num_misclf]:
            img_list.append(transforms(Image.open(p).convert("RGB")))
            gt.append(p.name.split("_")[0])

        misclf_out = softmax(model(torch.stack(img_list))).argmax(dim=1).tolist()
        del img_list
        for imp, pred, label in zip(mis_classified_imgs[:num_misclf], misclf_out, gt):
            pred = classes[pred]
            misclf_images_output.append(make_image(imp, pred, label))

    return org_img, preds, cam_outputs, misclf_images_output


title = "CIFAR10 trained on Custom Model inspired by ResNet with GradCAM"
description = "A simple Gradio interface to infer on ResNet model, and get GradCAM results"
# examples = [["cat.jpg", 0.5, -1], ["dog.jpg", 0.5, -1]]
demo = gr.Interface(
    inference,
    inputs=[
        gr.Image(shape=(32, 32), label="Input Image"),
        gr.Slider(1, 10, value=3, step=1, label="Top K predictions"),
        gr.Checkbox(label="Show Grad Cam"),
        gr.Slider(1, 20, value=5, step=1, label="Number of images"),
        gr.Radio(label="Which Block?", choices=["block1", "block2", "block3"]),
        gr.Slider(0, 1, value=1, step=1, label="Which Layer?"),
        gr.Slider(0, 1, value=0.5, label="Opacity of GradCAM"),
        gr.Checkbox(label="Show Misclassified Images"),
        gr.Slider(1, 20, value=5, step=5, label="Number of Misclassification Images"),
    ],
    outputs=[
        gr.Image(shape=(32, 32), label="Output", width=128, height=128),
        "label",
        gr.Gallery(label="GradCAM Output"),
        gr.Gallery(
            label="Misclassified Images Pred/G.T.",
            columns=[2],
            rows=[2],
            object_fit="contain",
            height="auto",
        ),
    ],
    title=title,
    description=description,
    # examples=examples,
)
demo.launch()
