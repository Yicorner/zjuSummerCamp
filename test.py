import argparse
import torch
import CLIP
from PIL import Image
import pinecone

from CLIP.model import set_stage
from utils import queryUtils 

queryUtils.Inference = True
parser = argparse.ArgumentParser(description="一个简单的命令行参数示例")

# 添加一个命令行参数
parser.add_argument('--stage', type=str, required=True, help="stage of the intermedia result, first or second or third or origin or all")
parser.add_argument('--image_name', type=str, required=True, help="pick a picture from test_images, airplane or dog or cat or ..., Be careful not to add articles(a or an).")
# 解析命令行参数
args = parser.parse_args()
stage = args.stage
image_path = "test_images/" + args.image_name + ".png"
cifar10_dict = {
    0: "a airplane",
    1: "an automobile",
    2: "a bird",
    3: "a cat",
    4: "a deer",
    5: "a dog",
    6: "a frog",
    7: "a horse",
    8: "a ship",
    9: "a truck",
}

if(stage == "origin" or stage =="all"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = CLIP.load("RN50", device=device)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = CLIP.tokenize(["a airplane", "an automobile", "a bird", "a cat", "a deer", "a dog", "a frog", "a horse", "a ship", "a truck"]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("origin clip predict result:", cifar10_dict[probs[0].argmax()]) 


if(stage == "first" or stage =="all"):
    pinecone.init(api_key="46b3775f-30d2-4e93-8be0-a0458aa5864d", environment="asia-southeast1-gcp-free")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 获取下一个迭代器元素
    # 处理数据...
    image = queryUtils._transform()(Image.open(image_path)).unsqueeze(0).to(device)
    queryUtils.query_accuracy(image, None, "0", 0)
    result = int(queryUtils.get_result())
    print(f"third stage predict result : {cifar10_dict[result]}")
        
if(stage == "second" or stage =="all"):
    pinecone.init(api_key="31d26300-a53f-449d-b93f-f1a036b052bc", environment="us-west4-gcp-free")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = CLIP.load("RN50", device=device)
    set_stage("second")
    with torch.no_grad():
    # 迭代数据集中的所有数据
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
    result = int(queryUtils.get_result())
    print(f"second stage predict result : {cifar10_dict[result]}")

if(stage == "third" or stage =="all"):
    pinecone.init(api_key="2306a66e-7368-4f29-b288-3d5c2be8aede", environment="gcp-starter")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = CLIP.load("RN50", device=device)
    set_stage("third")
    with torch.no_grad():
    # 迭代数据集中的所有数据
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
    result = int(queryUtils.get_result())
    print(f"third stage predict result : {cifar10_dict[result]}")
