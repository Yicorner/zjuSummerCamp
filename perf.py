import torch
import torchvision
from utils import queryUtils
import pinecone
from utils import data_disk
from utils.args import parse
from utils import drawUtils
from utils.time_test import TestTime
import CLIP
from CLIP.model import set_stage, stage, set_timer


if __name__ == "__main__":
    args = parse()
    stage = args.stage
    num = args.num
    batch_size = args.batch_size
    ifTestTime = args.time
    queryUtils.set_display_freq(args.display_freq)
    
    # deal with time
    timer = None
    if(ifTestTime):
        timer = TestTime()
        set_timer(timer)
        queryUtils.set_timer(timer)
        
    if(stage == "first"):
        pinecone.init(api_key="46b3775f-30d2-4e93-8be0-a0458aa5864d", environment="asia-southeast1-gcp-free")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=False, transform=queryUtils._transform())
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)   
        dataiter = iter(testloader)


        with torch.no_grad():
            cnt = 0
        # 迭代数据集中的所有数据
            while True:
                try:
                    # 获取下一个迭代器元素
                    # 处理数据...
                    images, labels = next(dataiter)
                    if(timer is not None):
                        timer.start("first stage query time")
                    queryUtils.query_accuracy(images, labels, "0", 0)
                    if(timer is not None):
                        timer.stop("first stage query time")
                    cnt = cnt + 1
                    if(cnt == num):
                        break
                except StopIteration:
                    # 所有数据都已经被迭代过了
                    break
            print(f"*********you have tested {cnt * batch_size} images**************") # 所有数据都测试完了  
            if(timer is not None):
                timer.report()
                print("Segment: first stage clip time, Time elapsed: 0 seconds") # CLIP time is 0
        queryUtils.cal_accuracy()
        #save data_score_correct
        dir_path = 'results/first-stage/'  # shift
        data_disk.save_data_to_disk(dir_path, queryUtils.data_score_correct)
        #load data_score_correct
        loaded_data = data_disk.load_data_to_memory(dir_path)
        
        # draw the accuracy curve
        drawUtils.draw_all(loaded_data, dir_path, "first stage")

    if(stage == "second" or stage == "third"):
        if(stage == "second"):
            pinecone.init(api_key="31d26300-a53f-449d-b93f-f1a036b052bc", environment="us-west4-gcp-free")
        elif (stage == "third"):
            pinecone.init(api_key="2306a66e-7368-4f29-b288-3d5c2be8aede", environment="gcp-starter")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = CLIP.load("RN50", device=device)
        set_stage(stage)
        # stage = "second"
        # CLIP.model.stage = "second"
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=False, transform=preprocess)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False)   
        dataiter = iter(testloader)

        with torch.no_grad():
            cnt = 0
        # 迭代数据集中的所有数据
            while True:
                try:
                    # 获取下一个迭代器元素
                    # 处理数据...
                    images, labels = next(dataiter)
                    
                    if(timer is not None):
                        if(stage == "second"):
                            timer.start("second stage clip time")
                        elif(stage == "third"):
                            timer.start("third stage clip time")
                            
                    image_features = model.encode_image(images, labels)
                    
                    if(timer is not None):
                        if(stage == "second"):
                            timer.stop("second stage clip time")
                        elif(stage == "third"):
                            timer.stop("third stage clip time")
                            
                    cnt = cnt + 1
                    if(cnt == num):
                        break
                except StopIteration:
                    # 所有数据都已经被迭代过了
                    break
            print(f"*********you have tested {cnt * batch_size} images**************") # 所有数据都测试完了  
            if(timer is not None):
                timer.report()
                
        queryUtils.cal_accuracy()       
        #save data_score_correct
        dir_path = f'results/{stage}-stage/'  # shift
        data_disk.save_data_to_disk(dir_path, queryUtils.data_score_correct)
        #load data_score_correct
        loaded_data = data_disk.load_data_to_memory(dir_path)
        
        # draw the accuracy curve
        drawUtils.draw_all(loaded_data, dir_path, "second stage")
    







