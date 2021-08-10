# 写给程序员的机器学习入门 (九) - 对象识别 RCNN 与 Fast-RCNN
# https://www.cnblogs.com/zkweb/p/14048685.html

        # RCNN (Region Based Convolutional Neural Network) 是最早期的对象识别模型，实现比较简单，可以分为以下步骤：
        #
        # 用某种算法在图片中选取 2000 个可能出现对象的区域
        # 截取这 2000 个区域到 2000 个子图片，然后缩放它们到一个固定的大小
        # 用普通的 CNN 模型分别识别这 2000 个子图片，得出它们的分类
        # 排除标记为 "非对象" 分类的区域
        # 把剩余的区域作为输出结果

# 使用open-cv库中的选择搜索算法
import cv2

img = cv2.imread('IMG.JPG')
s = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
s.setBaseImage(img)
s.switchToSelectiveSearchFast()
boxes = s.process()     # 按可能性排列出现对象的所有区域
candidate_boxes = boxes[:2000]  # 选取前2000个区域


# 重叠率 (IOU) 判断每个区域是否包含对象
    # 规定重叠率大于 70% 的候选区域包含对象，
    # 重叠率小于 30% 的区域不包含对象，
    # 而重叠率介于 30% ~ 70% 的区域不应该参与学习

def calc_iou(rect1, rect2):
    """计算两个区域重叠部分 / 合并部分的比率 (intersection over union)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
        # 先将横向进行排列， 选出最短长度，再减去初始长度
    hi = min(y1+h1, y2+h2) - yi
    if wi > 0 and hi > 0 : # 有公共部分
        area_overlap = wi*hi
        area_all = w1*h1 + w2*h2 - area_overlap
        iou = area_overlap / area_all
    else:
        iou = 0
    return iou


###############################################
# R_CNN_CODE
import cv2
import numpy as np
import torch
import torchvision
import gzip
import os
from collections import defaultdict
import pandas
import itertools

# 1. load
RESIZE = (32, 32)
IMAGE_DIR = 'target_img/train/image_data/'
BOX_CSV_PATH = 'target_img/train/bbox_train.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def img_to_tensor(img):
    """转换 opencv 图片对象到 tensor 对象"""
    # 注意 opencv 是 BGR，但对训练没有影响所以不用转为 RGB
    img = cv2.resize(img, dsize=RESIZE)
    arr = np.asarray(img)
    t = torch.from_numpy(arr)
    t = t.transpose(0, 2)     # 转换维度 H,W,C 到 C,W,H
    t = t / 255.0           # 正规化数值使得范围在 0 ~ 1
    return t


def save_tensor(tensor, path):
    """保存tensor文件"""
    return torch.save(tensor, gzip.GzipFile(path, 'wb'))


def load_tensor(path):
    """读取tensor文件"""
    return torch.load(gzip.GzipFile(path, 'rb'))


def calc_iou(rect1, rect2):
    """计算两个区域重叠部分 / 合并部分的比率 (intersection over union)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    xi = max(x1, x2)
    yi = max(y1, y2)
    wi = min(x1+w1, x2+w2) - xi
    hi = min(y1+h1, y2+h2) - yi

    if wi > 0 and hi > 0 :  # 有重叠部分
        area_overlap = wi*hi
        area_all = w1*h1 + w2*h2 - area_overlap
        iou = area_overlap / area_all
    else:
        iou = 0
    return iou

def selective_search(img):
    """计算 opencv 图片中可能出现对象的区域，只返回头 2000 个区域"""
    # 算法参考 https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
    s = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    s.setBaseImage(img)
    s.switchToSelectiveSearchFast()
    boxes = s.process()
    return boxes[:2000]


def prepare_save_batch(batch, img_tensor, img_label):
    # 生成输入和输出的tensor文件
    tensor_in = torch.stack(img_tensor)   # 维度: B,C,W,H
    tensor_out = torch.tensor(img_label, dtype=torch.long)   # 维度: B

    # 切分数据集
    random_indices = torch.randperm(tensor_in.shape[0])
    train_indices = random_indices[:int(len(random_indices)*0.8)]
    validate_indices = random_indices[int(len(random_indices)*0.8):int(len(random_indices)*0.9)]
    test_indices = random_indices[int(len(random_indices)*0.9):]
    train_set = (tensor_in[train_indices], tensor_out[train_indices])
    validate_set = (tensor_in[validate_indices], tensor_out[validate_indices])
    test_set = (tensor_in[test_indices], tensor_out[test_indices])

    # 保存
    save_tensor(train_set, f"data/train_set.{batch}.pt")
    save_tensor(validate_set, f"data/validate_set.{batch}.pt")
    save_tensor(test_set, f"data/test_set.{batch}.pt")
    print(f'batch {batch} saved')


class Res_18_model(torch.nn.Module):
    def __init__(self, num_class):
        super(Res_18_model, self).__init__()

        Conv = torchvision.models.resnet18(pretrained=False)
        self.cnn = torch.nn.Sequential(*list(Conv.children())[:-1],
                                       torch.nn.Flatten())
        self.fc = torch.nn.Linear(512, num_class)

    def forward(self, img):
        feat = self.cnn(img)
        # feat = feat.view(feat.shape[0], -1)
        out = self.fc(feat)
        return out


def prepare():
    """prepare train"""
    if not os.path.isdir('data'):
        os.makedirs('data')

    # 加载 csv 文件，构建图片到区域列表的索引 { 图片名: [ 区域, 区域, .. ] }
    box_map = defaultdict(lambda :[])
    df = pandas.read_csv(BOX_CSV_PATH)
    for row in df.values:
        filename, width, height, x1, y1, x2, y2 = row[:7]
        box_map[filename].append((x1, y1, x2-x1, y2-y1))

    # 从图片里面提取人脸 (正样本) 和非人脸 (负样本) 的图片
    batch_size = 1000
    batch = 0
    img_tensors = []
    img_labels = []
    for filename, true_boxes in box_map.items():
        path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(path)
        candidate_boxes = selective_search(img)
        positive_samples = 0
        negative_samples = 0
        for candidate_box in candidate_boxes:
            # 如果候选区域和任意一个实际区域重叠率大于 70%，则认为是正样本
            # 如果候选区域和所有实际区域重叠率都小于 30%，则认为是负样本
            # 每个图片最多添加正样本数量 + 10 个负样本，需要提供足够多负样本避免伪阳性判断
            iou_list = [calc_iou(candidate_box, true_box) for true_box in true_boxes]
            positive_index = next((index for index, iou in enumerate(iou_list) if iou > 0.7), None)
            is_negetive = all(iou < 0.3 for iou in iou_list)
            result = None
            if positive_index is not None:
                result = True
                positive_samples += 1
            elif is_negetive and negative_samples < positive_samples + 10:
                result = False
                negative_samples += 1
            if result is not None:
                x, y, w, h = candidate_box
                chile_img = img[y:y+h, x:x+w].copy()
                # 检验计算是否有问题
                img_tensors.append(img_to_tensor(chile_img))
                img_labels.append(int(result))
                if len(img_tensors) >= batch_size:
                    # 保存批次
                    prepare_save_batch(batch, img_tensors, img_labels)
                    img_tensors.clear()
                    img_labels.clear()
                    batch += 1
        # 保存剩余的批次: < 1000(batch_size)
        if len(img_tensors) > 10:
            prepare_save_batch(batch, img_tensors, img_labels)


def train():
    """start train"""

    model = Res_18_model(num_class=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_acc_history = []
    validate_acc_history=[]
    val_acc_max = -1
    val_acc_max_epoch=0

    # 读取批次的工具函数
    def read_batches(base_path):
        for batch in itertools.count():
            path = f"{base_path}.{batch}.pt"
            if not os.path.isfile(path):
                break
            yield [t.to(device) for t in load_tensor(path)]

    # 计算正确率的工具函数，正样本和负样本的正确率分别计算再平均
    def calc_acc(label, predicted):
        predicted = torch.max(predicted, 1).indices
        acc_positive = ((label > 0.5) & (predicted > 0.5)).sum()
        acc_negetive = ((label <= 0.5) & (predicted <= 0.5)).sum()
        acc = (acc_negetive + acc_positive) / 2
        return acc

    # 划分输入和输出的工具函数
    def split_batch_xy(batch, begin=None, end=None):
        batch_x = batch[0][begin:end]
        batch_y = batch[1][begin:end]
        return batch_x, batch_y

    # 开始训练过程
    for epoch in range(1, 1000):
        print(f"epoch {epoch}")

        model.train()
        train_acc_list = []
        for batch_index, batch in enumerate(read_batches('data/train_set')):
            # 切分小批次，有助于泛化模型
            train_acc_list_batch = []
            for index in range(0, batch[0].shape[0], 100):
                batch_x, batch_y = split_batch_xy(batch, index, index+100) # img_frame, label
                predicted = model(batch_x)
                loss = loss_function(predicted, batch_y)
                # 从损失自动微分求导函数值
                loss.backward()
                # 使用参数调整器调整参数
                optimizer.step()
                # 清空导函数值
                optimizer.zero_grad()
                # 记录这一个批次的正确率，torch.no_grad 代表临时禁用自动微分功能
                with torch.no_grad():
                    train_acc_list_batch.append(calc_acc(batch_y, predicted))
            # record acc
            train_acc_batch = sum(train_acc_list_batch) / len(train_acc_list_batch)
            train_acc_list.append(train_acc_batch)
            print(f"epoch:{epoch}, batch:{batch_index}, batch_acc:{train_acc_batch}")
        train_acc = sum(train_acc_list) / len(train_acc_list)
        train_acc_history.append(train_acc)
        print(f"train_acc:{train_acc}")


        # validata
        model.eval()
        val_acc_list = []
        for batch in read_batches('data/validate_set'):
            batch_x, batch_y = split_batch_xy(batch)
            predicted = model(batch_x)
            val_acc_list.append(calc_acc(batch_y, predicted))
        val_acc = sum(val_acc_list) / len(val_acc_list)
        validate_acc_history.append(val_acc)
        print(f'val_acc:{val_acc}')

        # 记录最高的验证集正确率与当时的模型状态，判断是否在 20 次训练后仍然没有刷新记录
        if val_acc > val_acc_max:
            val_acc_max = val_acc
            val_acc_max_epoch = epoch
            save_tensor(model.state_dict(), 'model_beat.pt')
            print('model_beat updated!')
        elif epoch - val_acc_max_epoch > 20:
            # 在 20 次训练后仍然没有刷新记录，结束训lian
            print('stop training because val_max not update in 20 epochs')
            break

    # 使用达到最高正确率时的模型
    print(f"val_acc_max:{val_acc_max},epoch:{val_acc_max_epoch}")
    model.load_state_dict(load_tensor('model_beat.pt'))

    # test
    test_acc_list = []
    for batch in read_batches('data/test_set'):
        batch_x, batch_y = split_batch_xy(batch)
        predicted = model(batch_x)
        test_acc_list.append(calc_acc(batch_y, predicted))
    test_acc = sum(test_acc_list) / len(test_acc_list)
    print(f'test_acc:{test_acc}')

    # 显示训练集和验证集的正确率变化
    from matplotlib import pyplot
    pyplot.plot(train_acc_history, label='train')
    pyplot.plot(validate_acc_history, label='val')
    # pyplot.ylim(0,1)
    pyplot.legend()
    pyplot.show()


def eval_model():
    """使用训练好的模型"""
    # 创建模型实例，加载训练好的状态，然后切换到验证模式
    model = Res_18_model(2).to(device)
    model.load_state_dict(load_tensor('model_beat.pt'))
    model.eval()

    # 询问图片路径，并显示所有可能是人脸的区域
    while True:
        try:
            # 选取可能出现对象的区域一览
            image_path = input("Image path: ")
            # image_path = 'target_img/train/image_data/10001.jpg'
            if not image_path:
                continue
            img = cv2.imread(image_path)

            candidate_boxes = selective_search(img)
            # 构建输入
            img_tensors = []
            for candidate_box in candidate_boxes:
                x, y, w, h = candidate_box
                chile_img = img[y:y+h, x:x+w]
                img_tensors.append(img_to_tensor(chile_img))
            tensor_in = torch.stack(img_tensors).to(device)

            #predict
            tensor_out = model(tensor_in)
            # 使用 softmax 计算是人脸的概率

            tensor_out = torch.nn.functional.softmax(tensor_out, dim=1)
            #tensor_out = tensor_out[:,1].resize(tensor_out.shape[0])########################
            tensor_out = tensor_out[:, 1].resize(tensor_out.shape[0])

            # 判断概率大于 99% 的是人脸，添加边框到图片并保存
            img_out = img.copy()
            indices = torch.where(tensor_out > 0.99)[0]
            print(len(indices))
            result_boxes = []
            result_boxes_all = []
            for index in indices:
                box = candidate_boxes[index]
                for exists_box in result_boxes_all:
                    if calc_iou(exists_box, box) > 0.30:
                        break
                else:
                    result_boxes.append(box)########################
                result_boxes_all.append(box)##########################
            for box in result_boxes:
                x, y, w, h = box
                print(x, y, w, h)
                cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 0xff), 1)
            cv2.imwrite("img_output.png", img_out)
            print('saved to img_out.png')
            print()
        except Exception as e:
            print("error:", e)
            print('eeeeeeeeeeeeeeeeeerror')

def main():
    prepare()
    train()
    eval_model()

if __name__ == '__main__':
    main()







