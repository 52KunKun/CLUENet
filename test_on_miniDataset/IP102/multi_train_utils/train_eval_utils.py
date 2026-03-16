import sys

from tqdm import tqdm
import torch
import time
import math
from torch import nn
from .distributed_utils import reduce_value, is_main_process, warmup_lr_scheduler
# 梯度爆炸检查
module2name = {}
module2parent = {}

def build_module_maps(model):
    for name, module in model.named_modules():
        module2name[module] = name
        for child_name, child_module in module.named_children():
            module2parent[child_module] = module
# 添加狗子
def check_nan_in_output(output):
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            return True
    elif isinstance(output, (list, tuple)):
        # 如果是容器（list 或 tuple），递归检查每个元素
        for item in output:
            if check_nan_in_output(item):
                return True
    return False
def check_inf_in_output(output):
    if isinstance(output, torch.Tensor):
        if torch.isinf(output).any():
            return True
    elif isinstance(output, (list, tuple)):
        # 如果是容器（list 或 tuple），递归检查每个元素
        for item in output:
            if check_inf_in_output(item):
                return True

def check_nan_hook(module, input, output):
    """检查输出是否包含NaN，并打印模块信息"""
    if check_nan_in_output(input):
        module_name = module2name.get(module, "<unnamed>")
        parent_module = module2parent.get(module, None)
        parent_name = module2name.get(parent_module, "<no parent>")
        print(f"[input NaN Detected] Module: {module_name} ({module.__class__.__name__}), Parent: {parent_name}")
    if check_nan_in_output(output):
        module_name = module2name.get(module, "<unnamed>")
        parent_module = module2parent.get(module, None)
        parent_name = module2name.get(parent_module, "<no parent>")
        print(f"[output NaN Detected] Module: {module_name} ({module.__class__.__name__}), Parent: {parent_name}")
    if check_inf_in_output(output):
        module_name = module2name.get(module, "<unnamed>")
        parent_module = module2parent.get(module, None)
        parent_name = module2name.get(parent_module, "<no parent>")
        print(f"[output NaN Detected] Module: {module_name} ({module.__class__.__name__}), Parent: {parent_name}")
    if check_inf_in_output(output):
        module_name = module2name.get(module, "<unnamed>")
        parent_module = module2parent.get(module, None)
        parent_name = module2name.get(parent_module, "<no parent>")
        print(f"[output NaN Detected] Module: {module_name} ({module.__class__.__name__}), Parent: {parent_name}")

def check_params_without_grad(model):
    print("\n🔍 Checking parameters without gradients...")
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"❌ No gradient for: {name}")
            no_grad_params.append(name)
    if not no_grad_params:
        print("✅ All parameters have gradients.")
    return no_grad_params

def check_atten_alpha(model):
    """
        检查 model 中带有 'atten_alpha' 的参数是否小于 torch.log(0.01)
        """
    threshold = torch.tensor(0.01)
    found = False
    violation = False

    print(f"Threshold = {threshold.item():.6f}")
    print("Checking atten_alpha parameters...\n")

    for name, param in model.named_parameters():
        if "attn_alpha" in name:
            found = True
            value = param.detach().cpu().item() if param.numel() == 1 else param.detach().cpu()
            value = math.exp(value)
            is_small = (value < threshold).any().item()
            if is_small:
                violation = True
                print(f"{name}: value = {value}, below threshold? {is_small}")

    if not found:
        print("⚠️ No parameter found containing 'atten_alpha'.")
    elif not violation:
        print("\n✅ All atten_alpha parameters are >= 0.01.")
    else:
        print("\n❌ Some atten_alpha parameters are smaller than 0.01.")

    return not violation


# 给模型加载狗子
def load_hook(model):
    for name, module in model.named_modules():
        module.register_forward_hook(check_nan_hook)

def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=True,clip_norm=None,loss_function=torch.nn.CrossEntropyLoss()):
    model.train()

    # 输出爆炸检查
    # build_module_maps(model)
    # load_hook(model)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    enable_amp = use_amp and "cuda" in device.type
    # 启用混合精度训练减少显存占用
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))  # 这样就非常润滑了！

            # 兼容多损失输出
            if isinstance(loss, (tuple, list)):
                loss_items = [l.detach().item() if hasattr(l, 'detach') else float(l) for l in loss]
                total_loss = sum(loss)
            else:
                loss_items = [loss.detach().item() if hasattr(loss, 'detach') else float(loss)]
                total_loss = loss
            if isinstance(pred, (tuple, list)):
                pred = pred[0]  # 默认选取第一个
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

            # 如果出现损失爆炸时进行分析调整
            # if not torch.isfinite(total_loss):
            #     print(f"[!] Loss exploded at batch {step}, value = {total_loss.item()}")
            #     # 保存现场：输入、标签、输出、模型参数
            #     torch.save({
            #         "epoch": epoch,
            #         "step": step,
            #         "images": images.cpu(),
            #         "labels": labels.cpu(),
            #         "pred": pred.detach().cpu() if isinstance(pred, torch.Tensor) else None,
            #         "loss": total_loss.detach().cpu(),
            #         "model_state_dict": model.state_dict()
            #     }, f"debug_loss_exploded_epoch{epoch}_step{step}.pt")
            #
            #     # 可选：保存 scaler 状态
            #     torch.save(scaler.state_dict(), f"debug_scaler_epoch{epoch}_step{step}.pt")
            #     sys.exit(1)

            # 出现损失爆炸记录并跳过
            if not torch.isfinite(total_loss) or total_loss.item() > 1e5:
                print(
                    f"[!] Loss exploded at epoch {epoch}, step {step}, value = {total_loss.item():.4e}, skipping batch")
                # ---- 清理显存与计算图 ----
                optimizer.zero_grad(set_to_none=True)
                if isinstance(total_loss, torch.Tensor):
                    total_loss.detach_()
                del total_loss, loss, pred, images, labels  # 删除本 batch 临时变量
                torch.cuda.empty_cache()  # 主动释放 CUDA 显存缓存
                continue  # 关键！跳过 backward 和 step

        scaler.scale(total_loss).backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        # check_params_without_grad(model)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 检查温度缩放因子 是否在训练过程中超过100
        # check_atten_alpha(model)

        total_loss = reduce_value(total_loss, average=True)
        accu_loss += total_loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            loss_str = ', '.join([f'loss{i}: {v:.3f}' for i, v in enumerate(loss_items)])
            info = f"[epoch {epoch}] {loss_str}, total_loss: {accu_loss.item() / (step + 1):.3f}, train_acc: {(accu_num.item() / sample_num) * 100:.2f}%, lr: {optimizer.param_groups[0]['lr']:.8f}"
            data_loader.desc = info
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1),accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval() # 设置评估模式

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num_top1 = torch.zeros(1).to(device)
    # 还要存储top-5的准确个数
    sum_num_top3 =torch.zeros(1).to(device)
    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 记录开始时间
    start_time = time.time()
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            # 加一个判断就行
            if isinstance(pred, (tuple, list)):
                pred = pred[0]  # 默认选取第一个
            # top-1正确个数
            top1_pred = torch.max(pred, dim=1)[1]  # 计算预测的最大值索引
            sum_num_top1 += torch.eq(top1_pred, labels.to(device)).sum()
            # top-5正确个数
            top5_pred = pred.topk(3,dim=1,largest=True,sorted=True)[1]
            sum_num_top3 += torch.sum(torch.eq(top5_pred,labels.unsqueeze(1).expand_as(top5_pred).to(device)),dim=1).sum()
    # 保存结束时间
    end_time = time.time()
    total_time = end_time-start_time
    total_fps = num_samples / total_time
    print(f"Total time for evaluation: {total_time:.2f} seconds")
    print(f"Average FPS: {total_fps:.2f} images per second")

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num_top1 = reduce_value(sum_num_top1, average=False)
    sum_num_top3 = reduce_value(sum_num_top3, average=False)
    acc_top1 = sum_num_top1.item() / num_samples
    acc_top3 = sum_num_top3.item()/num_samples
    return acc_top1,acc_top3

@torch.no_grad()
def evaluate_test(model, data_loader, device):
    model.eval() # 设置评估模式

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num_top1 = torch.zeros(1).to(device)
    # 还要存储top-5的准确个数
    sum_num_top3 =torch.zeros(1).to(device)
    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 记录开始时间
    start_time = time.time()
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images.to(device))
            # top-1正确个数
            top1_pred = torch.max(pred, dim=1)[1]  # 计算预测的最大值索引
            sum_num_top1 += torch.eq(top1_pred, labels.to(device)).sum()
            # top-5正确个数
            top5_pred = pred.topk(3,dim=1,largest=True,sorted=True)[1]
            sum_num_top3 += torch.sum(torch.eq(top5_pred,labels.unsqueeze(1).expand_as(top5_pred).to(device)),dim=1).sum()
    # 保存结束时间
    end_time = time.time()
    total_time = end_time-start_time
    total_fps = num_samples / total_time
    print(f"Total time for evaluation: {total_time:.2f} seconds")
    print(f"Average FPS: {total_fps:.2f} images per second")

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num_top1 = reduce_value(sum_num_top1, average=False)
    sum_num_top3 = reduce_value(sum_num_top3, average=False)
    acc_top1 = sum_num_top1.item() / num_samples
    acc_top3 = sum_num_top3.item()/num_samples
    return acc_top1,acc_top3,total_fps


# 专门用来做可视化测试的（正式的训练与测试不用管）
@torch.no_grad()
def evaluate_test_error(model, data_loader, device):
    model.eval()
    num_samples = len(data_loader.dataset)
    sum_num_top1 = torch.zeros(1).to(device)
    sum_num_top3 = torch.zeros(1).to(device)
    misclassified_filenames = []  # 存储错误样本文件名

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    start_time = time.time()
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            # 解包 data (images, labels, filenames)
            images, labels, filenames = data
            images, labels = images.to(device), labels.to(device)

            pred = model(images)
            top1_pred = torch.max(pred, dim=1)[1]
            top5_pred = pred.topk(3, dim=1, largest=True, sorted=True)[1]

            # top-1 正确数统计
            correct_top1 = top1_pred.eq(labels)
            sum_num_top1 += correct_top1.sum()

            # top-3 正确数统计
            correct_top3 = top5_pred.eq(labels.unsqueeze(1).expand_as(top5_pred))
            sum_num_top3 += correct_top3.sum(dim=1).sum()

            # 找出预测错误的样本
            for i, correct in enumerate(correct_top1):
                if not correct.item():  # 判断tensor_bool类型要先转为item
                    misclassified_filenames.append(filenames[i])

    end_time = time.time()
    total_time = end_time - start_time
    total_fps = num_samples / total_time
    print(f"Total time for evaluation: {total_time:.2f} seconds")
    print(f"Average FPS: {total_fps:.2f} images per second")

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num_top1 = reduce_value(sum_num_top1, average=False)
    sum_num_top3 = reduce_value(sum_num_top3, average=False)
    acc_top1 = sum_num_top1.item() / num_samples
    acc_top3 = sum_num_top3.item() / num_samples

    # 打印错误样本文件名
    print("\nMisclassified samples (Top-1):")
    for name in misclassified_filenames:
        print(f'"{name}",')

    return acc_top1, acc_top3, total_fps


