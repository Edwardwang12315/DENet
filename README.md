# 调试记录
## 2025.1.22
- test输出只有预测txt文件，补充了把预测框绘制出来的步骤
- 简单筛选了一下，置信度小于0.3的不显示，效果很好
- 以上测试用的是作者提供的权重文件，只适用于人脸检测
- _C.TOP_K = 20时，mAP=14.19
- _C.TOP_K = 750时，mAP=14.21
## 2025.1.23
- 修改了部分网络结构

## 2025.3.20
- 损失函数仍然没法下降，感觉应该是金字塔校正的那部分初始化有问题
- 取消了deyolo部分的所有初始化 还是没法收敛
- 取消其他所有的网络，金字塔部分恢复到DENet原始结构，收敛慢但是正常收敛
- 增加正常亮度金字塔对比的高频损失，出现了分类和回归损失消失的问题
- 对DENet权重初始化
- 把多余的函数套用简化了
- 在代码里标注清楚，图像数据都是归一后的值
- 
- 修改了权重初始化函数，把kaiming初始化换成了原先的xavier初始化
- KL输入没有展平均衡化处理，在DENet中怎么让通道数增加。无法实现，DENet返回交换LF的重组图像：Light_dark,Dark_light
- 在VGG中对两幅图像进行初步提取特征再进行KL损失计算，出现nan的问题已解决
- CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py  训练指令      
- **临时启用鼠标**：在tmux会话中按下`Ctrl + b`，然后输入`:set -g mouse on`并按回车
- 损失函数显示错误, 修改为显示平均值，总loss应该显示当前100个iter的平均值(发现严重问题：损失一百次累加一次，次数计数无限累加。已修改，只显示近100个batch的总损失)

- 修改后正确显示损失了，效果非常好（把mutual改成了5）
- 关机后也能正常运行
- val时显示loss不太准确

- DENet部分引入预训练权重，测试加速收敛效果如何：loss_mutual非常小
- 无预训练测试时长：3.25 日 20：00 至  日
- 含预训练训练时长：3.25 日 20：00 至  日

### 3.26
- 旧代码又有新问题：检测部分收敛效果很差
  - ~~检查输入给vgg的图像有没有问题:效果完全不对!!!~~
  - mutual损失设计出错，应该LF不变，HF交换
  - 尝试冻结DENet部分再训练检测部分
    - 检测部分没有一点点效果，尝试权重转移:完成 3.26
    - 先把主干网络模块融到DSFD中：完成 3.26
  - 训练DAINet，对比一下检测损失正常应该是多少：2-4
    - 权重迁移后，几乎不用训练
### 3.27
- 代码编写错误：检测网络应该处理HF_d LF_d
  - 解决上述问题，需要对LF_d进行提亮: 今日任务
  - 调好了损失显示，对HF_d LF_d训练测试一下 完成 3.27
  
- 增加动态全局亮度调整模块 完成 3.28
- 可以用IAYOLO，但工作量不小，先等等
- 增加细节亮度微调模块

### 3.28

- DENet处理极暗图像存在天生缺陷
  - 方法一：样本预处理把参数调高一点，目标域为弱照明 完成 3.28
    - 需要生成弱照明条件下的人脸检测数据集
    - 训练

### 4.3
- 还原初始网络，加入冻结训练（完成）
- 增加了一致性损失（完成）

### 4.7
- epoch:6 || iter:40200

## 显示图像的代码
```python
    print( '这里的位置' )
    image = np.transpose( Light_dark[ 0 ].detach().cpu().numpy() , (1 , 2 , 0) )  # 调整维度顺序 [C, H, W] → [H, W, C]
    image = (image * 255).astype( np.uint8 )
    plt.imshow( image )
    plt.axis( 'off' )
    plt.show()
```
## tmux远程进度分离指令
```bash
# 安装 tmux（如果未安装）：
sudo apt-get install tmux  # Ubuntu/Debian
# 启动 tmux 会话：
tmux new -s training
# 在 tmux 会话中运行训练任务：
python train.py
# 分离会话（保持任务运行）：
按 Ctrl + B，然后按 D。
# 重新连接会话：
tmux attach -t training
```

## Github操作：

**创建仓库**
```bash
git init
git branch -M main # 建立仓库的main分支
git remote add origin git@github.com:**** # 和仓库建立远程联系
```

**提交修改**
```bash
git add -A # 添加所有文件到 缓存区
git commit -m "修改内容" # 缓存区内容提交到云区
git push --set-upstream origin main
```

##### 新建hub

```bash
git init
git add -A
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Edwardwang12315/DENet.git
git push -u origin main
```

