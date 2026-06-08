# 强化学习实验：基于 DQN 的俄罗斯方块 (Tetris Based on DQN)

## 项目简介
本项目是基于深度强化学习（Deep Q-Network, DQN）实现的俄罗斯方块游戏 AI。
智能体通过提取游戏面板的 4 个核心特征（消除行数、空洞数量、平整度、总高度），结合 Experience Replay 机制，在无像素卷积的轻量级全连接网络下实现了快速收敛。经过几千轮次的自我对弈，AI 能够掌握“填补空洞”与“连消多行”的进阶策略。

---

## 目录结构
~~~text
Tetris_Based_on_DQN/
├── figures/                 # 存放生成的演示动图 (.gif)
├── models/                  # 存放训练过程中保存的模型权重文件 (.pth)
├── src/                     # 核心代码目录
│   ├── deep_q_network.py    # DQN 神经网络架构 (特征提取 + MLP)
│   └── tetris.py            # 俄罗斯方块游戏底层逻辑与渲染引擎
├── tensorboard/             # TensorBoard 训练日志目录
├── train.py                 # 模型训练主脚本
├── test.py                  # 单模型测试与动图生成脚本
├── generate_all_gifs.py     # 自动化批处理脚本 (一键生成各训练阶段对比图)
└── README.md                # 项目说明文档
~~~

---

## 环境搭建

为了确保依赖包版本完美兼容，建议使用 Anaconda 创建独立的虚拟环境。

**1. 创建并激活虚拟环境**
~~~bash
conda create -n dqn python=3.9 -y
conda activate dqn
~~~

**2. 安装 PyTorch 核心引擎** (针对 CUDA 11.8)
~~~bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
~~~

**3. 安装依赖库**
*(注意：为兼容深度学习组件，强制指定了稳定的 numpy 与 opencv 版本，并包含了视频解码器)*
~~~bash
pip install numpy==1.26.4
pip install opencv-python==4.9.0.80
pip install tensorboard tensorboardX matplotlib imageio[ffmpeg] Pillow
~~~

---

## 项目运行

### 1. 训练模型
运行以下命令开始训练模型。每 200 轮保存一次模型，遇到历史最高分时会自动保存为 `tetris_best`。
~~~bash
python train.py --save_interval 200
~~~
*(提示：如需生成训练过程的阶段性对比图，建议启动时指定更密集的保存间隔，例如 `python train.py --save_interval 200`)*

### 2. 监控训练曲线
保持训练终端运行，新开一个终端窗口，激活环境后输入：
~~~bash
tensorboard --logdir=tensorboard
~~~
随后在浏览器中打开 `http://localhost:6006/`，即可实时查看模型得分 (Score)、消除行数 (Cleared lines) 和损失函数 (Loss) 的变化曲线。

### 3. 测试与生成单张动图
当模型训练完毕后，运行以下命令验证其实际表现，并生成指定帧率的慢速演示 GIF：
~~~bash
python test.py --model_path models/tetris_best --output_gif figures/output_best.gif --fps 10
~~~

### 4. 一键批量生成对比动图
项目中包含了自动化脚本。只要你在训练时生成了 `tetris_200`, `tetris_400` 等阶段性模型，直接运行此脚本即可一键完成测试、视频录制与 GIF 转换：
~~~bash
python generate_all_gifs.py
~~~
