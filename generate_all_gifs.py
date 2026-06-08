import os
import subprocess

# 定义10个特定迭代次数
target_epochs = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

print("==================================================")
print(" 开始批量生成指定轮次的 Tetris AI 演示 GIF ")

# 确保存放图片的文件夹存在
os.makedirs("figures", exist_ok=True)

for epoch in target_epochs:
    model_path = f"models/tetris_{epoch}"
    output_gif = f"figures/output_{epoch}.gif"
    
    # 检查对应的模型文件是否存在
    if os.path.exists(model_path):
        print(f"\n[运行中] 正在为第 {epoch} 轮模型生成动图...")
        
        # 自动调用之前写好的 test.py 脚本
        command = f"python test.py --model_path {model_path} --output_gif {output_gif} --fps 6"
        
        # 执行命令
        subprocess.run(command, shell=True)
        print(f"[已完成] 动图已成功保存至: {output_gif}")
    else:
        print(f"\n[跳过] 未找到模型文件 {model_path}。")
        print(f"      请确保 train.py 已经训练并产生该文件（启动训练时需加 --save_interval 200）。")

print(" 所有可用的 GIF 动图已批量生成完毕")
print("==================================================")