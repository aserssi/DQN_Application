import argparse
import torch
import cv2
import os
import imageio
from src.tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser("Implementation of Deep Q Network to play Tetris")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument("--model_path", type=str, default="models/tetris_best", help="Path to the trained model")
    parser.add_argument("--output_gif", type=str, default="figures/output.gif", help="Output gif file path")
    
    args = parser.parse_args()
    return args

def test(opt):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(opt.output_gif), exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        model = torch.load(opt.model_path)
    else:
        torch.manual_seed(123)
        model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    
    if torch.cuda.is_available():
        model.cuda()
        
    frames = [] # 用于保存 GIF 每一帧
    
    # 获取渲染时的宽高，传给 cv2
    render_width = int(1.5 * opt.width * opt.block_size)
    render_height = opt.height * opt.block_size
    out = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*"mp4v"), opt.fps, (render_width, render_height))

    print(f"Loading model from {opt.model_path} ...")
    print("Playing Tetris and generating GIF...")

    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        
        if torch.cuda.is_available():
            next_states = next_states.cuda()
            
        with torch.no_grad(): # 测试时不需要计算梯度
            predictions = model(next_states)[:, 0]
            
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        
        # 此时渲染一帧并获取画面图像 (需要确保你的 tetris.step 支持提取 frame)
        _, done = env.step(action, render=True, video=out)
        
        # 采用抓取当前 OpenCV 窗口截图的方式来生成 GIF (要求环境中渲染出窗口)
        # 兼容处理：如果没有直接返回图像，可以利用 cv2 读取视频后转 GIF，或直接在 tetris 内部保存画面
        # 这里我们在游戏结束后，统一将 temp.mp4 转换为 gif 以保证色彩一致性

        if done:
            out.release()
            break
            
    # 将 MP4 转换为 GIF 以适配 README 要求
    print("Game over! Converting to GIF...")
    reader = imageio.get_reader('temp.mp4')
    writer = imageio.get_writer(opt.output_gif, fps=opt.fps)
    for frame in reader:
        writer.append_data(frame)
    writer.close()
    
    if os.path.exists('temp.mp4'):
        os.remove('temp.mp4')
        
    print(f"GIF successfully saved to {opt.output_gif}")

if __name__ == "__main__":
    opt = get_args()
    test(opt)