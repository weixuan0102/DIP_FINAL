import os
import subprocess
import glob

# 設定路徑
base_path = "data/custom_scene"  # 修改成你的資料夾名稱
video_path = os.path.join(base_path, "videos")
output_path = os.path.join(base_path, "input")

# 如果 output 資料夾不存在就建立
os.makedirs(output_path, exist_ok=True)

# 取得所有 mp4 檔案
videos = glob.glob(os.path.join(video_path, "*.mp4"))

if not videos:
    print(f"❌ 錯誤：在 {video_path} 找不到任何 .mp4 檔案")
    exit()

print(f"📂 發現 {len(videos)} 個影片，開始抽幀...")

# 設定抽幀頻率 (每幾幀取一張？)
# 如果影片很長，建議設為 2 或 4，避免圖片太多 COLMAP 跑不動
# 如果影片很短，設為 1 (每一幀都要)
FRAME_RATE = 1 

for video in videos:
    video_name = os.path.splitext(os.path.basename(video))[0]
    print(f"Processing {video_name}...")
    
    # 呼叫 ffmpeg
    # -vf "fps=..." 可以控制每秒抽幾張，這裡我們用 %05d 讓它自動編號
    # 輸出的檔名格式會是：cam01_00001.jpg
    cmd = [
        "ffmpeg", 
        "-i", video, 
        "-qscale:v", "1", 
        "-r", str(30/FRAME_RATE), # 假設原始影片是 30fps
        os.path.join(output_path, f"{video_name}_%05d.jpg")
    ]
    
    # 執行指令 (隱藏詳細輸出以免洗版)
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ 所有影片抽幀完成！圖片已存入 data/custom_scene/input")