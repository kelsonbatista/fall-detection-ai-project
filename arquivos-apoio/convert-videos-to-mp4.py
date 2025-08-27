import os
import subprocess

INPUT_DIR = "videos_test/real_simulation"
OUTPUT_DIR = "videos_test/real_simulation"

# lista arquivos .mpg e .avi
files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".mpg", ".avi"))]
files.sort()  # ordena alfabeticamente

for idx, file in enumerate(files, start=1):
    num = f"{idx:03d}"  # 3 dígitos: 001, 002, ...
    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, f"{num}.mp4")

    print(f"🔄 Convertendo: {input_path} -> {output_path}")

    command = [
        "ffmpeg",
        "-i", input_path,
        "-vcodec", "libopenh264",
        "-acodec", "aac",
        "-b:a", "192k",
        "-y", output_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print(f"✅ Sucesso: {output_path}")
    else:
        print(f"❌ Erro ao converter: {input_path}")
        print(result.stderr.decode("utf-8"))

print("🎬 Conversão concluída!")

