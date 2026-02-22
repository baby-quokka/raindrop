"""
Train 데이터를 90/10으로 분할하여 train/valid 생성
"""
import random
import os

# 시드 고정 (재현성)
random.seed(42)

# 원본 파일 읽기
with open('data/train/RaindropClarity.txt', 'r') as f:
    all_lines = [ln.strip() for ln in f if ln.strip()]

print(f"Total images: {len(all_lines)}")

# 셔플
random.shuffle(all_lines)

# 90/10 분할
split_idx = int(len(all_lines) * 0.9)
train_lines = all_lines[:split_idx]
valid_lines = all_lines[split_idx:]

print(f"Train: {len(train_lines)} images (90%)")
print(f"Valid: {len(valid_lines)} images (10%)")

# 기존 파일 백업
os.makedirs('data/backup', exist_ok=True)
os.rename('data/train/RaindropClarity.txt', 'data/backup/RaindropClarity_original.txt')
os.rename('data/test/RaindropClarity.txt', 'data/backup/RaindropClarity_test_original.txt')

# 새 파일 저장
with open('data/train/RaindropClarity.txt', 'w') as f:
    f.write('\n'.join(train_lines) + '\n')

with open('data/test/RaindropClarity.txt', 'w') as f:
    f.write('\n'.join(valid_lines) + '\n')

print("\n✅ Split complete!")
print(f"   - Train: data/train/RaindropClarity.txt ({len(train_lines)} images)")
print(f"   - Valid/Test: data/test/RaindropClarity.txt ({len(valid_lines)} images)")
print(f"   - Backup: data/backup/")
