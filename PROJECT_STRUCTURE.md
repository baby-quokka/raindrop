## 루트 디렉터리

| 파일 | 역할 |
|------|------|
| `main.py` | **진입점**. `--mode train` / `--mode test`로 학습 또는 테스트 실행. Accelerator·WandB 설정, 모델 선택(`--model_name`), `train.py`·`eval.py` 호출. |
| `train.py` | 일반 모델(Restormer, NAFNet 등) **학습 루프**. semantic label 생성, dataloader·optimizer·scheduler, 검증 주기 호출(`valid.py`). |
| `train_diffusion.py` | **Diffusion 계열**(DiffIR 등) 전용 학습 스크립트. |
| `train_flow.py` | **RectifiedFlow** 기반 학습. Stable Diffusion 등 외부 모델 연동. |
| `train_strrnet.py` | **STRRNet**(NTIRE 2025 1st) 전용 학습. L1+MS-SSIM, 92k constant → cosine decay LR. |
| `eval.py` | **테스트/평가**: PSNR(Y채널), SSIM, LPIPS 계산, 이미지 저장. `test_dataloader` 사용. |
| `valid.py` | **학습 중 검증**: 동일 메트릭(PSNR/SSIM/LPIPS)으로 `valid_dataloader` 평가. |
| `utils.py` | 공통 유틸: `Adder`, `Timer`, `norm_range`, `norm_ip`, `tensor_to_numpy` 등. |
| `run_eval.py` | 학습된 모델만 **따로 평가**할 때 사용. `--model_name`, `--exp_name`, `--test_model`, `--test_data` 등 지정 후 `eval._eval` 호출. |
| `inference_ntire_dev.py` | **NTIRE dev**(예: Drop 407장)용 추론. `data/valid/RaindropClarity.txt` 기준으로 submission 폴더에 결과 저장, readme 생성. |
| `inference_strrnet.py` | **STRRNet 전용 추론**: 128×128 슬라이딩 윈도우, 중앙값 융합 등 논문 전략 적용. |
| `split_train_valid.py` | `data/train/RaindropClarity.txt`를 **90/10 분할**하여 train·valid(test)용 txt 생성, 원본은 backup. |


---

## data/

| 파일/폴더 | 역할 |
|-----------|------|
| `dataset.py` | **IRDataset**: txt에서 입력/GT 경로 읽기, RaindropClarity(Drop↔Clear), NH-HAZE 등 경로 규칙, crop·flip augmentation, value_range (0,1)/(-1,1). |
| `strrnet_dataset.py` | STRRNet용 데이터셋·dataloader (patch 128 등). |
| `build_filename.py` | 데이터 경로 리스트를 txt로 만드는 스크립트(데이터 준비용). |
| `utils.py` | `read_rgb`, `crop_pair`, `flip_pair`, `to_m01`, `to_m11` 등 데이터 로딩/전처리. |
| `train/` | 학습용 이미지 경로가 적힌 `.txt` (예: RaindropClarity.txt). |
| `test/` | 테스트용 이미지 경로가 적힌 `.txt`. |
| `valid/` | 검증용 이미지 경로가 적힌 `.txt`. |
| `backup/` | `split_train_valid.py` 실행 시 원본 txt 백업. |

---

## models/

| 파일 | 역할 |
|------|------|
| `Restormer.py` | Restormer 계열 (기본·Small·Base·MultiScale). |
| `NAFNet.py` | NAFNet. |
| `SwinIR.py` | SwinIR. |
| `Uformer.py` | Uformer. |
| `ConvIR.py` | ConvIR. |
| `IDT.py` | IDT. |
| `PWFNet.py` | PWFNet. |
| `LWN.py` | LWN. |
| `MambaIR.py` | MambaIR (Tiny/Small/Base). |
| `DiffIR.py` | Diffusion 기반 복원(DiffIR 등). |
| `RectifiedFlow.py` | Rectified Flow 복원 모델. |
| `STRRNet.py` | NTIRE 2025 1st STRRNet + SemanticClassifier (day/night, raindrop focus 등). |
| `SimpleDiffusion.py` | 단순 Diffusion 복원(주석 처리된 경우 있음). |

---

## losses/

| 파일 | 역할 |
|------|------|
| `ms_ssim.py` | MS-SSIM 손실 (STRRNet 등에서 사용). |
| `__init__.py` | losses 패키지 노출. |

---

## 기타 폴더

| 폴더 | 역할 |
|------|------|
| `results/` | 학습 결과: 체크포인트(.pkl), 로그, CSV, 샘플 이미지 등. |
| `submission/` | NTIRE 제출용 추론 결과 이미지·readme 등. |


