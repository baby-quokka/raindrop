## 루트 디렉터리

이 레포는 **ConvIR 단일 모델** 기반의 빗방울 제거(또는 저-level 복원) 실험에 맞춰 정리되어 있습니다.

| 파일 | 역할 |
|------|------|
| `main.py` | **진입점**. `--mode train` / `--mode test` 로 학습 또는 테스트 실행. Accelerator·WandB 설정, 모델 선택(`--model_name=ConvIR`), `train.py`·`eval.py` 호출. |
| `train.py` | ConvIR **학습 루프**. dataloader·optimizer·scheduler 구성, multi-scale loss(L1 + LPIPS + MS-SSIM + FFT), EMA, 주기적 검증(`valid.py`) 및 체크포인트 저장. |
| `eval.py` | **테스트/평가**: PSNR(Y 채널), SSIM, LPIPS 계산, 복원 결과 이미지 저장. `data/test/*.txt` 기반 dataloader 사용. |
| `valid.py` | **학습 중 검증**: 동일 메트릭(PSNR/SSIM/LPIPS)으로 `valid_dataloader` 평가, best PSNR/SSIM/LPIPS 갱신 및 로그 기록. |
| `utils.py` | 공통 유틸: 시간/평균 계산, 텐서 ↔ 넘파이/이미지 변환 등. |
| `inference_ntire_dev.py` | **NTIRE dev 세트**(예: Drop 407장)용 추론. `data/valid/RaindropClarity.txt` 기준으로 `submission/`에 결과 및 readme 생성. |
| `split_train_valid.py` | `data/train/RaindropClarity.txt`를 **train/valid(txt) 90/10 분할**로 생성, 원본은 `data/backup/`에 백업. |


---

## data/

| 파일/폴더 | 역할 |
|-----------|------|
| `dataset.py` | **IRDataset**: txt에서 입력/GT 경로 읽기, RaindropClarity(Drop↔Clear), NH-HAZE 등 경로 규칙, crop·flip augmentation, value_range (0,1)/(-1,1). |
| `build_filename.py` | 데이터 경로 리스트를 txt로 만드는 스크립트(데이터 준비용). |
| `utils.py` | `read_rgb`, `crop_pair`, `flip_pair`, `to_m01`, `to_m11` 등 데이터 로딩/전처리. |
| `train/` | 학습용 이미지 경로가 적힌 `.txt` (예: RaindropClarity.txt). |
| `test/` | 학습 중 검증용 이미지 경로가 적힌 `.txt`. |
| `valid/` | 챌린지 validation용 이미지 경로가 적힌 `.txt`. |
| `backup/` | `split_train_valid.py` 실행 시 원본 txt 백업. |

---

## models/

| 파일 | 역할 |
|------|------|
| `ConvIR.py` | ConvIR 메인 모델. encoder·decoder, multi-scale branch, multi-shape kernel(dynamic filter + strip attention) 등을 포함. `build_net(version, data)`로 인스턴스 생성. |
| `__init__.py` | `from models import build_net` 형태로 ConvIR 빌더를 노출. |

---

## 기타 폴더

| 폴더 | 역할 |
|------|------|
| `results/` | 학습 결과: ConvIR 체크포인트(.pkl), 로그, CSV, 샘플 이미지 등이 저장. |
| `submission/` | NTIRE 제출용 추론 결과 이미지·readme 등을 저장. |
| `wandb/` | 옵션(`--use_wandb`) 사용 시 Weights & Biases 로컬 로그 디렉터리. |
| `trash/` | 실험 중 더 이상 사용하지 않는 파일·모델을 임시 보관하는 폴더(필요 시 수동 정리). |



