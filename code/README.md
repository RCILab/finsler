# code — 논문의 모든 수치와 그림을 만드는 코드

옆 폴더 `../paper/`(main.tex + supplement.tex)에 들어가는 **모든 표·그림·숫자**가 여기서 나옵니다.
논문이 인용하지 않는 코드는 2026-09-28에 전부 걷어냈습니다(아래 "정리 기록").

```powershell
cd 'C:\Users\ggory\Desktop\new manifold\finsler\code'
python tests/run_tests.py            # 34개 수치 검사 (numpy, scipy)
python tests/run_tests_general.py    # 22개 (torch)
python tests/run_tests_se2.py        # 17개
```

필요한 패키지: `numpy`, `scipy`, `torch`, `matplotlib`. 시뮬만 추가로 `mujoco`, `pyyaml`.

## 논문 → 코드 대응표

논문의 주장마다 어떤 스크립트가 그 숫자를 만들었는지입니다.

| 논문 위치 | 내용 | 스크립트 | 결과 |
|---|---|---|---|
| Fig. 1 (teaser) | 봉투, SE(2) 두 계획, H1-2 실행 | `../paper/figs/make_teaser.py` | `results/fm_planner_se2/*_paths.npz`, `sim/results/*_limited_trajs.npz` |
| Table I, Fig. 3 위 | SE(2) 포즈 계획 0.979 / 1.073 / 1.254 | `experiments/fm_planner_se2.py` | `results/fm_planner_se2/` |
| III-B, Fig. 3 아래 | 평면 채널 1.000 / 1.08 / 2.55 (β=0.9, 0.7) | `experiments/fm_planner.py` | `results/fm_planner_beta0.9/`, `results/fm_planner_beta0.7/` |
| III-B, Fig. 4a | 장애물 0.999, 충돌 0 | `experiments/fm_planner_obs.py` | `results/fm_planner_obs_beta0.9/` |
| Table II, Fig. 5 아래 | 롤아웃에서 봉투 학습 1.023 / 1.036 / 1.126 | `experiments/fit_envelope.py` | `results/fit_envelope/` |
| III-C, Fig. 4b | 미지 β 외삽 (0.9, 0.95) | `experiments/fm_planner_betacond.py` | `results/fm_planner_betacond/` |
| III-C, Fig. 5 위 | H1-2 봉투 식별 + 실행 데모 | `sim/identify_envelope.py`, `sim/demo_turn_or_reverse.py` | `sim/results/*_limited.*` |
| III-D, Fig. 4c | latency 표 | `experiments/latency.py` | `results/latency/` |
| III-D, Fig. 4d | 폐루프 외란 (BC·Dijkstra 포함) | `experiments/closed_loop.py` | `results/closed_loop/` |
| III-B, Suppl. 정확 기준 | 정확한 측지선 대비 0.998 / 1.064 / 1.212 | `experiments/se2_refined_reference.py` | `results/fm_planner_se2/se2_refined_reference.md` |
| (논문 미반영) 외부 베이스라인 | 같은 측지선으로 학습한 궤적 회귀·궤적 확산모델과 비교 | `experiments/external_baselines.py` (SE(2)), `experiments/external_baselines_planar.py` (평면) | `results/external_baselines/` |
| Suppl. 격자 수렴 | 평면 +0.4%, SE(2) +2~3% | `experiments/grid_convergence.py` | `results/grid_convergence/` |
| Suppl. 손실 가중 실험 | ED 0.35/0.93/1.28 대 0.002 | `experiments/fm_loss_weight.py` | `results/fm_loss_weight.md`, `.png` |
| Suppl. 데모 추종 안정화 | \|δ\|=0.8에서 −13% | `experiments/fm_path_benefit.py` | `results/fm_path_beta0.9/`, `results/fm_path_beta0.7/` |
| Suppl. 닫힌 형태·증명 검증 | H 보존 1e-9, EL=Hamiltonian 1e-10, SE(2) 3.0/3.35 s | `tests/run_tests*.py` | (콘솔) |

그림 두 개는 결과에서 직접 그립니다: `../paper/figs/make_teaser.py`(Fig. 1)와 `../paper/figs/make_panels.py`(Fig. 4의 b·c·d 패널).

## 라이브러리

| 파일 | 내용 |
|---|---|
| [finsler/randers.py](finsler/randers.py) | Randers = Zermelo. $F$, 기본 텐서 $g_v$, Legendre와 **닫힌 형태 역변환**, Finsler 기울기, Hamilton 측지선(RK4), exp/log, 역 메트릭 $\bar F$, 봉투 사영, conformal 인자 $s(x)F$(장애물) |
| [finsler/fields.py](finsler/fields.py) | 채널·소용돌이 바람장과 세 메트릭 생성기(Finsler / 리만 부분 $a(x)$ / 유클리드) |
| [finsler/general.py](finsler/general.py) | 구조화된 가족 $F=\sum_k\|L_k(x)v\|+b(x)^\top v$와 $p$-합 변형 (torch). $g_v$ 닫힌 형태, Newton $\mathcal L^{-1}$, Euler–Lagrange 측지선 |
| [finsler/se2.py](finsler/se2.py), [finsler/se2_randers.py](finsler/se2_randers.py) | SE(2) 좌불변 메트릭. Euler–Poincaré $\dot p=\mathrm{ad}^*_\xi p$ (torch 일반 / numpy Randers 닫힌 형태, 후자가 17배 빠름) |
| [finsler/eikonal.py](finsler/eikonal.py) | 비대칭 간선비용 격자 Dijkstra 최적시간 참조. 평면(32방향) / SE(2)(98방향) |
| [notes/01_target_field_derivation.md](notes/01_target_field_derivation.md) | 유도 노트(작업 기록). 0–5b절이 논문 II장의 바탕 |

## 시뮬 (`sim/`)

MuJoCo + `unitree_rl_gym`(third_party, 195 MB clone)의 H1-2 12-DoF MJCF와 사전학습 속도명령 정책.

- `h1_2_env.py` — 헤드리스 환경. `cmd_limits`로 비대칭 명령 제한을 부과합니다.
- `identify_envelope.py` — 명령 스윕 → 달성 속도. `--limits`, `--tag`.
- `demo_turn_or_reverse.py` — 식별 봉투 vs 대칭 부분으로 계획해 실행.

**파일 이름 규칙**: `*_limited.*` = 비대칭 명령 제한을 부과한 실행이고 **논문이 쓰는 결과**입니다.
꼬리표 없는 파일은 원 정책(전후 대칭 봉투) 실행입니다.
`--tag`가 `.md` 요약에 반영되지 않아 두 실행의 요약이 서로 덮어쓰이던 버그를 2026-09-28에 고쳤습니다.
그 과정에서 원 정책 실행의 `.md` 두 개는 이미 덮어써진 상태였고, `.npz`가 남아 있으므로 필요하면 재생성하면 됩니다.

## 해석 주의

- 격자 Dijkstra 참조는 방향 양자화 때문에 참 최적시간을 평면 +0.4%, SE(2) +2~3% 과대추정합니다. 표의 1 미만 비율은 그 오차 이내입니다.
- 모든 봉투는 매끄럽고 강볼록한 근사입니다. 상자형 속도 제한은 $p$-합으로 근사하고, 차동구동(측면 속도 0)은 sub-Finsler 문제로 이 코드의 범위 밖입니다.
- H1-2의 비대칭 명령 제한은 실기에서 측정한 값이 아니라 **부과한 값**입니다. 논문도 그렇게 씁니다.
- 이 폴더의 어떤 결과도 실기 성능을 말하지 않습니다. Go2 실기 항목은 논문에서 빨간 `\nd{}`이고 `../paper/HARDWARE.md`에 측정 절차가 있습니다.

## 정리 기록 (2026-09-28)

논문이 인용하지 않는 것을 지웠습니다. 지운 파일은 전부 워크스페이스 바깥의 `_removed_from_claude_try_2026-09-28.zip`(11.9 MB)에 있습니다.

- MPPI 샘플링 공분산 실험(`mppi_wind.py`, `mppi_vortex.py`)과 그 결과 — 논문을 5섹션 구조로 바꾸면서 부록에서 뺐습니다.
- 분지 조건부 실험(`fm_planner_branch.py`)과 결과 — 부정 결과이고 논문이 인용하지 않습니다.
- `old_fm_path_beta*_reverse_shooting/` — 폐기한 데모 생성기의 결과. 원래부터 인용하지 않습니다.
- `results_quick/` — `--quick` 스크래치(MPPI 전용).

노트의 5절·6.1절·6.4절(MPPI)과 6.10절의 분지 조건부 문단은 이제 그 zip 안의 코드를 가리킵니다. 유도 기록으로 남겨 두었습니다.
