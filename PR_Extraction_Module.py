import numpy as np
import pandas as pd
import random
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
import torch
import os
import csv
import time
import matplotlib.pyplot as plt
import json, datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

from tqdm import tqdm


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv  # ✅ 추가
from stable_baselines3.common.vec_env import VecNormalize  # ✅ 추가

import multiprocessing

# ✅ 스i레드 수 설정 (강제 제한할 경우)
torch.set_num_threads(8)  # 또는 multiprocessing.cpu_count() // 2
# ✅ 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
# ✅ 멀티 환경 수 자동 설정 (선택사항)
#num_envs = min(8, multiprocessing.cpu_count() // 2)
num_envs = 4

KNN = 20

TRAINING = False
TESTING = not TRAINING  # TESTING 모드는 TRAINING이 아닐 때 활성화
#MODEL_PATH = "model/default/pred_base.zip"
#MODEL_PATH = "model/default/default2.zip"
#MODEL_PATH = "model/noisemodel/hetero_gaussian.zip"
#MODEL_PATH = rf"model/dataset9/knn_{KNN}.zip"

MODEL_PATHS = {
    "DRL": "model/default/default2.zip",
    "Baseline_DRL": "model/default/pred_base.zip"
}

# 실행 option  트레이닝 타임 스텝 수와 타임 슬롯 용량 크기  
 
TOTAL_STEPS = 1_000_000
#TOTAL_STEPS = 800_000
SIZE_FACTOR = 0.3


# FINETUNE 과정이 쓰일 경우 필요함 
FINETUNE = False       # ← 추가 미세 학습을 할지 여부
FINETUNE_STEPS = 200_000 # ← 10~30만 스텝 정도 권장

# 인기도 변화 모델 인자 실험 시 다양한 인자 바꿀 수 있음 
VIDEO_TRAINING_MODEL  = "hetero_gaussian"   #  "gaussian"  | "hetero_gaussian" | "hetero_gaussian_head" | "hetero_gaussian_tail" | "none" | "real_dataset_train" | "real_dataset_test"
VER_TRAINING_MODEL = "gaussian"  # "gaussian"   # "dirichlet" | "gaussian" | "none"
VIDEO_PARAM = 0.05  # SIGMA   0.02 ~ 0.05

HETERO_BETA = 0.4
VER_PARAM = 0.02 # SIGMA   0.02 ~ 0.05


ZIPF_PARAMETER = 0.791
#ZIPF_PARAMETER = 1.0
SLOT_CONCENTRATION = 4
FIXED_ZIPF = 1   # 1 or 0 

# NOT USED 
VIDEO_TAU = 1.5 
VIDEO_LAMBDA = 0.3


class TransEnv(gym.Env):
    def __init__(self,
                 video_noise_model="gaussian",   
                 video_noise_param=0.05,            # gaussian
                 video_tau=0.10,                    # Not used
                 video_lambda=0.40,                 # Not used
                 # 버전 노이즈
                 version_noise_model="gaussian",   # "dirichlet" | "gaussian" | "none"
                 version_noise_param=0.05,          # 기본 강도
                 seed=None):
        super(TransEnv, self).__init__()

        self.video_noise_model   = video_noise_model
        self.video_noise_param   = float(video_noise_param)
        self.video_tau           = float(video_tau)
        self.video_lambda        = float(video_lambda)
        self.version_noise_model = version_noise_model
        self.version_noise_param = float(version_noise_param)

        #self.n_slots = 100  # 10 X 10 
        self.n_deadline = 10
        #self.n_servers = 10
        self.n_servers = 3
        self.n_slots = self.n_deadline * self.n_servers
        #self.n_videos = 1000
        self.n_videos = 174
        self.n_ver = 7
        self.episode_count = 0

        self.bitrates = np.array([0.3, 0.7, 1.5, 2.5, 5.0, 8.0, 12.0], dtype=np.float32)
        self.current_video = 0
        self.time_used = np.zeros(self.n_slots, dtype=np.float32)
        self.slot_groups = np.arange(self.n_slots, dtype=np.int32) % self.n_deadline  # ✅ 추가

        self.allocation_dict = {s: [] for s in range(self.n_slots)}





        csv_path = "transcoding_dataset_final.csv"     # CSV 경로
        uniform_length_sec = 30.0                     # 비디오 길이
        uniform_bitrate_kbps = 8000.0                  # 비디오 비트레이트

         # (1) 모델 학습(한 번만) & (2) 매트릭스 생성
        self._m_time = None   # RTF 모델
        self._m_vmaf = None   # VMAF 모델
        self._train_models_from_csv(csv_path)

        # resolutions: 반드시 CSV와 일치해야 함
        self._resolutions = np.array([144, 240, 288, 360, 480, 720, 1080], dtype=np.int32)

        # 전 비디오 동일 입력 → 길이/비트레이트 벡터 생성
        #video_lengths = np.full(self.n_videos, uniform_length_sec, dtype=np.float64)
        #bitrates_kbps = np.full(self.n_videos, uniform_bitrate_kbps, dtype=np.float64)

        # length: 300 ~ 600초
        video_lengths = np.random.uniform(
            low=uniform_length_sec,
            high=uniform_length_sec,
            size=self.n_videos
        )

        self.video_length = video_lengths

        # bitrate: 8000 ~ 10000 kbps
        bitrates_kbps = np.random.uniform(
            low=uniform_bitrate_kbps,
            high=uniform_bitrate_kbps + 2000,
            size=self.n_videos
        )
        

        self.time_size, self.vmaf = self._predict_time_vmaf_matrices(video_lengths, bitrates_kbps)
        # 반올림(옵션)
        self.time_size = np.round(self.time_size, 3).astype(np.float32)
        self.vmaf = np.round(self.vmaf, 2).astype(np.float32)

        total_time_size = np.sum(self.time_size)
        total_time_budget = total_time_size * SIZE_FACTOR
        self.time_limit = np.full(self.n_slots, total_time_budget / self.n_slots, dtype=np.float32)  # (n_slots,)
        
        #print(self.time_size)
        #print(self.vmaf)
        #exit()
        
        
        # 데드라인 (균등 분배 대신) 가우시안 분포 기반 랜덤 배정
        mu = self.n_deadline / 2     # 평균 (예: 5)
        sigma = self.n_deadline / 3  # 표준편차 (예: 3.3)
        raw = np.random.normal(mu, sigma, self.n_videos)
        # 실수 → 정수 변환 후 범위 제한
        self.deadline = np.clip(raw.astype(int), 0, self.n_deadline - 1)
        
        # Action Space
        self.n_combos = 64     # 6비트 → ver1..6 중 저장 조합, ver0은 항상 포함
        #self.n_slots  = 100
        self.n_slots  = self.n_deadline * self.n_servers
        self.action_space = spaces.Discrete(self.n_combos)

        state_dim = self.n_combos * 3 + 13 + 1

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(state_dim,),
            dtype=np.float32
        )

        # ── add in __init__ (or as class fields)
        self._cached_combo_vid = -1
        self._combo_sizes = np.zeros(self.n_combos, dtype=np.float32)

        if seed is not None:
            np.random.seed(seed)

        self.reset()


    @staticmethod
    def _make_features(df_like: pd.DataFrame) -> pd.DataFrame:
        # RTF 모델: 길이/BR/해상도 모두 사용
        return pd.DataFrame({
            "log_len": np.log1p(df_like["video_length_sec"].astype(float)),
            "log_br":  np.log1p(df_like["original_bitrate_kbps"].astype(float)),
            "log_res": np.log1p(df_like["target_resolution"].astype(float)),
        })

    # ---------- 내부: CSV로부터 모델 1회 학습 ----------
    def _train_models_from_csv(self, csv_path: str, random_state: int = 42):
        if (self._m_time is not None) and (self._m_vmaf is not None):
            return

        df = pd.read_csv(csv_path)
        # 안전 보정
        df = df.dropna(subset=[
            "video_id","video_length_sec","original_bitrate_kbps","target_resolution",
            "transcoding_time_sec","vmaf_score"
        ]).copy()
        df["vmaf_score"] = df["vmaf_score"].clip(0, 100)
        df["rtf"] = df["transcoding_time_sec"] / df["video_length_sec"].replace(0, np.nan)

        # 특징
        X_all = self._make_features(df)
        y_rtf  = df["rtf"].values
        y_vmaf = df["vmaf_score"].values

        # video_id 기준 그룹 분할(누수 방지)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_idx, _ = next(gss.split(X_all, groups=df["video_id"]))

        X_tr = X_all.iloc[train_idx]
        y_rtf_tr  = y_rtf[train_idx]
        y_vmaf_tr = y_vmaf[train_idx]

        # 모델 학습
        m_time = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        )
        m_vmaf = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=random_state
        )
        m_time.fit(X_tr, y_rtf_tr)
        # VMAF는 길이에 의존시키지 않음(길이 제외)
        m_vmaf.fit(X_tr[["log_br","log_res"]], y_vmaf_tr)

        self._m_time = m_time
        self._m_vmaf = m_vmaf

    # ---------- 내부: (비디오×버전) 매트릭스 예측 ----------
    def _predict_time_vmaf_matrices(self, video_lengths_sec: np.ndarray, bitrates_kbps: np.ndarray):
        """
        입력:  length[k], bitrate[k]  (k=0..n_videos-1), 전 비디오 동일 값도 OK
        출력: time_size[n_videos, n_ver], vmaf[n_videos, n_ver]
        """
        assert self._m_time is not None and self._m_vmaf is not None, "Models must be trained first."
        assert video_lengths_sec.shape[0] == self.n_videos
        assert bitrates_kbps.shape[0] == self.n_videos

        # (비디오×해상도) 배치 구성
        vid_idx = np.repeat(np.arange(self.n_videos), self.n_ver)
        res_tile = np.tile(self._resolutions, self.n_videos)

        df_in = pd.DataFrame({
            "video_length_sec": video_lengths_sec[vid_idx],
            "original_bitrate_kbps": bitrates_kbps[vid_idx],
            "target_resolution": res_tile
        })
        X = self._make_features(df_in)

        # RTF → 시간(초)
        rtf_pred = self._m_time.predict(X.values)
        time_pred = rtf_pred * df_in["video_length_sec"].values

        # VMAF(0~100)
        vmaf_pred = self._m_vmaf.predict(X[["log_br","log_res"]].values)
        vmaf_pred = np.clip(vmaf_pred, 0, 100)

        # (n_videos, n_ver)로 재배열
        time_mat = time_pred.reshape(self.n_videos, self.n_ver)
        vmaf_mat = vmaf_pred.reshape(self.n_videos, self.n_ver)
        return time_mat, vmaf_mat
    


    def get_HUF_value(self):
        return self.HUF_value

    def get_HUTF_value(self):
        return self.HUTF_value

    def get_HUF_strict_value(self):
        return self.HUF_strict_value

    def get_HUTF_strict_value(self):
        return self.HUTF_strict_value

    def get_MCKP_predict_value(self):
        return self.MCKP_predict_value

    def get_MCKP_true_value(self):
        return self.MCKP_true_value

    def get_PPO_value(self):
        return self.PPO_value



    def _resample_deadlines(self):
        mu = self.n_deadline / 2
        sigma = self.n_deadline / 3
        raw = np.random.normal(mu, sigma, self.n_videos)
        self.deadline = np.clip(raw.astype(int), 0, self.n_deadline - 1)    

    def _combo_size_vecs(self, vid: int):
        """현재 비디오 vid에 대한 (total_size, delta_same) 각각 64차원, 둘 다 [0,1] 정규화."""
        self._ensure_combo_sizes(vid)
        size0 = float(self.time_size[vid, 0])

        total = self._combo_sizes.copy()                           # 절대 크기
        delta_same = np.maximum(total - size0, 0.0)                # in-place 증분

        # 간단 정규화(각 벡터의 per-video max로 나눔; 0 division 방지)
        total_norm = total / (total.max() + 1e-6)
        delta_norm = delta_same / (delta_same.max() + 1e-6)

        return total_norm.astype(np.float32), delta_norm.astype(np.float32)

    def _ensure_combo_sizes(self, vid: int) -> None:
        """현재 비디오 vid에 대해 콤보(64개) 총 용량을 캐시. vid 바뀔 때만 갱신."""
        if self._cached_combo_vid == vid:
            return
        sizes = np.zeros(self.n_combos, dtype=np.float32)
        for c in range(self.n_combos):
            vs = self._combo_to_versions(c)  # 항상 ver0 포함
            sizes[c] = float(np.sum(self.time_size[vid, vs]))
        self._combo_sizes[:] = sizes
        self._cached_combo_vid = vid

    def _decode_action(self, a):
        combo = a // self.n_slots      # 0..63
        slot  = a %  self.n_slots      # 0..99
        return combo, slot

    def _combo_to_versions(self, combo_idx):
        # ver0은 항상 포함, 6비트는 ver1..6
        versions = [0]
        for b in range(6):
            if (combo_idx >> b) & 1:
                versions.append(b + 1)
        return sorted(versions)
    
    def _combo_fallback_qoe_vec(self, vid, use="pred"):
        # 조합별로: 저장세트(항상 ver0 포함) 기준 fallback QoE 계산
        dist = self.pred_popularity if use=="pred" else self.true_popularity
        out = np.zeros(self.n_combos, dtype=np.float32)
        for c in range(self.n_combos):
            saved = self._combo_to_versions(c)
            # fallback 사용된 최종 제공 버전 매핑
            fb = []
            for ver in range(self.n_ver):
                if ver in saved:
                    fb.append(ver)
                else:
                    lowers = [v for v in saved if v < ver]
                    fb.append(max(lowers) if lowers else 0)
            out[c] = float(np.sum(dist[vid] * self.vmaf[vid, fb]))
        # 정규화(선택): 0~1로 스케일
        # mx = (self.vmaf[vid].max() + 1e-6)
        return out / 10.0

    def _versions_of_combo(self, combo_idx: int):
        vs = [0]
        for b in range(6):
            if (combo_idx >> b) & 1:
                vs.append(b + 1)
        return sorted(vs)

    def _combo_size(self, vid: int, combo_idx: int) -> float:
        vs = self._versions_of_combo(combo_idx)
        return float(np.sum(self.time_size[vid, vs]))

    def _leftover_after(self, slot: int, add_size: float) -> float:
        return float(self.time_limit[slot] - (self.time_used[slot] + add_size))

    def _pick_slot_for_combo(self, v: int, combo_idx: int):
        """
        정책(업데이트):
        1) in-place(앵커 유지): old 에서 필요한 증분 여유 있으면 그대로
        2) 같은 데드라인 창에서 '가까운 그룹부터'(d, d-1, ..., lo) 그룹별 best-fit
        3) 그래도 없으면 ≤ d 전체에서 worst-fit 재앵커
        4) 전혀 불가 → ('fallback', 0, old_slot)
        """
        d = int(self.deadline[v])
        old = int(self.lowest_version_slot[v])
        size0 = float(self.time_size[v, 0])
        total_size = self._combo_size(v, combo_idx)

        # 1) in-place: 증분만 필요
        inc_same = total_size - size0
        if inc_same <= (self.time_limit[old] - self.time_used[old] + 1e-9):
            return ('inplace', combo_idx, old)

        # 2) 창 내에서 '가까운 그룹부터' 그룹별 best-fit
        lo = max(0, d - SLOT_CONCENTRATION)
        for g in range(d, lo - 1, -1):  # d, d-1, ..., lo
            cand = [
                i for i in range(self.n_slots)
                if (i % self.n_deadline) == g
                and i != old
                and total_size <= (self.time_limit[i] - self.time_used[i] + 1e-9)
            ]
            if cand:
                # best-fit (남는 용량 최소). 동률이면 현재 사용률이 낮은 슬롯을 선호(단편화+밸런스)
                s = min(
                    cand,
                    key=lambda i: (
                        self._leftover_after(i, total_size),
                        self.time_used[i] / (self.time_limit[i] + 1e-9)
                    )
                )
                return ('reanchor_bestfit', combo_idx, s)

        # 3) ≤ d 전체에서 worst-fit
        feasible = [
            i for i in range(self.n_slots)
            if (i % self.n_deadline) <= d
            and i != old
            and total_size <= (self.time_limit[i] - self.time_used[i] + 1e-9)
        ]
        if feasible:
            s = max(feasible, key=lambda i: self._leftover_after(i, total_size))
            return ('reanchor_worstfit', combo_idx, s)

        # 4) 완전 불가 → 콤보 0 폴백(앵커 유지)
        return ('fallback', 0, old)

    def summarize_video_rank_noise(self, topk=20):
        t = self.true_popularity.sum(axis=1)
        p = self.pred_popularity.sum(axis=1)
        order_t = np.argsort(-t)
        order_p = np.argsort(-p)
        top_t = set(order_t[:topk]); top_p = set(order_p[:topk])
        jacc = len(top_t & top_p) / max(1, len(top_t | top_p))
        rank_t = np.empty_like(order_t); rank_t[order_t] = np.arange(1, len(order_t)+1)
        rank_p = np.empty_like(order_p); rank_p[order_p] = np.arange(1, len(order_p)+1)
        rho = float(np.corrcoef(rank_t, rank_p)[0, 1])
        print(f"[Video-Noise] J@{topk}={jacc:.3f}, Spearman ρ={rho:.3f}")

    def print_video_rank_two_columns(self, topn=20):
        true_video = self.true_popularity.sum(axis=1)
        pred_video = self.pred_popularity.sum(axis=1)
        order_true = np.argsort(-true_video)[:topn]
        order_pred = np.argsort(-pred_video)[:topn]
        print(f"\n[Video-level Ranking] Top-{topn}")
        print(f"{'TRUE rank (video ids):':24s}{order_true}")
        print(f"{'PRED rank (video ids):':24s}{order_pred}")

    def _util_stats(self, time_used: np.ndarray):
        util = time_used / (self.time_limit + 1e-9)
        avg = float(util.mean())
        mn = float(util.min())
        mx = float(util.max())
        zeros = np.where(util == 0.0)[0].tolist()
        return avg, mn, mx, zeros

    def _log_util_stats(self, time_used: np.ndarray, tag: str):
        avg, mn, mx, zeros = self._util_stats(time_used)
        print(f"📊[{tag}] 슬롯 이용률 - 평균: {avg:.2f}, 최소: {mn:.2f}, 최대: {mx:.2f}")
        if zeros:
            print(f"🛑[{tag}] 이용률 0인 슬롯: {zeros}")
        else:
            print(f"✅[{tag}] 모든 슬롯이 일부라도 사용됨")

    def print_rank_two_columns(self, vid_ids=None):
        """
        각 비디오에 대해 (TRUE rank, PRED rank)를 두 칼럼으로 출력.
        예: [6 5 4 3 2 1 0] 형태 (내림차순 랭크 인덱스)
        """
        true = self.true_popularity
        pred = self.pred_popularity
        n_videos, _ = true.shape

        if vid_ids is None:
            vid_ids = np.arange(min(5, n_videos))  # 기본 5개 샘플

        for vid in vid_ids:
            order_t = np.argsort(-true[vid])
            order_p = np.argsort(-pred[vid])
            print(f"\n[Video {vid}]")
            print(f"{'TRUE rank:':12s}{order_t}")
            print(f"{'PRED rank:':12s}{order_p}")

    def _print_video_rank_topk(self, k=20):
        t = self.true_popularity.sum(axis=1)
        p = self.pred_popularity.sum(axis=1)
        print("\n[Video-level Ranking] Top-{}" .format(k))
        print("TRUE:", np.argsort(-t)[:k])
        print("PRED:", np.argsort(-p)[:k])

    def summarize_video_noise_profile(self, head_pct=0.05, mid_range=(0.40, 0.60), tail_pct=0.05, show_deciles=True):
        """
        비디오-레벨 p_true vs p_pred의 변화가 head/mid/tail 어디에 많이 들어갔는지 요약.
        - |Δ|, 상대변화, TVD, head/mid/tail의 질량 변화를 출력
        - (hetero 계열일 때) 가중치 w와 |Δ|의 상관도 출력
        """
        eps = 1e-12
        pt = self.video_popularity_true.astype(np.float64)
        pp = self.video_popularity_pred.astype(np.float64)
        assert np.isclose(pt.sum(), 1.0, atol=1e-6), "pt sum!=1"
        assert np.isclose(pp.sum(), 1.0, atol=1e-6), "pp sum!=1"

        n = len(pt)
        order = np.argsort(-pt)              # true 인기 내림차순(랭크)
        delta = np.abs(pp - pt)
        rel = np.abs((pp - pt) / (pt + eps)) # 상대변화

        # 버킷 인덱스
        H = max(1, int(n * head_pct))
        T = max(1, int(n * tail_pct))
        M0 = int(n * mid_range[0])
        M1 = int(n * mid_range[1])
        idx_head = order[:H]
        idx_mid  = order[M0:M1]
        idx_tail = order[-T:]

        # 통계 함수
        def stats(idx):
            return dict(
                mean_abs_delta=float(delta[idx].mean()),
                mean_rel_delta=float(rel[idx].mean()),
                mass_true=float(pt[idx].sum()),
                mass_pred=float(pp[idx].sum())
            )

        s_head = stats(idx_head)
        s_mid  = stats(idx_mid)
        s_tail = stats(idx_tail)

        tv_total = 0.5 * float(np.abs(pp - pt).sum())
        tv_head  = 0.5 * float(np.abs(pp[idx_head] - pt[idx_head]).sum())
        tv_mid   = 0.5 * float(np.abs(pp[idx_mid]  - pt[idx_mid]).sum())
        tv_tail  = 0.5 * float(np.abs(pp[idx_tail] - pt[idx_tail]).sum())

        print("---- Video popularity noise profile ----")
        print(f"Total Variation Distance (all): {tv_total:.6f}")
        print(f"  • Head({head_pct*100:.1f}%): TV={tv_head:.6f} | |Δ|={s_head['mean_abs_delta']:.6e} | rel={s_head['mean_rel_delta']:.3f} | mass true→pred: {s_head['mass_true']:.3f}→{s_head['mass_pred']:.3f}")
        print(f"  • Mid ({int(mid_range[0]*100)}~{int(mid_range[1]*100)}%): TV={tv_mid:.6f} | |Δ|={s_mid['mean_abs_delta']:.6e} | rel={s_mid['mean_rel_delta']:.3f} | mass: {s_mid['mass_true']:.3f}→{s_mid['mass_pred']:.3f}")
        print(f"  • Tail({tail_pct*100:.1f}%): TV={tv_tail:.6f} | |Δ|={s_tail['mean_abs_delta']:.6e} | rel={s_tail['mean_rel_delta']:.3f} | mass: {s_tail['mass_true']:.3f}→{s_tail['mass_pred']:.3f}")

        # decile별 평균 |Δ| (선택)
        if show_deciles:
            dec = 10
            print("Decile mean |Δ| by true-rank (1=head → 10=tail):")
            for d in range(dec):
                lo = int(n * d/dec); hi = int(n * (d+1)/dec)
                m = float(delta[order[lo:hi]].mean())
                print(f"  D{d+1}: {m:.6e}")

        # hetero 계열이면, 설계 가중치 w와 |Δ|의 상관 확인
        if "hetero" in str(self.video_noise_model):
            mode_map = {"hetero_gaussian": "mid",
                        "hetero_gaussian_head": "head",
                        "hetero_gaussian_tail": "tail"}
            mode = mode_map.get(self.video_noise_model, "mid")
            w = self._hetero_weight(pt, mode=mode, alpha=getattr(self, "hetero_alpha", 1.0))
            corr = np.corrcoef(w, delta)[0, 1]
            print(f"corr(|Δ|, weight[{mode}]) = {float(corr):.3f}")

    def summarize_video_head_mid_tail_jaccard(self, k_list=(10, 20, 50, 100)):
        pt = self.video_popularity_true
        pp = self.video_popularity_pred
        order_t = np.argsort(-pt)
        order_p = np.argsort(-pp)

        print("Top-K Jaccard (video-level, true vs pred):")
        for k in k_list:
            s_t = set(order_t[:k]); s_p = set(order_p[:k])
            j = len(s_t & s_p) / max(1, len(s_t | s_p))
            print(f"  K={k:4d}: J={j:.3f}")

        # Head/Mid/Tail 질량(확률) 이동
        n = len(pt)
        head = int(0.05*n); tail = int(0.05*n)
        mid0, mid1 = int(0.40*n), int(0.60*n)
        H = order_t[:head]; M = order_t[mid0:mid1]; T = order_t[-tail:]
        def mass(idx, p): return float(p[idx].sum())
        print("Mass shift by rank buckets (true→pred):")
        print(f"  Head(Top5%): {mass(H, pt):.3f} → {mass(H, pp):.3f}")
        print(f"  Mid (40~60%): {mass(M, pt):.3f} → {mass(M, pp):.3f}")
        print(f"  Tail(Bot5%): {mass(T, pt):.3f} → {mass(T, pp):.3f}")


    def _hetero_weight(self, p, mode='mid', alpha=1.0):
        import numpy as np
        eps = 1e-12
        p = np.asarray(p, dtype=np.float64)

        if mode == 'mid':
            w = np.sqrt(p * (1.0 - p))                   # 중간에서 최대
        else:
            # 랭크 기반 가중: 큰 p가 상위(헤드)
            order = np.argsort(-p)                       # 내림차순
            r = np.empty_like(order, dtype=np.float64)
            r[order] = np.linspace(0.0, 1.0, len(p))     # 헤드~꼬리: 0→1
            if mode == 'head':
                w = (1.0 - r) ** alpha                   # 헤드 쪽 가중↑
            elif mode == 'tail':
                w = (r) ** alpha                         # 꼬리 쪽 가중↑
            else:
                w = np.ones_like(p)

        w /= (w.max() + eps)                             # [0,1] 정규화
        return w
    
    def _perturb_video_prob_hetero_gaussian(self, p, sigma=0.05, hetero_beta=0.5,
                                        mode='mid', alpha=1.0, floor=1e-12, rng=None):
        import numpy as np
        if rng is None:
            rng = np.random.default_rng()
        p = np.asarray(p, dtype=np.float64)
        eps = 1e-12

        w = self._hetero_weight(p, mode=mode, alpha=alpha)
        sig = sigma * ((1.0 - hetero_beta) + hetero_beta * w)

        q = np.clip(p + rng.normal(0.0, sig, size=p.shape), 0, None)
        q = np.maximum(q, floor)
        q = q / (q.sum() + eps)
        return q.astype(np.float32)

 
    # 비디오 인기도 변이 함수들 3개 
    def _perturb_video_prob_gaussian(self, p, sigma=0.05, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        q = p.astype(np.float64).copy()
        q = np.clip(q + rng.normal(0.0, sigma, size=q.shape), 0, None)
        s = q.sum()
        if s <= 0:
            q = np.ones_like(q) / len(q)
        else:
            q = q / s
        return q.astype(np.float32)
   

    # 버전 인기도 변이 함수 
    def _perturb_version_probs(self, base_probs, scale, kind, rng):

        probs = np.array(base_probs, dtype=np.float64)

        if scale > 0:
            if kind == "gaussian":
                noise = rng.normal(0, scale, size=len(probs))
                probs = np.clip(probs + noise, 0, None)

            elif kind == "dirichlet":
                alpha = np.clip(probs * (1.0 / scale), 1e-3, None)
                probs = rng.dirichlet(alpha)

            elif kind == "swap":
                probs = probs.copy()
                k = max(1, int(scale * len(probs)))
                idx = rng.choice(len(probs), size=2 * k, replace=False)
                for i in range(0, len(idx), 2):
                    probs[idx[i]], probs[idx[i+1]] = probs[idx[i+1]], probs[idx[i]]

        total = probs.sum()
        if total <= 0:
            # 모든 값이 0이면 균등분포로 초기화
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= total

        return probs

    def init_popularity(self, skewness=None):
        """
        TRUE(oracle) 분포와 PRED(예측) 분포를 비디오/버전 레벨에서 따로 만들고,
        최종 pred_popularity = (비디오 PRED) × (버전 PRED) 로 구성.
        """
        rng = np.random.default_rng()

        # ---------- (1) 비디오 인기도 TRUE : Zipf ----------
        if skewness is None:
            skewness = np.random.uniform(0.5, 1.0)
            print("Zipf parameter:",skewness)

        ranks = np.arange(1, self.n_videos + 1)
        p_video_true = (1.0 / (ranks ** skewness))
        p_video_true = (p_video_true / p_video_true.sum()).astype(np.float32)
        self.video_popularity_true = p_video_true  # (n_videos,)

        

        # ---------- (1') 비디오 인기도 PRED : noise_model 선택 ----------
        if  self.video_noise_model == "gaussian":
            self.video_popularity_pred = self._perturb_video_prob_gaussian(
                p_video_true,
                sigma=self.video_noise_param,
                rng=rng
            ).astype(np.float32)

        elif self.video_noise_model == "hetero_gaussian":
        # 기존: 중간 구간 강조
            self.video_popularity_pred = self._perturb_video_prob_hetero_gaussian(
                p_video_true, sigma=self.video_noise_param,
                hetero_beta=HETERO_BETA, mode='mid', alpha=getattr(self, "hetero_alpha", 1.0),
                floor=1e-6, rng=rng
            ).astype(np.float32)

        elif self.video_noise_model == "hetero_gaussian_head":
            self.video_popularity_pred = self._perturb_video_prob_hetero_gaussian(
                p_video_true, sigma=self.video_noise_param,
                hetero_beta=HETERO_BETA, mode='head', alpha=getattr(self, "hetero_alpha", 1.0),
                floor=1e-6, rng=rng
            ).astype(np.float32)

        elif self.video_noise_model == "hetero_gaussian_tail":
            self.video_popularity_pred = self._perturb_video_prob_hetero_gaussian(
                p_video_true, sigma=self.video_noise_param,
                hetero_beta=HETERO_BETA, mode='tail', alpha=getattr(self, "hetero_alpha", 1.0),
                floor=1e-6, rng=rng
            ).astype(np.float32)
        
        elif self.video_noise_model == "real_dataset_train":
            #rand_idx = random.randint(0, 39)
            #csv_path = rf"knn/{KNN}/knn_prediction_A_k{KNN}_{rand_idx}_.csv"
            #data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            #self.pred_rank = data[:,2]
            #self.video_popularity_pred = (1.0 / (self.pred_rank ** skewness))


            rand_idx = random.randint(0, 39)
            csv_path = rf"K241124/{KNN}/knn_prediction_A_k{KNN}_{rand_idx}_.csv"
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            self.video_popularity_true = data[:, 1]
            self.video_popularity_pred = data[:, 2]

            

        elif self.video_noise_model == "real_dataset_test":
            #rand_idx = random.randint(40, 49)
            #csv_path = rf"knn/{KNN}/knn_prediction_A_k{KNN}_{rand_idx}_.csv"
            #data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            #self.pred_rank = data[:,2]
            #self.video_popularity_pred = (1.0 / (self.pred_rank ** skewness))

            rand_idx = random.randint(40, 49)
            csv_path = rf"K241124/{KNN}/knn_prediction_A_k{KNN}_{rand_idx}_.csv"
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            self.video_popularity_true = data[:, 1]
            self.video_popularity_pred = data[:, 2]


        else:
            # 지정 안 하면 TRUE 그대로


            #self.video_popularity_pred = self.video_popularity_true.copy()

            rand_idx = random.randint(0, 39)
            csv_path = rf"K241124/{KNN}/knn_prediction_A_k{KNN}_{rand_idx}_.csv"
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            self.video_popularity_true = data[:, 1]
            self.video_popularity_pred = data[:, 2]

        

        # ---------- (2) 버전 인기도 TRUE : 중앙 버전 중심 가우시안 ----------
        centers = np.arange(self.n_ver)
        self.version_popularity_true = np.zeros((self.n_videos, self.n_ver), dtype=np.float32)
        center = (self.n_ver - 1) / 2.0    # MVP
        #center = self.n_ver - 1             # HVP
        #center = 0.0                       # LVP
        std = self.n_ver / 4.0
        for vid in range(self.n_videos):
            probs = np.exp(-0.5 * ((centers - center) / std) ** 2)
            probs /= probs.sum()
            self.version_popularity_true[vid] = probs

        # ---------- (2') 버전 인기도 PRED 교란 모델  ----------
        self.version_popularity_pred = np.zeros_like(self.version_popularity_true, dtype=np.float32)
        vm = self.version_noise_model
        for vid in range(self.n_videos):
            self.version_popularity_pred[vid] = self._perturb_version_probs(
                base_probs=self.version_popularity_true[vid],
                scale=self.version_noise_param,
                kind=vm,  # 또는 kind="dirichlet"/"gaussian"/"swap"으로 매핑
                rng=rng
            )
        
        # ---------- (3) 결합 ----------
        # TRUE 결합
        self.true_popularity = (self.version_popularity_true.T * self.video_popularity_true).T
        # PRED 결합
        pred_from_ver = (self.version_popularity_pred.T * self.video_popularity_pred).T
        self.pred_popularity = pred_from_ver

        # ---------- (4) 호환성 유지 ----------
        # 기존 코드가 self.popularity 를 참조하더라도 TRUE와 동일하도록 둡니다.
        self.popularity = self.true_popularity

        # 모두 안다라고 가정할때
        # self.pred_popularity = self.popularity
        

    

    def reset(self, seed=None, options=None):
        # ----- RNG -----
        if seed is not None:
            np.random.seed(seed)
        if TRAINING:
            # 멀티프로세스 학습 시 에피소드/프로세스별 시드 차별화
            np.random.seed(self.episode_count + os.getpid())
        if TESTING: 
            np.random.seed(self.episode_count + os.getpid())
        

        self.allocation_dict = {s: [] for s in range(self.n_slots)}

        '''
        self.vmaf = np.zeros((self.n_videos, self.n_ver), dtype=np.float32)
        vmaf_means = np.linspace(40, 100, self.n_ver)
        vmaf_std = 5.0

        for vid in range(self.n_videos):
            vmaf_values = np.clip(np.random.normal(loc=vmaf_means, scale=vmaf_std), 0, 100)
            vmaf_values = np.sort(vmaf_values)
            vmaf_values[-1] = 100.0
            self.vmaf[vid, :] = vmaf_values

        self.video_length = np.full(self.n_videos, 300, dtype=np.float32)
        base_time_size = (self.bitrates[None, :] * self.video_length[:, None]) / (8 * 1024)

        # Time Distribution
        size_noise_std = 0.000001
        rng = np.random.default_rng()
        mult = rng.lognormal(mean=0.0, sigma=size_noise_std, size=base_time_size.shape)
        self.time_size = (base_time_size * mult).astype(np.float32)
        self.time_size=base_time_size
        '''

        self.reset_size_factor = SIZE_FACTOR

        if TRAINING:
            self.reset_size_factor = np.random.uniform(0.2, 0.5)
            #self.reset_size_factor = SIZE_FACTOR
            total_time_size = np.sum(self.time_size)
            total_time_budget = total_time_size * self.reset_size_factor
            self.time_limit = np.full(self.n_slots, total_time_budget / self.n_slots, dtype=np.float32)  # (n_slots,)
        

        self._resample_deadlines() 

        # ----- 초기화 -----
        self.current_video = 0
        self.time_used = np.zeros(self.n_slots, dtype=np.float32)
        self.total_reward = 0.0
        # (선택) 기존 기록 초기화
        #self.allocation_dict = {s: [] for s in range(self.n_slots)}

        # ----- ver=0 선할당: 데드라인-호환 슬롯만 + 베스트핏 -----
        # Strict: slot_group == deadline[v]
        # Fallback: (필요 시) slot_group <= deadline[v]
        self.lowest_version_slot = np.zeros(self.n_videos, dtype=int)
        for v in range(self.n_videos):
            size0 = float(self.time_size[v, 0])
            d = int(self.deadline[v])

            # 1) strict 후보
            strict_candidates = [
                i for i in range(self.n_slots)
                if (i % self.n_deadline) == d and (self.time_used[i] + size0) <= self.time_limit[i]
            ]

            # 2) strict가 꽉 찼으면 완화(≤ d)
            if not strict_candidates:
                le_candidates = [
                    i for i in range(self.n_slots)
                    if (i % self.n_deadline) <= d and (self.time_used[i] + size0) <= self.time_limit[i]
                ]
            else:
                le_candidates = []

            candidates = strict_candidates if strict_candidates else le_candidates

            if candidates:
                # 베스트핏: 남는 용량 최소 슬롯 선택 → 단편화 감소
                def leftover(i):
                    return float(self.time_limit[i] - (self.time_used[i] + size0))
                idx = min(candidates, key=leftover)

                self.lowest_version_slot[v] = idx
                self.time_used[idx] += size0
                # (옵션) 기록
                # self.allocation_dict[idx].append((v, 0))
            else:
                # 모든 후보가 불가(드묾): 가장 여유로운 슬롯에 시도(오버 방지)
                idx = int(np.argmin(self.time_used / (self.time_limit + 1e-6)))
                if (self.time_used[idx] + size0) <= self.time_limit[idx]:
                    self.lowest_version_slot[v] = idx
                    self.time_used[idx] += size0
                    # self.allocation_dict[idx].append((v, 0))
                # else: ver=0조차 못 넣는 극히 예외 상황 → 이후 단계에서 자연스레 fallback 처리

        # ----- 인기도 초기화 -----  Variable Zipf
        if FIXED_ZIPF==1:
            skew_parameter = ZIPF_PARAMETER # 필요 시 조정
            self.init_popularity(skew_parameter) 
        else:
            self.init_popularity(None)
        
        self.base_qoe_true = (self.true_popularity.sum(axis=1) * self.vmaf[:, 0]).astype(np.float64)
  
        # ----- 기대 QoE(스케일링 기준) 계산 -----
        self.expected_total_reward = 0.0
        for v in range(self.n_videos):
            self.expected_total_reward += float(np.sum(self.true_popularity[v] * self.vmaf[v]))
        
        # Noise Statistics 
        """
        if TESTING and self.episode_count == 0:
            self.summarize_video_noise_profile()
            self.summarize_video_head_mid_tail_jaccard()
        """

        # ----- 초기 상태/마스크 반환 -----
        self.mask_done = False
        self.state = self._compute_state()
        info = {'action_mask': self.get_valid_action_mask()}

        return self.state.astype(np.float32), info


    def get_valid_action_mask(self):
        v = int(self.current_video)
        mask = np.zeros(self.n_combos, dtype=bool)

        # 각 콤보가 폴백 없이 배치 가능한지 검사
        for c in range(self.n_combos):
            plan = self._pick_slot_for_combo(v, c)
            if plan[0] != 'fallback':
                mask[c] = True

        # 안전장치: combo=0 은 항상 허용
        mask[0] = True
        return mask

    # Average utilization for each deadline slot 
    def _avg_util_by_deadline(self):
        util = self.time_used / (self.time_limit + 1e-9)  # [0,1] 근처
        out = np.zeros(self.n_deadline, dtype=np.float32)
        for d in range(self.n_deadline):
            mask = (self.slot_groups == d)
            out[d] = util[mask].mean() if np.any(mask) else 0.0
        return np.clip(out, 0.0, 1.0)

    # Observation Space
    def _compute_state(self):
        vid = int(self.current_video)

        combo_qoe = self._combo_fallback_qoe_vec(vid, use="pred").astype(np.float32)  # (64,)
        total_norm, delta_norm = self._combo_size_vecs(vid)                           # (64,), (64,)

        avg_util_global = np.mean(self.time_used / (self.time_limit + 1e-6)).astype(np.float32)
        avg_util_global = np.array([avg_util_global], dtype=np.float32)

        denom = max(1, self.n_videos - 1)
        progress = np.array([self.current_video / denom], dtype=np.float32)

        avg_util_by_deadline = self._avg_util_by_deadline()  # (n_deadline,)
        norm_deadline = np.array([ self.deadline[vid] / (self.n_deadline - 1) ], dtype=np.float32)  # (1,)

        avg_util_global = np.clip(avg_util_global, 0.0, 1.0)
        progress = np.clip(progress, 0.0, 1.0)
        norm_deadline = np.clip(norm_deadline, 0.0, 1.0)

        size_factor = np.array([self.reset_size_factor], dtype=np.float32)


        state = np.concatenate([
            combo_qoe, total_norm, delta_norm, 
            avg_util_global, progress,
            avg_util_by_deadline, norm_deadline, size_factor
        ]).astype(np.float32)

        return state




    def step(self, action):
        v = int(self.current_video)
        combo = int(action)               
        plan_type, used_combo, slot = self._pick_slot_for_combo(v, combo)
        versions = self._versions_of_combo(used_combo)

        old_slot = int(self.lowest_version_slot[v])
        size0 = float(self.time_size[v, 0])
        total_size = float(np.sum(self.time_size[v, versions]))

        # 배치 적용
        if slot == old_slot:
            # in-place 또는 fallback(ver0 유지)
            add_size = total_size - size0   # fallback(used_combo=0)면 0
            if add_size > 0:
                self.time_used[slot] += add_size
        else:
            # 재앵커: old에서 ver0 제거 후 새 슬롯에 전체 추가
            self.time_used[old_slot] -= size0
            self.time_used[old_slot] = max(0.0, self.time_used[old_slot])
            self.time_used[slot] += total_size
            self.lowest_version_slot[v] = slot

        # 배치 기록
        self.allocation_dict[slot].append((v, versions))
        


        # 리워드(폴백 반영)
        fb = []
        for ver in range(self.n_ver):
            if ver in versions:
                fb.append(ver)
            else:
                lower = [x for x in versions if x < ver]
                fb.append(max(lower) if lower else 0)

        # 기존 fb 계산 그대로
        #new_qoe = float(np.sum(self.pred_popularity[v] * self.vmaf[v, fb]))
        new_qoe = float(np.sum(self.true_popularity[v] * self.vmaf[v, fb]))
        delta_qoe = new_qoe - float(self.base_qoe_true[v])          # ← 인기도 가중 Δ

        # 전역 정규화로 스케일만 안정화(인기도 차이는 유지)
        reward = (delta_qoe / (self.expected_total_reward + 1e-6)) * 1000.0

        # 진행
        self.total_reward += reward
        self.current_video += 1

        done = (self.current_video >= self.n_videos)   # 마지막까지 처리하면 종료
        truncated = False

        if not done:
            self.state = self._compute_state()
            info = {'action_mask': self.get_valid_action_mask()}
        else:
            info = {
                'terminal_observation': self.state.copy(),
                'episode': {'r': float(self.total_reward), 'l': int(self.n_videos)}
            }
            self.episode_count += 1
            #print(f"🎬 에피소드 {self.episode_count} 종료 - 총 보상: {self.total_reward:.2f}")
            
            
            
            if TESTING:
                
                self.HUF_value, self.HUF_alloc               = self.greedy_allocation_HUF()
                self.HUTF_value, self.HUTF_alloc             = self.greedy_allocation_HUTF()
                self.HUF_strict_value, self.HUF_strict_alloc = self.greedy_allocation_HUF_strictDL()
                self.HUTF_strict_value, self.HUTF_strict_alloc = self.greedy_allocation_HUTF_strictDL()
                self.MCKP_predict_value, self.MCKP_predict_alloc = self.greedy_allocation_combo_delta(mode="mckp")
                self.MCKP_true_value, self.MCKP_true_alloc      = self.greedy_allocation_combo_delta(mode="oracle")
                

                '''
                self.HUF_value = self.greedy_allocation_HUF()
                self.HUTF_value = self.greedy_allocation_HUTF()
                self.HUF_strict_value = self.greedy_allocation_HUF_strictDL()
                self.HUTF_strict_value = self.greedy_allocation_HUTF_strictDL()
                self.MCKP_predict_value = self.greedy_allocation_combo_delta(mode="mckp")     # 예측 분포 기반 (MCKP)
                self.MCKP_true_value = self.greedy_allocation_combo_delta(mode="oracle")   # 실제 분포 기반 (Oracle)
                '''
                actual_qoe, full_qoe = self.compute_final_qoe(self.allocation_dict)
                rl_scaled10 = float((actual_qoe / (full_qoe + 1e-6)) * 10.0)  # ← 베이스라인(그리디)와 동일 스케일
                #print("DRL:",rl_scaled10*100)
                self.PPO_value = rl_scaled10*100
                #print("time used : ", np.sum(self.time_used))
                #exit()


            # ✅ 슬롯 이용률 요약 출력
            slot_utilization = self.time_used / self.time_limit
            avg_util = np.mean(slot_utilization)
            min_util = np.min(slot_utilization)
            max_util = np.max(slot_utilization)

            zero_util_slots = np.where(slot_utilization == 0.0)[0]  # 이용률이 0인 슬롯 인덱스 추출

            #print(f"📊 슬롯 이용률 요약 - 평균: {avg_util:.2f}, 최소: {min_util:.2f}, 최대: {max_util:.2f}")

            if len(zero_util_slots) > 0:
                aaa=1
                #print(f"🛑 이용률 0인 슬롯: {zero_util_slots.tolist()}")
  
        return self.state.astype(np.float32), reward, done, truncated, info

    def greedy_allocation_HUF(self):
        time_used = np.zeros(self.n_slots, dtype=np.float32)
        allocation_dict = {i: [] for i in range(self.n_slots)}
        assigned = np.zeros((self.n_videos, self.n_ver), dtype=bool)

        # (0-1) lowest version 선할당 (ver=0), load balancing + deadline 고려
        lowest_slot = {}
        for v in range(self.n_videos):
            size = self.time_size[v, 0]
            candidate_slots = [
                i for i in range(self.n_slots)
                if self.deadline[v] >= (i % self.n_deadline) and time_used[i] + size <= self.time_limit[i]
            ]
            if candidate_slots:
                idx = min(candidate_slots, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))
                time_used[idx] += size
                allocation_dict[idx].append((v, 0))
                assigned[v, 0] = True
                lowest_slot[v] = idx

        # (1) score 계산 및 정렬
        score_list = []
        for v in range(self.n_videos):
            for ver in range(1, self.n_ver):  # lowest 제외
                #score = self.popularity[v, ver] * self.vmaf[v, ver]
                score = self.pred_popularity[v, ver] * self.vmaf[v, ver]
                score_list.append((score, v, ver))
        score_list.sort(reverse=True)

        # (2) 비디오별 최초 등장 ver과 이후 처리
        slot_per_video = {}  # v: slot index

        for _, v, ver in score_list:
            if assigned[v, ver]:
                continue
            size = self.time_size[v, ver]

            if v not in slot_per_video:
            # 최초 등장한 비디오인 경우: lowest + 현재 ver 재할당 시도
                lowest_size = self.time_size[v, 0]


                candidates = [
                    i for i in range(self.n_slots)
                    if self.deadline[v] >= (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]
                ]

                if candidates:
                    # 후보 중 현재 채움 비율이 가장 낮은 슬롯 선택
                    idx = min(candidates, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))

                    if v in lowest_slot:
                        prev = lowest_slot[v]
                        time_used[prev] -= self.time_size[v, 0]
                        allocation_dict[prev] = [item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)]
                        assigned[v, 0] = False

                    time_used[idx] += lowest_size + size
                    allocation_dict[idx].append((v, 0))
                    allocation_dict[idx].append((v, ver))
                    assigned[v, 0] = True
                    assigned[v, ver] = True
                    slot_per_video[v] = idx



                '''
                for i in range(self.n_slots):
                    if self.deadline[v] >= (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]:
                    # 기존 lowest 할당 취소
                        if v in lowest_slot:
                            prev = lowest_slot[v]
                            time_used[prev] -= self.time_size[v, 0]
                            allocation_dict[prev] = [
                                item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)
                            ]
                            assigned[v, 0] = False

                    # 재할당
                        time_used[i] += lowest_size + size
                        allocation_dict[i].append((v, 0))
                        allocation_dict[i].append((v, ver))
                        assigned[v, 0] = True
                        assigned[v, ver] = True
                        slot_per_video[v] = i
                        break
                '''
            else:
            # 이미 등장한 비디오의 경우 해당 슬롯에만 할당
                i = slot_per_video[v]
                if time_used[i] + size <= self.time_limit[i]:
                    time_used[i] += size
                    allocation_dict[i].append((v, ver))
                    assigned[v, ver] = True

        # (3) reward 계산 (fallback 포함)
        total_reward = 0.0
        for v in range(self.n_videos):
            allocated_versions = [ver for i in range(self.n_slots)
                                  for (vid, ver) in allocation_dict[i] if vid == v]
            fallback_versions = []
            for ver in range(self.n_ver):
                if ver in allocated_versions:
                    fallback_versions.append(ver)
                else:
                    lower = [v2 for v2 in allocated_versions if v2 < ver]
                    fallback_versions.append(max(lower) if lower else 0)

            #version_popularities = self.popularity[v]
            version_popularities = self.true_popularity[v]
            version_vmaf = self.vmaf[v, fallback_versions]
            reward = np.sum(version_popularities * version_vmaf) * 100
            total_reward += reward

        # (4) 스케일링
        scaled_reward = (total_reward / (self.expected_total_reward + 1e-6)) * 10.0
        #print(f"🎯 그리디(HUF) 총 QoE (스케일된): {scaled_reward:.2f}")
        # self._log_util_stats(time_used, "Greedy HUF")
        
        return scaled_reward, allocation_dict
    

    def greedy_allocation_HUTF(self):
        time_used = np.zeros(self.n_slots, dtype=np.float32)
        allocation_dict = {i: [] for i in range(self.n_slots)}
        assigned = np.zeros((self.n_videos, self.n_ver), dtype=bool)

        # (0-1) lowest version 선할당 (ver=0), load balancing + deadline 고려
        lowest_slot = {}
        for v in range(self.n_videos):
            size = self.time_size[v, 0]
            candidate_slots = [
                i for i in range(self.n_slots)
                if self.deadline[v] >= (i % self.n_deadline) and time_used[i] + size <= self.time_limit[i]
            ]
            if candidate_slots:
                idx = min(candidate_slots, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))
                time_used[idx] += size
                allocation_dict[idx].append((v, 0))
                assigned[v, 0] = True
                lowest_slot[v] = idx

        # (1) score 계산 및 정렬
        score_list = []
        for v in range(self.n_videos):
            for ver in range(1, self.n_ver):  # lowest 제외
                #score = (self.popularity[v, ver] * self.vmaf[v, ver]) / (self.time_size[v, ver] + 1e-6)
                score = (self.pred_popularity[v, ver] * self.vmaf[v, ver]) / (self.time_size[v, ver] + 1e-6)
                score_list.append((score, v, ver))
        score_list.sort(reverse=True)

        # (2) 비디오별 최초 등장 ver과 이후 처리
        slot_per_video = {}  # v: slot index

        for _, v, ver in score_list:
            if assigned[v, ver]:
                continue
            size = self.time_size[v, ver]

            if v not in slot_per_video:
            # 최초 등장한 비디오인 경우: lowest + 현재 ver 재할당 시도
                lowest_size = self.time_size[v, 0]


                candidates = [
                    i for i in range(self.n_slots)
                    if self.deadline[v] >= (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]
                ]

                if candidates:
                    # 후보 중 현재 채움 비율이 가장 낮은 슬롯 선택
                    idx = min(candidates, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))

                    if v in lowest_slot:
                        prev = lowest_slot[v]
                        time_used[prev] -= self.time_size[v, 0]
                        allocation_dict[prev] = [item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)]
                        assigned[v, 0] = False

                    time_used[idx] += lowest_size + size
                    allocation_dict[idx].append((v, 0))
                    allocation_dict[idx].append((v, ver))
                    assigned[v, 0] = True
                    assigned[v, ver] = True
                    slot_per_video[v] = idx




                '''
                for i in range(self.n_slots):
                    if self.deadline[v] >= (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]:
                    # 기존 lowest 할당 취소
                        if v in lowest_slot:
                            prev = lowest_slot[v]
                            time_used[prev] -= self.time_size[v, 0]
                            allocation_dict[prev] = [
                                item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)
                            ]
                            assigned[v, 0] = False

                    # 재할당
                        time_used[i] += lowest_size + size
                        allocation_dict[i].append((v, 0))
                        allocation_dict[i].append((v, ver))
                        assigned[v, 0] = True
                        assigned[v, ver] = True
                        slot_per_video[v] = i
                        break
                '''
            else:
            # 이미 등장한 비디오의 경우 해당 슬롯에만 할당
                i = slot_per_video[v]
                if time_used[i] + size <= self.time_limit[i]:
                    time_used[i] += size
                    allocation_dict[i].append((v, ver))
                    assigned[v, ver] = True

        # (3) reward 계산 (fallback 포함)
        total_reward = 0.0
        for v in range(self.n_videos):
            allocated_versions = [ver for i in range(self.n_slots)
                                  for (vid, ver) in allocation_dict[i] if vid == v]
            fallback_versions = []
            for ver in range(self.n_ver):
                if ver in allocated_versions:
                    fallback_versions.append(ver)
                else:
                    lower = [v2 for v2 in allocated_versions if v2 < ver]
                    fallback_versions.append(max(lower) if lower else 0)

            #version_popularities = self.popularity[v]
            version_popularities = self.true_popularity[v]
            version_vmaf = self.vmaf[v, fallback_versions]
            reward = np.sum(version_popularities * version_vmaf) * 100
            total_reward += reward

        # (4) 스케일링
        scaled_reward = (total_reward / (self.expected_total_reward + 1e-6)) * 10.0
        #print(f"🎯 그리디(HUTF) 총 QoE (스케일된): {scaled_reward:.2f}")
        # self._log_util_stats(time_used, "Greedy HUTF")
        
        return scaled_reward, allocation_dict

    def greedy_allocation_HUF_strictDL(self):
        time_used = np.zeros(self.n_slots, dtype=np.float32)
        allocation_dict = {i: [] for i in range(self.n_slots)}
        assigned = np.zeros((self.n_videos, self.n_ver), dtype=bool)

        # (0-1) lowest version 선할당 (ver=0), load balancing + strict deadline
        lowest_slot = {}
        for v in range(self.n_videos):
            size = self.time_size[v, 0]
            candidate_slots = [
                i for i in range(self.n_slots)
                if self.deadline[v] == (i % self.n_deadline) and time_used[i] + size <= self.time_limit[i]
            ]
            if candidate_slots:
                idx = min(candidate_slots, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))
                time_used[idx] += size
                allocation_dict[idx].append((v, 0))
                assigned[v, 0] = True
                lowest_slot[v] = idx

        # (1) score 계산 및 정렬
        score_list = []
        for v in range(self.n_videos):
            for ver in range(1, self.n_ver):
                #score = self.popularity[v, ver] * self.vmaf[v, ver]
                score = self.pred_popularity[v, ver] * self.vmaf[v, ver]
                score_list.append((score, v, ver))
        score_list.sort(reverse=True)

        # (2) 비디오별 최초 등장 ver과 이후 처리
        slot_per_video = {}

        for _, v, ver in score_list:
            if assigned[v, ver]:
                continue
            size = self.time_size[v, ver]

            if v not in slot_per_video:
                lowest_size = self.time_size[v, 0]

                candidates = [
                    i for i in range(self.n_slots)
                    if self.deadline[v] == (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]
                ]

                if candidates:
                    # 후보 중 현재 채움 비율이 가장 낮은 슬롯 선택
                    idx = min(candidates, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))

                    if v in lowest_slot:
                        prev = lowest_slot[v]
                        time_used[prev] -= self.time_size[v, 0]
                        allocation_dict[prev] = [item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)]
                        assigned[v, 0] = False

                    time_used[idx] += lowest_size + size
                    allocation_dict[idx].append((v, 0))
                    allocation_dict[idx].append((v, ver))
                    assigned[v, 0] = True
                    assigned[v, ver] = True
                    slot_per_video[v] = idx



                '''
                for i in range(self.n_slots):
                    if self.deadline[v] == (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]:
                        if v in lowest_slot:
                            prev = lowest_slot[v]
                            time_used[prev] -= self.time_size[v, 0]
                            allocation_dict[prev] = [item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)]
                            assigned[v, 0] = False
                        time_used[i] += lowest_size + size
                        allocation_dict[i].append((v, 0))
                        allocation_dict[i].append((v, ver))
                        assigned[v, 0] = True
                        assigned[v, ver] = True
                        slot_per_video[v] = i
                        break
                '''
            else:
                i = slot_per_video[v]
                if time_used[i] + size <= self.time_limit[i]:
                    time_used[i] += size
                    allocation_dict[i].append((v, ver))
                    assigned[v, ver] = True

        total_reward = 0.0
        for v in range(self.n_videos):
            allocated_versions = [ver for i in range(self.n_slots)
                                  for (vid, ver) in allocation_dict[i] if vid == v]
            fallback_versions = []
            for ver in range(self.n_ver):
                if ver in allocated_versions:
                    fallback_versions.append(ver)
                else:
                    lower = [v2 for v2 in allocated_versions if v2 < ver]
                    fallback_versions.append(max(lower) if lower else 0)

            #version_popularities = self.popularity[v]
            version_popularities = self.true_popularity[v]
            version_vmaf = self.vmaf[v, fallback_versions]
            reward = np.sum(version_popularities * version_vmaf) * 100
            total_reward += reward

        scaled_reward = (total_reward / (self.expected_total_reward + 1e-6)) * 10.0
        
        #print(f"✨ 그리디(HUF-StrictDL) 총 QoE (스케일된): {scaled_reward:.2f}")

        return scaled_reward, allocation_dict
    
    def greedy_allocation_combo_delta(self, mode="mckp", *, verbose=True):
        """
        통합 Δ-그리디 (조합 단위, 항상 ver0 포함)
        - mode="mckp": 정렬/선택(랭킹)에 pred_popularity 사용 (이전 greedy_allocation_MCKP)
        - mode="oracle": 정렬/선택(랭킹)에 true_popularity 사용 (이전 greedy_allocation_Oracle)
        공통 정책:
        * ver0 선할당 (STRICT deadline 그룹 내에서 load balancing)
        * 비디오별 조합 생성 → ΔQoE/ΔSize → (ΔQoE>0, ΔSize>0)만 남기고 파레토 전선
        * 전역 score = ΔQoE_rank / (ΔSize+eps) 내림차순
        * in-place 업그레이드 우선, 불가 시 strict-deadline 내 worst-fit 재앵커
        * 최종 평가는 항상 true_popularity × VMAF (fallback)로 계산
        """
        import math
        import numpy as np

        assert mode in ("mckp", "oracle")
        eps = 1e-9

        # -------------------------- 헬퍼 --------------------------
        def _qoe_for_combo(v, versions_set, use="true"):
            """
            비디오 v가 versions_set(항상 0 포함)만 저장됐을 때의 QoE.
            use: "true" -> true_popularity, "pred" -> pred_popularity
            """
            dist = self.true_popularity if use == "true" else self.pred_popularity
            used = sorted(versions_set)
            # fallback: 요청 ver에 대해 저장된 <= ver 중 최대, 없으면 0
            fb = []
            for ver in range(self.n_ver):
                if ver in used:
                    fb.append(ver)
                else:
                    lowers = [x for x in used if x < ver]
                    fb.append(max(lowers) if lowers else 0)
            fb = np.array(fb, dtype=int)
            return float(np.sum(dist[v] * self.vmaf[v, fb]))

        def _size_for_combo(v, versions_set):
            """비디오 v가 versions_set 저장 시 총 용량(시간)"""
            idx = list(versions_set)
            return float(np.sum(self.time_size[v, idx]))

        def _pareto_prune(cands):
            """
            cands: list of (delta_qoe, delta_size, mask_set)
            퇴행 제거(ΔQoE<=0 & ΔSize>0) + 파레토 전선만 남김.
            """
            filt = [(dq, ds, ms) for (dq, ds, ms) in cands if (dq > 0 and ds > 0)]
            if not filt:
                return []
            # ΔSize 오름차순, 같은 크기면 ΔQoE 내림차순
            filt.sort(key=lambda x: (x[1], -x[0]))
            # 파레토 전선: ΔSize가 작으면서 ΔQoE가 큰 것만 채택
            frontier, best_qoe = [], -math.inf
            for dq, ds, ms in filt:
                if dq > best_qoe:
                    frontier.append((dq, ds, ms))
                    best_qoe = dq
            return frontier

        def _strict_slots_for(v):
            """비디오 v가 엄격 데드라인으로 사용할 수 있는 슬롯 인덱스들."""
            d = int(self.deadline[v])
            return [i for i in range(self.n_slots) if (i % self.n_deadline) == d]

        # 랭킹(정렬/선택)에 사용할 분포 선택
        use_rank = "pred" if mode == "mckp" else "true"

        # ------------------ (0) 초기화: ver0 선할당 ------------------
        time_used = np.zeros(self.n_slots, dtype=np.float32)
        allocation_dict = {i: [] for i in range(self.n_slots)}
        current_combo = {v: {0} for v in range(self.n_videos)}       # 현재 버전 집합(초기 {0})
        current_slot  = {}                                            # v -> slot

        # baseline QoE (평가/정렬 각각)
        baseline_qoe_eval = np.zeros(self.n_videos, dtype=np.float64)   # true 기준
        baseline_qoe_rank = np.zeros(self.n_videos, dtype=np.float64)   # use_rank 기준
        # 현재까지 채택된 ΔQoE(랭킹 기준)
        current_delta_rank = np.zeros(self.n_videos, dtype=np.float64)

        # ver0를 strict deadline 안에서 load balancing(상대 사용률 낮은 곳)
        for v in range(self.n_videos):
            size0 = float(self.time_size[v, 0])
            slots = _strict_slots_for(v)
            feasible = [i for i in slots if time_used[i] + size0 <= self.time_limit[i]]
            if feasible:
                def usage_ratio(i): return float(time_used[i] / (self.time_limit[i] + eps))
                i_sel = min(feasible, key=usage_ratio)
                time_used[i_sel] += size0
                allocation_dict[i_sel].append((v, 0))
                current_slot[v] = i_sel
            else:
                current_slot[v] = None  # ver0조차 불가한 극단 상황

            baseline_qoe_eval[v] = _qoe_for_combo(v, {0}, use="true")
            baseline_qoe_rank[v] = _qoe_for_combo(v, {0}, use=use_rank)

        # ------------------ (1) 비디오별 후보 생성/프루닝 ------------------
        per_video_frontier = {}   # v -> list of (ΔQoE_rank, ΔSize, mask_set)
        for v in range(self.n_videos):
            cand = []
            # {ver1..ver6}의 모든 조합(0~2^(n_ver-1)-1). mask==0은 baseline이므로 제외.
            for mask in range(1, 1 << (self.n_ver - 1)):
                vs = {0}
                for b in range(self.n_ver - 1):  # 버전1..6
                    if (mask >> b) & 1:
                        vs.add(b + 1)

                size_v = _size_for_combo(v, vs)
                size_0 = float(self.time_size[v, 0])
                delta_size = size_v - size_0
                if delta_size <= 0:
                    continue

                # 랭킹 기준 ΔQoE
                qoe_rank = _qoe_for_combo(v, vs, use=use_rank)
                delta_qoe_rank = qoe_rank - baseline_qoe_rank[v]
                cand.append((delta_qoe_rank, delta_size, frozenset(vs)))

            per_video_frontier[v] = _pareto_prune(cand)

        # ------------------ (2) 전역 후보 정렬 (score=ΔQoE/ΔSize) ------------------
        global_cands = []
        for v, items in per_video_frontier.items():
            for dq, ds, s in items:
                score = dq / (ds + eps)
                global_cands.append((score, v, dq, ds, s))
        global_cands.sort(reverse=True, key=lambda x: x[0])

        # 현재 조합 사이즈 캐시
        cur_size = {v: _size_for_combo(v, current_combo[v]) for v in range(self.n_videos)}

        # ------------------ (3) 높은 score부터 적용 (앵커/재앵커) ------------------
        for score, v, dq_rank, ds, new_set in global_cands:
            # 랭킹 기준 QoE 개선 없으면 skip
            if dq_rank <= current_delta_rank[v] + 1e-12:
                continue

            new_total_size = _size_for_combo(v, new_set)
            old_total_size = cur_size[v]
            extra_needed = new_total_size - old_total_size

            anchor = current_slot.get(v, None)
            placed = False

            def _apply_to_slot(slot_idx):
                """v의 기존 배치를 slot_idx로 교체."""
                nonlocal placed, time_used, allocation_dict
                # 기존 앵커에서 제거
                if current_slot[v] is not None:
                    old_slot = current_slot[v]
                    allocation_dict[old_slot] = [(vid, ver) for (vid, ver) in allocation_dict[old_slot] if vid != v]
                    time_used[old_slot] -= old_total_size
                    time_used[old_slot] = max(0.0, time_used[old_slot])

                # 새 슬롯에 새 조합 추가
                for ver in sorted(new_set):
                    allocation_dict[slot_idx].append((v, ver))
                time_used[slot_idx] += new_total_size

                # 상태 갱신
                current_slot[v] = slot_idx
                current_combo[v] = set(new_set)
                cur_size[v] = new_total_size
                current_delta_rank[v] = dq_rank
                placed = True

            # 3-1) in-place 업그레이드
            if anchor is not None:
                if extra_needed <= 0:
                    _apply_to_slot(anchor)
                else:
                    if time_used[anchor] + extra_needed <= self.time_limit[anchor]:
                        _apply_to_slot(anchor)

            # 3-2) 재앵커: strict 데드라인 내 worst-fit
            if not placed:
                strict_slots = _strict_slots_for(v)
                feasible = [i for i in strict_slots if time_used[i] + new_total_size <= self.time_limit[i]]
                if feasible:
                    def leftover_after(i): return float(self.time_limit[i] - (time_used[i] + new_total_size))
                    i_sel = max(feasible, key=leftover_after)  # worst-fit
                    _apply_to_slot(i_sel)

            # 3-3) 여전히 불가면 skip

        # ------------------ (4) 최종 QoE 계산(평가=TRUE) ------------------
        total_reward = 0.0
        for v in range(self.n_videos):
            # v에 대해 실제 저장된 버전 집합 수집
            allocated_versions = set()
            for i in range(self.n_slots):
                for (vid, ver) in allocation_dict[i]:
                    if vid == v:
                        allocated_versions.add(ver)

            # fallback 맵
            fallback_versions = []
            for ver in range(self.n_ver):
                if ver in allocated_versions:
                    fallback_versions.append(ver)
                else:
                    lower = [vv for vv in allocated_versions if vv < ver]
                    fallback_versions.append(max(lower) if lower else 0)

            version_popularities = self.true_popularity[v]  # 평가(oracle)용
            version_vmaf = self.vmaf[v, fallback_versions]
            reward = np.sum(version_popularities * version_vmaf) * 100.0
            total_reward += float(reward)

        scaled_reward = (total_reward / (self.expected_total_reward + eps)) * 10.0
        if verbose:
            tag = "MCKP" if mode == "mckp" else "ORACLE"
            #print(f"✨ Combo-ΔGreedy({tag}) 총 QoE (스케일): {scaled_reward:.2f}")
        """
        # ------------------ (5) 재계산 기반 검증 출력 ------------------
        re_time_used = np.zeros(self.n_slots, dtype=np.float32)
        for i in range(self.n_slots):
            if allocation_dict[i]:
                re_time_used[i] = sum(float(self.time_size[vid, ver]) for (vid, ver) in allocation_dict[i])

        overflow = np.where(re_time_used > self.time_limit + 1e-6)[0]
        if verbose:
            print("OVERFLOW slots:", overflow.tolist(), flush=True)
            print("max util:", float(np.max(re_time_used / (self.time_limit + 1e-9))), flush=True)

        # strict deadline 위반 스캔: i%dead != deadline[vid] 이면 위반
        violations = []
        for i in range(self.n_slots):
            for (vid, ver) in allocation_dict[i]:
                slot_group = i % self.n_deadline
                if slot_group != int(self.deadline[vid]):
                    violations.append((i, int(vid), int(ver), int(self.deadline[vid]), int(slot_group)))

        if verbose:
            if violations:
                print(f"DEADLINE VIOLATIONS: {len(violations)} found", flush=True)
                print("  examples (slot, vid, ver, deadline[vid], slot_group):", violations[:10], flush=True)
            else:
                print("DEADLINE VIOLATIONS: none", flush=True)
        """
        return scaled_reward, allocation_dict



    

    def greedy_allocation_HUTF_strictDL(self):
        time_used = np.zeros(self.n_slots, dtype=np.float32)
        allocation_dict = {i: [] for i in range(self.n_slots)}
        assigned = np.zeros((self.n_videos, self.n_ver), dtype=bool)

        # (0-1) lowest version 선할당 (ver=0), load balancing + strict deadline
        lowest_slot = {}
        for v in range(self.n_videos):
            size = self.time_size[v, 0]
            candidate_slots = [
                i for i in range(self.n_slots)
                if self.deadline[v] == (i % self.n_deadline) and time_used[i] + size <= self.time_limit[i]
            ]
            if candidate_slots:
                idx = min(candidate_slots, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))
                time_used[idx] += size
                allocation_dict[idx].append((v, 0))
                assigned[v, 0] = True
                lowest_slot[v] = idx
        
        # (1) score 계산 및 정렬
        score_list = []
        for v in range(self.n_videos):
            for ver in range(1, self.n_ver):
                #score = (self.popularity[v, ver] * self.vmaf[v, ver]) / (self.time_size[v, ver] + 1e-6)
                score = (self.pred_popularity[v, ver] * self.vmaf[v, ver]) / (self.time_size[v, ver] + 1e-6)
                score_list.append((score, v, ver))
        score_list.sort(reverse=True)

        # (2) 비디오별 최초 등장 ver과 이후 처리
        slot_per_video = {}

        for _, v, ver in score_list:
            if assigned[v, ver]:
                continue
            size = self.time_size[v, ver]

            if v not in slot_per_video:
                lowest_size = self.time_size[v, 0]


                #'''
                # here
                candidates = [
                    i for i in range(self.n_slots)
                    if self.deadline[v] == (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]
                ]

                if candidates:
                    # 후보 중 현재 채움 비율이 가장 낮은 슬롯 선택
                    idx = min(candidates, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))

                    if v in lowest_slot:
                        prev = lowest_slot[v]
                        time_used[prev] -= self.time_size[v, 0]
                        allocation_dict[prev] = [item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)]
                        assigned[v, 0] = False

                    time_used[idx] += lowest_size + size
                    allocation_dict[idx].append((v, 0))
                    allocation_dict[idx].append((v, ver))
                    assigned[v, 0] = True
                    assigned[v, ver] = True
                    slot_per_video[v] = idx
                #'''


                '''
                for i in range(self.n_slots):
                    if self.deadline[v] == (i % self.n_deadline) and \
                    time_used[i] + lowest_size + size <= self.time_limit[i]:
                        if v in lowest_slot:
                            prev = lowest_slot[v]
                            time_used[prev] -= self.time_size[v, 0]
                            allocation_dict[prev] = [item for item in allocation_dict[prev] if not (item[0] == v and item[1] == 0)]
                            assigned[v, 0] = False
                        time_used[i] += lowest_size + size
                        allocation_dict[i].append((v, 0))
                        allocation_dict[i].append((v, ver))
                        assigned[v, 0] = True
                        assigned[v, ver] = True
                        slot_per_video[v] = i
                        break
                '''
            else:
                i = slot_per_video[v]
                if time_used[i] + size <= self.time_limit[i]:
                    time_used[i] += size
                    allocation_dict[i].append((v, ver))
                    assigned[v, ver] = True
        #print(np.sum(time_used))
        #exit()

        total_reward = 0.0
        for v in range(self.n_videos):
            allocated_versions = [ver for i in range(self.n_slots)
                                  for (vid, ver) in allocation_dict[i] if vid == v]
            fallback_versions = []
            for ver in range(self.n_ver):
                if ver in allocated_versions:
                    fallback_versions.append(ver)
                else:
                    lower = [v2 for v2 in allocated_versions if v2 < ver]
                    fallback_versions.append(max(lower) if lower else 0)

            #version_popularities = self.popularity[v]
            version_popularities = self.true_popularity[v]
            version_vmaf = self.vmaf[v, fallback_versions]
            reward = np.sum(version_popularities * version_vmaf) * 100
            total_reward += reward

        scaled_reward = (total_reward / (self.expected_total_reward + 1e-6)) * 10.0
        #print(f"✨ 그리디(HUTF-StrictDL) 총 QoE (스케일된): {scaled_reward:.2f}")
        # self._log_util_stats(time_used, "Greedy HUTF-StrictDL")
        return scaled_reward, allocation_dict

    def EDF_allocation(self):
        # 초기화
        time_used = np.zeros(self.n_slots, dtype=np.float32)
        allocation_dict = {i: [] for i in range(self.n_slots)}
        assigned = np.zeros((self.n_videos, self.n_ver), dtype=bool)

        # (0-1) ver=0 선할당: 반드시 "데드라인 이하 슬롯" 중에서
        for v in range(self.n_videos):
            size = self.time_size[v, 0]
            feasible_slots = [
                i for i in range(self.n_slots)
                if (i % self.n_deadline) <= self.deadline[v] and
                time_used[i] + size <= self.time_limit[i]
            ]
            if feasible_slots:
                # 가장 여유 있는 슬롯 선택
                idx = min(feasible_slots, key=lambda i: time_used[i] / (self.time_limit[i] + 1e-6))
                time_used[idx] += size
                allocation_dict[idx].append((v, 0))
                assigned[v, 0] = True
            # else: ver=0조차 데드라인 내에 못 넣으면 이 비디오는 미스 가능

        # (1) EDF 순서
        video_order = sorted(range(self.n_videos), key=lambda v: self.deadline[v])

        # (2) 각 비디오에 대해 추가 버전 할당 (데드라인 이하 슬롯만)
        for v in video_order:
            for ver in range(1, self.n_ver):
                if assigned[v, ver]:
                    continue
                size = self.time_size[v, ver]
                for i in range(self.n_slots):
                    slot_deadline = i % self.n_deadline
                    if slot_deadline <= self.deadline[v] and time_used[i] + size <= self.time_limit[i]:
                        time_used[i] += size
                        allocation_dict[i].append((v, ver))
                        assigned[v, ver] = True
                        break

        # (3) QoE 계산: "접근 가능한 슬롯(슬롯그룹 ≤ 비디오 데드라인)에 저장된 버전만" 사용 가능
        total_reward = 0.0
        for v in range(self.n_videos):
            # 데드라인 내 슬롯에 실제 저장된 버전 집합
            accessible_versions = set()
            for i in range(self.n_slots):
                if (i % self.n_deadline) <= self.deadline[v]:
                    for (vid, ver) in allocation_dict[i]:
                        if vid == v:
                            accessible_versions.add(ver)

            for ver in range(self.n_ver):
                if ver in accessible_versions:
                    used_ver = ver
                else:
                    lower = [vv for vv in accessible_versions if vv < ver]
                    used_ver = max(lower) if lower else None

                if used_ver is None:
                    # 데드라인 내에 어떤 버전도 없으면 미스(보수적으로 0 QoE)
                    qoe = 0.0
                else:
                    qoe = self.vmaf[v, used_ver]

                total_reward += self.true_popularity[v, ver] * qoe

        total_reward *= 100.0
        normalized_reward = (total_reward / (self.expected_total_reward + 1e-6)) * 10.0
        print(f"EDF 총 QoE (스케일된, strict deadline): {normalized_reward:.2f}")
        # self._log_util_stats(time_used, "EDF-StrictDL")
        return normalized_reward, allocation_dict

    def compute_final_qoe(self, allocation_dict):
        """
        현재 에피소드의 최종 배치만으로 QoE 재계산.
        - full_qoe: 모든 요청이 원하는 버전 그대로 제공됐다고 가정(= expected_total_reward)
        - actual_qoe: 저장된 버전 + fallback 규칙으로 제공될 때의 QoE
        반환은 (actual_qoe, full_qoe), 그리고 로그로 비율/스케일 출력.
        """
        # 1) 비디오별 저장된 버전 집합 통합(중복/이동 정리)
        saved_by_video = [set() for _ in range(self.n_videos)]
        for slot in range(self.n_slots):
            for (vid, vers) in allocation_dict.get(slot, []):
                # vers 가 [0, 2, 5] 같은 리스트로 들어오므로 set 업데이트
                if isinstance(vers, (list, tuple, np.ndarray)):
                    saved_by_video[vid].update(int(v) for v in vers)
                else:
                    saved_by_video[vid].add(int(vers))

        # 2) fallback 규칙
        def served_version(saved_set, req_ver):
            if req_ver in saved_set:
                return req_ver
            lowers = [v for v in saved_set if v < req_ver]
            return max(lowers) if lowers else 0

        # 3) 실제 QoE 합산(actual) / 이상적 QoE(full)
        actual_qoe = 0.0
        for v in range(self.n_videos):
            saved = saved_by_video[v]
            for ver in range(self.n_ver):
                use_ver = served_version(saved, ver)
                actual_qoe += float(self.true_popularity[v, ver] * self.vmaf[v, use_ver])

        full_qoe = float(self.expected_total_reward)  # 이미 동일 정의

        # 4) 지표 출력
        ratio = (actual_qoe / (full_qoe + 1e-6)) * 100.0
        scaled = (actual_qoe * 1000.0) / (full_qoe + 1e-6)   # 학습시 스케일과 동일(≈ 984 등)
        #print(f"🔮 제공된 총 QoE / 기대 QoE: {ratio:.2f}% (스케일={scaled:.2f})")

        return actual_qoe, full_qoe


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=1, num_envs=4):  # 🔹 num_envs 추가
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = np.zeros(num_envs)  # 🔹 각 환경별 보상 저장

    def _on_step(self):
        # ✅ 현재 step에서 발생한 reward를 환경별로 누적
        if "rewards" in self.locals:
            self.current_episode_reward += self.locals["rewards"]  # 🔹 개별 환경별 리워드 저장

        # ✅ 에피소드가 종료될 때 (각 환경별 개별 저장)
        if "dones" in self.locals:
            for i, done in enumerate(self.locals["dones"]):  # 🔹 각 환경별 종료 확인
                if done:
                    self.episode_rewards.append(self.current_episode_reward[i])  # 개별 환경의 보상 저장
                    self.current_episode_reward[i] = 0  # 🔹 새로운 에피소드 시작 시 해당 환경 초기화

        return True  # 계속 학습 진행

if __name__ == "__main__":

    def make_env(video_model="logit_blend", version_model="dirichlet",
             video_param=0.03, video_tau=0.10, video_lambda=0.40,
             ver_param=0.25, seed=None):
        def _init():
            env = TransEnv(
                video_noise_model=video_model,
                video_noise_param=video_param,
                video_tau=video_tau,
                video_lambda=video_lambda,
                version_noise_model=version_model,
                version_noise_param=ver_param,
                seed=seed
            )
            return ActionMasker(env, lambda e: e.get_valid_action_mask())
        return _init


    # ----------------- 하이퍼파라미터 스케줄 -----------------
    lr0 = 3e-4
    def lr_schedule(p):         # p: 1 -> 0
        return lr0 * (0.3 + 0.7 * p)

    def ent_schedule(p):        # 초반 0.022 -> 후반 0.002
        return 0.02 * p + 0.002

    # ----------------- 환경 생성 -----------------
    '''
    env = SubprocVecEnv([
        make_env(VIDEO_TRAINING_MODEL, VER_TRAINING_MODEL,
                video_param=VIDEO_PARAM, video_tau=VIDEO_TAU, video_lambda=VIDEO_LAMBDA,
                ver_param=VER_PARAM, seed=100+i)
        for i in range(num_envs)
    ])
    '''

    env = DummyVecEnv([
        make_env(VIDEO_TRAINING_MODEL, VER_TRAINING_MODEL,
                video_param=VIDEO_PARAM, video_tau=VIDEO_TAU, video_lambda=VIDEO_LAMBDA,
                ver_param=VER_PARAM, seed=100+i)
        for i in range(num_envs)
    ])


    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=5.0)  # ✅ 관측 정규화

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.SiLU,   # or Tanh
        ortho_init=False               # 큰 네트워크면 False가 실전에서 종종 안정적
    )

    # ----------------- 모델 구성 -----------------
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        n_steps=2048,          # ✅ 4096 -> 1024
        batch_size=2048,
        n_epochs=12,           # ✅ 20 -> 12
        learning_rate=lr_schedule,
        ent_coef=0.018, # ✅ 스케줄
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02         # ✅ 0.01 -> 0.02 (업데이트 폭 조금 허용)
    )

    reward_callback = RewardLoggerCallback()

    if TRAINING:
        if os.path.exists(MODEL_PATH):
            print(f"✅ 기존 모델 {MODEL_PATH} 불러오기...")
            if FINETUNE:
                model = MaskablePPO.load(
                MODEL_PATH, env=env, device=device,
                custom_objects={
                    "ent_coef": 0.0,          # 탐색 끄기
                    "learning_rate": 1e-5,    # 작은 학습률
                    "clip_range": 0.1         # 과도한 업데이트 방지
                }
            )
                print("🔧 결정론 테스트 대비 미세 학습 시작...")
                model.learn(total_timesteps=FINETUNE_STEPS, log_interval=10, callback=reward_callback)
            else:
                model = MaskablePPO.load(MODEL_PATH, env=env, device=device)
                model.learn(total_timesteps=1000000, log_interval=10, callback=reward_callback)
    
        else:
            model.learn(total_timesteps=TOTAL_STEPS, log_interval=10, callback=reward_callback)
   
        # ✅ 저장
        model.save(MODEL_PATH)
        print(f"✅ 모델 저장 완료: {MODEL_PATH}")
        env.save("vecnorm.pkl")   # ✅ VecNormalize 통계 저장

        # ✅ 리워드 그래프 출력
        plt.figure(figsize=(10, 5))
        plt.plot(reward_callback.episode_rewards, label="Total Reward per Episode")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Learning Curve: PPO Training Progress")
        plt.legend()
        plt.grid()
        plt.show()



    def export_plan(env, policy_name="DRL", alloc=None, out_dir="plans"):
        """
        env: test_env.envs[0].unwrapped (TransEnv)
        policy_name: "DRL" / "HUF" / "HUTF" / ...
        alloc: 알고리즘에서 직접 리턴한 allocation_dict (없으면 env.allocation_dict 사용)
        """
        import os, json, datetime

        if alloc is None:
            alloc = env.allocation_dict   # DRL 경우 기본값

        n_slots = env.n_slots
        n_deadlines = env.n_deadline
        n_servers = n_slots // n_deadlines
        time_limits = env.time_limit.tolist()

        ladder = [
            {"ver": 0, "width": 256,  "height": 144,  "label": "144p"},
            {"ver": 1, "width": 320,  "height": 240,  "label": "240p"},
            {"ver": 2, "width": 384,  "height": 288,  "label": "288p"},
            {"ver": 3, "width": 480,  "height": 360,  "label": "360p"},
            {"ver": 4, "width": 640,  "height": 480,  "label": "480p"},
            {"ver": 5, "width": 1280, "height": 720,  "label": "720p"},
            {"ver": 6, "width": 1920, "height": 1080, "label": "1080p"},
        ]

        slots = {}
        for s in range(n_slots):
            jobs_by_vid = {}
            for (vid0, ver_or_list) in alloc.get(s, []):
                vid = int(vid0) + 1
                # DRL: ver_or_list = [0,1,2] / HUF: ver_or_list = 2
                vers = ver_or_list if isinstance(ver_or_list, (list, tuple)) else [ver_or_list]
                if vid not in jobs_by_vid:
                    jobs_by_vid[vid] = {"video_id": vid, "video": f"{vid}.mp4", "versions": []}
                jobs_by_vid[vid]["versions"].extend(int(v) for v in vers)
            # 중복 버전 제거 + 정렬
            for job in jobs_by_vid.values():
                job["versions"] = sorted(set(job["versions"]))
            slots[str(s)] = list(jobs_by_vid.values())

        plan = {
            "policy": policy_name,
            "plan_id": f"{policy_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "n_slots": n_slots,
            "n_deadlines": n_deadlines,
            "n_servers": n_servers,
            "ladder": ladder,
            "time_limits": {str(i): float(t) for i, t in enumerate(time_limits)},
            "slots": slots
        }

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{policy_name}.json")
        with open(out_path, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"[INFO] plan exported -> {out_path}")






    if TESTING:

        num_test_episodes = 50
        results = []  # 결과 저장 리스트

        # SIZE_FACTOR를 0.01 ~ 0.99까지 0.01씩 증가
        for SIZE_FACTOR in tqdm(np.arange(0.3, 0.85, 0.01), desc="SIZE FACTOR Progress"):
            print(f"\n===== SIZE_FACTOR = {SIZE_FACTOR:.2f} =====")

            # === 환경 생성 ===
            test_env = DummyVecEnv([
                make_env(VIDEO_TRAINING_MODEL, VER_TRAINING_MODEL,
                        video_param=VIDEO_PARAM,
                        video_tau=VIDEO_TAU,
                        video_lambda=VIDEO_LAMBDA,
                        ver_param=VER_PARAM)
            ])
            test_env = VecNormalize.load("vecnorm.pkl", test_env)
            test_env.training = False
            test_env.norm_reward = False

            # === MCKP_predict baseline 계산 ===
            MCKP_predict_QoE = 0
            for episode in tqdm(range(num_test_episodes), desc="MCKP_Episodes", leave=False):
                obs = test_env.reset()
                done = False
                while not done:
                    # action 없이 환경이 결정 (greedy baseline)
                    obs, reward, done, info = test_env.step([0])   # dummy action
                MCKP_predict_QoE += test_env.env_method("get_MCKP_predict_value")[0]

            MCKP_predict_val = (MCKP_predict_QoE / num_test_episodes) / 10

            # === DRL & Baseline_DRL ===
            drl_results = {}

            for model_name, model_path in MODEL_PATHS.items():
                PPO_QoE = 0

                model = MaskablePPO.load(model_path, env=test_env, device=device)

                for episode in tqdm(range(num_test_episodes), desc=f"{model_name} Episodes", leave=False):
                    obs = test_env.reset()
                    done = False
                    while not done:
                        mask = test_env.env_method('get_valid_action_mask')[0]
                        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                        obs, reward, done, info = test_env.step(action)

                    PPO_QoE += test_env.env_method("get_PPO_value")[0]

                drl_results[model_name] = (PPO_QoE / num_test_episodes) / 10

            # === 결과 저장 ===
            results.append([
                SIZE_FACTOR,
                drl_results["DRL"],
                drl_results["Baseline_DRL"],
                MCKP_predict_val
            ])

        # === CSV 파일 저장 ===
        with open("Provisioning_Ratio.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "SIZE_FACTOR", "PPO_DRL", "PPO_BASELINE", "MCKP_predict"
            ])
            writer.writerows(results)

        print("All saved to Provisioning_Ratio.csv")

        

        THRESHOLDS = [90, 95]
        algorithms = ["PPO_DRL", "PPO_BASELINE", "MCKP_predict"]

        df = pd.read_csv("Provisioning_Ratio.csv")

        pr_results = {alg: {} for alg in algorithms}

        for alg in algorithms:
            for th in THRESHOLDS:
                row = df[df[alg] >= th]
                if not row.empty:
                    pr_results[alg][f"PR_{int(th)}"] = float(row.iloc[0]["SIZE_FACTOR"])
                else:
                    pr_results[alg][f"PR_{int(th)}"] = None

        
        import json
        with open("PR_results.json", "w") as f:
            json.dump(pr_results, f, indent=4)

        print("🎯 PR extraction complete. Saved to PR_results.json")




        '''
        num_test_episodes = 50
        results = []  # 결과 저장 리스트

        # SIZE_FACTOR를 0.01 ~ 0.99까지 0.01씩 증가
        for SIZE_FACTOR in tqdm(np.arange(0.01, 1.0, 0.01), desc="SIZE FACTOR Progress"):
            print(f"\n===== SIZE_FACTOR = {SIZE_FACTOR:.2f} =====")

            # === 환경 생성 ===
            test_env = DummyVecEnv([
                make_env(VIDEO_TRAINING_MODEL, VER_TRAINING_MODEL,
                        video_param=VIDEO_PARAM,
                        video_tau=VIDEO_TAU,
                        video_lambda=VIDEO_LAMBDA,
                        ver_param=VER_PARAM)
            ])
            test_env = VecNormalize.load("vecnorm.pkl", test_env)
            test_env.training = False
            test_env.norm_reward = False


            # 결과 변수 초기화
            PPO_QoE = HUF_QoE = HUTF_QoE = 0
            HUF_strict_QoE = HUTF_strict_QoE = 0
            MCKP_predict_QoE = MCKP_true_QoE = 0

            model = MaskablePPO.load(MODEL_PATH, env=test_env, device=device)

            # === 에피소드 반복 ===
            for episode in tqdm(range(num_test_episodes), desc="Episode Progress", leave=False):
                #print(f"\n🚀 테스트 에피소드 {episode+1} 시작")
                obs = test_env.reset()
                done = False
                while not done:
                    mask = test_env.env_method('get_valid_action_mask')[0]
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    obs, reward, done, info = test_env.step(action)

                # QoE 값 누적
                HUF_QoE          += test_env.env_method("get_HUF_value")[0]
                HUTF_QoE         += test_env.env_method("get_HUTF_value")[0]
                HUF_strict_QoE   += test_env.env_method("get_HUF_strict_value")[0]
                HUTF_strict_QoE  += test_env.env_method("get_HUTF_strict_value")[0]
                MCKP_predict_QoE += test_env.env_method("get_MCKP_predict_value")[0]
                MCKP_true_QoE    += test_env.env_method("get_MCKP_true_value")[0]
                PPO_QoE          += test_env.env_method("get_PPO_value")[0]

            # === 평균 QoE 계산 ===
            PPO_val          = (PPO_QoE / num_test_episodes) / 10
            HUF_val          = (HUF_QoE / num_test_episodes) / 10
            HUTF_val         = (HUTF_QoE / num_test_episodes) / 10
            HUF_strict_val   = (HUF_strict_QoE / num_test_episodes) / 10
            HUTF_strict_val  = (HUTF_strict_QoE / num_test_episodes) / 10
            MCKP_predict_val = (MCKP_predict_QoE / num_test_episodes) / 10
            MCKP_true_val    = (MCKP_true_QoE / num_test_episodes) / 10

            print("====decision results====")
            print("PPO value : ", PPO_val)
            print("HUF value : ", HUF_val)
            print("HUTF value : ", HUTF_val)
            print("HUF_strict value : ", HUF_strict_val)
            print("HUTF_strict value : ", HUTF_strict_val)
            print("MCKP_predict value : ", MCKP_predict_val)
            print("MCKP_true value : ", MCKP_true_val)

            # === 결과 저장 ===
            results.append([
                SIZE_FACTOR, PPO_val, HUF_val, HUTF_val,
                HUF_strict_val, HUTF_strict_val,
                MCKP_predict_val, MCKP_true_val
            ])

        # === CSV 파일로 저장 ===
        with open("decision_results_hetero_baseline.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "SIZE_FACTOR", "PPO", "HUF", "HUTF",
                "HUF_strict", "HUTF_strict",
                "MCKP_predict", "MCKP_true"
            ])
            writer.writerows(results)

        print("✅ 모든 결과 decision_results.csv 파일에 저장 완료!")

        '''





        '''


        num_test_episodes = 50  # 테스트할 에피소드 수
        #num_test_episodes = 1  # 테스트할 에피소드 수

        # DummyVecEnv
        test_env = DummyVecEnv([
            make_env(VIDEO_TRAINING_MODEL, VER_TRAINING_MODEL,
                    video_param=VIDEO_PARAM,
                    video_tau=VIDEO_TAU,
                    video_lambda=VIDEO_LAMBDA,
                    ver_param=VER_PARAM)
        ])

        test_env = VecNormalize.load("vecnorm.pkl", test_env)
        test_env.training = False
        test_env.norm_reward = False

        PPO_QoE = 0
        HUF_QoE = 0
        HUTF_QoE = 0
        HUF_strict_QoE = 0
        HUTF_strict_QoE = 0
        MCKP_predict_QoE = 0
        MCKP_true_QoE = 0

        model = MaskablePPO.load(MODEL_PATH, env=test_env, device=device)

        start = time.time()     # 시작 시각
        for episode in range(num_test_episodes):
            print(f"\n🚀 테스트 에피소드 {episode+1} 시작")
            obs = test_env.reset()
            done = False
            while not done:
                mask = test_env.env_method('get_valid_action_mask')[0]  # 단일 env
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                #action, _ = model.predict(obs, action_masks=mask)
                obs, reward, done, info = test_env.step(action)
                

            
            HUF_QoE          += test_env.env_method("get_HUF_value")[0]
            HUTF_QoE         += test_env.env_method("get_HUTF_value")[0]
            HUF_strict_QoE   += test_env.env_method("get_HUF_strict_value")[0]
            HUTF_strict_QoE  += test_env.env_method("get_HUTF_strict_value")[0]
            MCKP_predict_QoE += test_env.env_method("get_MCKP_predict_value")[0]
            MCKP_true_QoE    += test_env.env_method("get_MCKP_true_value")[0]
            PPO_QoE          += test_env.env_method("get_PPO_value")[0]
            

        end = time.time()       # 종료 시각

        total_time = end - start
        avg_time = total_time / num_test_episodes
        print(f"에피소드당 평균 실행 시간: {avg_time:.2f}초")
        



        
        print("====decision results====")
        print("PPO value : ", (PPO_QoE / num_test_episodes) / 10)
        print("HUF value : ", (HUF_QoE / num_test_episodes) / 10)
        print("HUTF value : ", (HUTF_QoE / num_test_episodes) / 10)
        print("HUF_strict value : ", (HUF_strict_QoE / num_test_episodes) / 10)
        print("HUTF_strict value : ", (HUTF_strict_QoE / num_test_episodes) / 10)
        print("MCKP_predict value : ", (MCKP_predict_QoE / num_test_episodes) / 10)
        print("MCKP_true value : ", (MCKP_true_QoE / num_test_episodes) / 10)

        '''

        '''
        env = test_env.envs[0].unwrapped
        #export_plan(env, "DRL", out_dir="plans")  # alloc 인자 생략
        export_plan(env, "baseline_DRL", out_dir="plans")  # alloc 인자 생략
        export_plan(env, "HUF", out_dir="plans", alloc=env.HUF_alloc)
        export_plan(env, "HUTF", out_dir="plans", alloc=env.HUTF_alloc)
        export_plan(env, "HUF_strictDL", out_dir="plans", alloc=env.HUF_strict_alloc)
        export_plan(env, "HUTF_strictDL", out_dir="plans", alloc=env.HUTF_strict_alloc)
        export_plan(env, "MCKP_predict", out_dir="plans", alloc=env.MCKP_predict_alloc)
        export_plan(env, "MCKP_true", out_dir="plans", alloc=env.MCKP_true_alloc)
        '''

        
