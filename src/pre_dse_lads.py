"""
pre_dse_lads.py — LADS + Gating replacement for pre_dse.py
============================================================

Interface mirrors pre_dse.py (Processor class with the same public methods),
but the event branch is rebuilt around the LADS time-surface and the
SmartRouterFrontEnd gating module instead of `accumulate_to_rgb`.

New dependencies (compared to pre_dse.py):
  - event_lads.LADS                       (NEW)
  - smartrouterFrontend.SmartRouterFrontEnd (NEW)
  - dataset_lads._events_to_raw_frame / _normalize_map (NEW, reused helpers)

Public methods kept identical (signature + name):
  - Processor.__init__(args)
  - Processor.load_event_image(image_dir, event_dir, image_timestamp_dir)
  - Processor._get_camera(calib_dir)
  - Processor.save_warpped_event_pair(...)
  - Processor._process_subfolder(subfolder)
  - Processor.run(subfolders=None, n_workers=1)
  - Processor.compute_rgb_stats(subfolders=None, save_to=None, workers=1)
  - Processor.process_tokens(subfolders=None, workers=1)
  - _compute_stats_for_subfolder(image_root, subfolder)

Output:
  - eventImage/{timestamp}.pt   (3, H_out, W_out) float32 hybrid LADS frame
                                channels = [pos_count_norm, neg_count_norm, lads_surface_norm]
  - eventImage/{timestamp}.png  visualization (for inspection only)
  - warpped/{timestamp}.png     RGB warped to event view (unchanged from pre_dse.py)

Why .pt for the event side?
  The hybrid LADS frame contains both polarity counts and a signed time-surface
  with continuous values; PNG quantization would lose that. The downstream
  trainer (train_dsec.py) reads the .pt directly.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# sys.path.append("dinov2")
# sys.path.append("segmentation")
import os
dinov2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dinov2")
sys.path.append(dinov2_path)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "segmentation"))

import glob
import os
import shutil

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import yaml
from PIL import Image

from utils import EventSlicer
from dinov2.models.vision_transformer import vit_small
from dataset import PadToMinSide, PairedProcessor, Normalize, RandomCrop, RandomHorizontalFlip, RandomSwapEventRedBlue, ResizeKeepRatio, ToTensor, CenterCrop
from utils import accumulate_to_rgb

# NEW: LADS + gating imports
from dataset_lads import _events_to_raw_frame, _normalize_map  # NEW
from event_lads.LADS import LADS  # NEW
from smartrouterFrontend import SmartRouterFrontEnd  # NEW


# ---------------------------------------------------------------------------
# NEW: dataset-wide stats over LADS-hybrid .pt frames
# ---------------------------------------------------------------------------
def _compute_stats_for_subfolder(image_root: str, subfolder: str):
    """Compute per-channel sums, squared sums, and pixel counts for one subfolder.

    NEW: reads `eventImage/*.pt` (hybrid LADS frames) instead of PNGs. Image
    stats are still computed from `warpped/*.png` because the teacher branch
    is still RGB.

    Returns tuple: (ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count)
    """
    ev_sum = np.zeros(3, dtype=np.float64)
    ev_sq_sum = np.zeros(3, dtype=np.float64)
    ev_count = 0

    img_sum = np.zeros(3, dtype=np.float64)
    img_sq_sum = np.zeros(3, dtype=np.float64)
    img_count = 0

    save_root = os.path.join(image_root, subfolder, "images", "left")
    warpped_dir = os.path.join(save_root, "warpped")
    eventImage_dir = os.path.join(save_root, "eventImage")
    if not (os.path.isdir(warpped_dir) and os.path.isdir(eventImage_dir)):
        return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count

    # NEW: pair by basename across .pt (event) and .png (image)
    warpped_files = {os.path.splitext(f)[0] for f in os.listdir(warpped_dir) if f.lower().endswith(".png")}
    event_files = {os.path.splitext(f)[0] for f in os.listdir(eventImage_dir) if f.lower().endswith(".pt")}
    common = sorted(warpped_files & event_files)
    if not common:
        return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count

    for stem in common:
        # NEW: event tensor is stored as .pt (3,H,W) float32
        ev_path = os.path.join(eventImage_dir, f"{stem}.pt")
        try:
            ev = torch.load(ev_path, map_location="cpu")
        except Exception:
            ev = None
        if ev is not None and ev.ndim == 3 and ev.shape[0] == 3:
            ev_np = ev.numpy().astype(np.float64)  # (3,H,W)
            c, h, w = ev_np.shape
            ev_flat = ev_np.reshape(c, -1)
            ev_sum += ev_flat.sum(axis=1)
            ev_sq_sum += (ev_flat * ev_flat).sum(axis=1)
            ev_count += h * w

        # image branch is unchanged
        img_path = os.path.join(warpped_dir, f"{stem}.png")
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is not None:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            h, w, _ = im.shape
            im_reshaped = im.reshape(-1, 3)
            img_sum += im_reshaped.sum(axis=0)
            img_sq_sum += (im_reshaped * im_reshaped).sum(axis=0)
            img_count += h * w

    return ev_sum, ev_sq_sum, ev_count, img_sum, img_sq_sum, img_count


class Processor():
    def __init__(self, args):
        super().__init__()

        self.root               = args.root
        self.split              = args.split
        self.device             = args.device

        self.image_root             = os.path.join(args.root, f"{args.split}_images")
        self.events_root            = os.path.join(args.root, f"{args.split}_events")
        self.calib_root             = os.path.join(args.root, f"{args.split}_calibration")
        self.sementatic_root        = os.path.join(args.root, f"{args.split}_semantic_segmentation/{self.split}")
        self.DSEC_ME = [0.8993729784963826, 0.7969581014619264, 0.8928228776286392]
        self.DSEC_SE = [0.2726268701553345, 0.2706460952758789, 0.2812058925628662]
        self.DSEC_MI = [0.23862534365611895, 0.24712072838418375, 0.2574492927024542]
        self.DSEC_SI = [0.2195923498258092, 0.2361895432347472, 0.2633142601113288]
        self.origi_H, self.origi_W  = None, None
        self.type = 'EI'
        self.DSEC_H, self.DSEC_W         = 224, 224
        self.scale_range = (0.5, 2)

        # NEW: LADS + gating configuration (read from args; safe defaults match dataset_lads.py)
        self.lads_kwargs = {
            "decay_func": getattr(args, "lads_decay_func", "er"),
            "decay_param": getattr(args, "lads_decay_param", 0.2),
            "patch_size": getattr(args, "lads_patch_size", 32),
            "interpolate_patches": getattr(args, "lads_interpolate_patches", True),
            "min_decay": getattr(args, "lads_min_decay", 0.0),
            "ts_to_seconds_factor": getattr(args, "lads_ts_to_seconds_factor", 1.0),
        }
        # NEW: target spatial resolution for saved event tensors (DSEC: 480x640 native; resized to 224x224 for ViT)
        self.out_H = int(getattr(args, "out_h", self.DSEC_H))  # DIMENSION ADAPT
        self.out_W = int(getattr(args, "out_w", self.DSEC_W))  # DIMENSION ADAPT
        # NEW: whether to apply SmartRouterFrontEnd offline. If a checkpoint is provided we use it;
        # otherwise we skip gating and save the raw hybrid frame (the trainer applies gating online).
        self.gate_ckpt = getattr(args, "gate_ckpt", None)
        # NEW: gating runs on CPU by default to avoid blocking GPU workers; override with --gate_device
        self.gate_device = getattr(args, "gate_device", "cpu")
        # NEW: whether to also write a PNG preview for visual inspection
        self.save_png_preview = bool(getattr(args, "save_png_preview", True))

        self._router = None  # lazily built per worker

    # ---------------------------------------------------------------------
    # NEW: lazy router builder so each ProcessPoolExecutor worker owns one
    # ---------------------------------------------------------------------
    def _build_router_if_needed(self, H: int, W: int):
        if self.gate_ckpt is None:
            return None
        if self._router is not None and self._router.H == H and self._router.W == W:
            return self._router
        router = SmartRouterFrontEnd(in_channels=3, hidden_channels=16, state_channels=3, H=H, W=W)
        ckpt = torch.load(self.gate_ckpt, map_location="cpu")
        state = ckpt.get("front_end", ckpt) if isinstance(ckpt, dict) else ckpt
        try:
            router.load_state_dict(state, strict=False)
            print(f"[gate] loaded SmartRouterFrontEnd weights from {self.gate_ckpt}")
        except Exception as e:
            print(f"[gate] WARN failed to load gate ckpt ({e}); using random init")
        router.to(self.gate_device).eval()
        for p in router.parameters():
            p.requires_grad = False
        self._router = router
        return router

    def load_event_image(self, image_dir, event_dir, image_timestamp_dir):
        """
            Args:
                event_root       str: Event data file, should be a h5 file.
                image_dir       str: Image data folder.
            Returns:
                event_dict      dict: {"t": np.ndarray, "p": np.ndarray, "x": np.ndarray, "y": np.ndarray}
                unique_index    list: The index of unique moments in the event data t.
                image_names:    list(dict): [{"image_name": image_name str, "image_timestamp": image_timestamp np.ndarray}]
        """
        image_names     = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        image_timestamp = np.loadtxt(image_timestamp_dir, dtype=np.int64)
        image_dict = [{"image_name": image_names[i], "image_timestamp": image_timestamp[i].item()} for i in range(len(image_names))]

        event = h5py.File(event_dir, "r")
        event_slicer = EventSlicer(event)
        start_timestamp = event_slicer.get_start_time_us()
        end_timestamp = event_slicer.get_final_time_us()
        duration = (end_timestamp - start_timestamp)
        event_dict = event_slicer.get_events(start_timestamp, start_timestamp + duration)
        t = event_dict["t"]
        unique_moments, unique_index = np.unique(t, return_index=True)
        event_dict["unique_index"] = unique_index

        print(f"num of events: {t.shape[0]}")
        print(f"num of unique moments: {unique_moments.shape[0]}")
        print(f"event start timestamp: {t[0]}, end_timestamp: {t[-1]}, duration in sec: {(t[-1] - t[0]) / 1e6:.4f}s, in microseconds: {t[-1] - t[0]}us")
        print(f"image start timestamp: {image_dict[0]['image_timestamp']}, end_timestamp: {image_dict[-1]['image_timestamp']}")
        print(f"event fps: {len(unique_moments) / (end_timestamp - start_timestamp) * 1e6}")
        print(f"image fps: {len(image_dict) / (end_timestamp - start_timestamp) * 1e6}")
        print(f"num of images: {len(image_dict)}")
        return event_dict, image_dict

    def _get_camera(self, calib_dir) -> np.ndarray:
        def create_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            return T
        def invert_transform(T: np.ndarray) -> np.ndarray:
            R = T[:3, :3]
            t = T[:3, 3]
            R_inv = np.linalg.inv(R)
            t_inv = - R_inv @ t
            return create_transform(R_inv, t_inv)
        def cretate_K(intrinsic):
            fx_e, fy_e, cx_e, cy_e = intrinsic
            return np.array([[fx_e,    0,  cx_e],
                            [   0,  fy_e, cy_e],
                            [   0,     0,    1]])
        with open(calib_dir, 'r') as file:
            data = yaml.safe_load(file)
        file.close()
        intrin_event_dist = np.array(data["intrinsics"]["cam0"]["camera_matrix"])
        dist_coeffs = np.array(data["intrinsics"]["cam0"]["distortion_coeffs"])
        intrin_event_rect = np.array(data["intrinsics"]["camRect0"]["camera_matrix"])
        resolution = np.array(data["intrinsics"]["camRect0"]["resolution"])
        intrin_image = np.array(data["intrinsics"]["camRect1"]["camera_matrix"])
        T_i2e = np.array(data["extrinsics"]["T_10"])
        Re = np.array(data['extrinsics']['R_rect0'])
        Ri = np.array(data['extrinsics']['R_rect1'])
        Te = create_transform(Re, np.zeros(3))
        Ti = create_transform(Ri, np.zeros(3))

        K_event = cretate_K(intrin_event_rect)
        K_dist  = cretate_K(intrin_event_dist)
        K_image = cretate_K(intrin_image)

        T_i2e = Ti @ T_i2e @ invert_transform(Te)
        R_i2e = T_i2e[:3, :3]

        H_homography = K_image @ R_i2e @ np.linalg.inv(K_event)
        H_homography = np.linalg.inv(H_homography)

        return H_homography, K_event, K_dist, dist_coeffs, resolution, Re

    # ---------------------------------------------------------------------
    # NEW: replace accumulate_to_rgb with a LADS hybrid frame builder
    # ---------------------------------------------------------------------
    def _build_lads_hybrid(self, lads_proc: LADS, accu_x, accu_y, accu_p, sensor_HW):
        """NEW: build a (3, H, W) hybrid LADS frame from accumulated events.

        Channel 0/1: positive/negative event count map (|max|-normalized).
        Channel 2:   LADS time-surface (|max|-normalized).

        Returns: torch.float32 tensor (3, H, W).
        """
        H, W = sensor_HW
        if len(accu_x) == 0:
            return torch.zeros((3, H, W), dtype=torch.float32)
        # NEW: dataset_lads expects (N,4) [t_sec, x, y, p]; LADS itself only uses x,y,p,t.
        # For per-window LADS integration we synthesize a single-step timestamp delta.
        t_sec = np.zeros(len(accu_x), dtype=np.float32)
        chunk = np.stack([t_sec,
                          np.asarray(accu_x, dtype=np.float32),
                          np.asarray(accu_y, dtype=np.float32),
                          np.asarray(accu_p, dtype=np.float32)], axis=1)
        raw = _events_to_raw_frame(chunk, sensor_size=(H, W))  # (2,H,W) float32 in [-1,1]
        # NEW: feed window to LADS with an explicit positive time_diff_s (default 1 frame ~ 1.0)
        surface, _, _ = lads_proc.integrateEvents(chunk, time_diff_s=getattr(self, "_lads_dt", 1.0))
        surface = _normalize_map(surface).unsqueeze(0)  # (1,H,W)
        hybrid = torch.cat([raw, surface], dim=0)        # (3,H,W)
        return hybrid.to(torch.float32)

    @torch.no_grad()
    def save_warpped_event_pair(self, event_dict, image_dict, calib_dir, eventImage_dir, warpped_dir, vis_dir, label_dir):
        """Same iteration logic as pre_dse.py, but the event branch produces a
        LADS hybrid frame (and optionally passes through SmartRouterFrontEnd)
        saved as .pt; the image branch is unchanged.
        """
        t, x, y, p = event_dict["t"], event_dict["x"], event_dict["y"], event_dict["p"]
        accu_x, accu_y, accu_p = [], [], []
        j = 0
        H_homography, K_event, K_dist, dist_coeffs, resolution, Re = self._get_camera(calib_dir)
        W, H = resolution  # event-camera plane (W,H)
        unique_index = event_dict["unique_index"]

        # NEW: per-subfolder LADS processor (state is recursive across windows)
        lads_kwargs = dict(self.lads_kwargs)
        lads_proc = LADS(H=H, W=W,
                         device=lads_kwargs.pop("device", "cpu"),
                         ts_to_seconds_factor=lads_kwargs.pop("ts_to_seconds_factor", 1.0),
                         **lads_kwargs)
        router = self._build_router_if_needed(H=self.out_H, W=self.out_W)  # NEW

        # NEW: prepare remap once
        mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, Re, K_event, resolution, cv2.CV_32FC2)[0]

        prev_t_us = None  # NEW: track real per-window dt for LADS
        for i in tqdm(range(len(unique_index) - 1)):
            if j == len(image_dict):
                break
            accu_x.extend(x[unique_index[i]: unique_index[i+1]])
            accu_y.extend(y[unique_index[i]: unique_index[i+1]])
            accu_p.extend(p[unique_index[i]: unique_index[i+1]])

            current_image_timestamp = image_dict[j]["image_timestamp"]
            next_image_timestamp = image_dict[j + 1]["image_timestamp"] if (j + 1) < len(image_dict) else image_dict[j]["image_timestamp"]
            middle_timestamp = (current_image_timestamp + next_image_timestamp) / 2
            current_event_timestamp = t[unique_index[i]]

            if current_event_timestamp > middle_timestamp:
                # 1. warp the RGB intensity image (teacher branch — unchanged)
                image  = cv2.imread(image_dict[j]["image_name"])
                warped = cv2.warpPerspective(image, H_homography, (W, H),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT)

                # 2. NEW: build (3,H,W) hybrid LADS frame from the accumulated events
                if prev_t_us is None:
                    self._lads_dt = 1.0
                else:
                    self._lads_dt = max(1e-6, (current_event_timestamp - prev_t_us) / 1e6)
                prev_t_us = current_event_timestamp
                hybrid = self._build_lads_hybrid(lads_proc, accu_x, accu_y, accu_p, (H, W))  # (3,H,W)

                # 3. NEW: undistort/rectify each channel to the event-camera frame
                hybrid_np = hybrid.numpy().transpose(1, 2, 0)  # (H,W,3)
                hybrid_np = cv2.remap(hybrid_np, mapping, None, interpolation=cv2.INTER_LINEAR)
                hybrid = torch.from_numpy(hybrid_np.transpose(2, 0, 1)).contiguous().to(torch.float32)  # (3,H,W)

                # 4. NEW: resize to target output resolution (e.g. 224x224 for ViT-patch14)
                if (hybrid.shape[-2] != self.out_H) or (hybrid.shape[-1] != self.out_W):  # DIMENSION ADAPT
                    hybrid = F.interpolate(hybrid.unsqueeze(0), size=(self.out_H, self.out_W),
                                           mode="bilinear", align_corners=False).squeeze(0)

                # 5. NEW: optionally apply SmartRouterFrontEnd (gating) offline
                if router is not None:
                    inp = hybrid.unsqueeze(0).to(self.gate_device)
                    gated = router(inp).squeeze(0).to(torch.float32).cpu()
                    out_tensor = gated
                else:
                    out_tensor = hybrid

                # 6. save tensor (float32, 3 channels) and a PNG preview
                pt_path = os.path.join(eventImage_dir, f"{current_image_timestamp}.pt")
                torch.save(out_tensor, pt_path)

                if self.save_png_preview:
                    # NEW: simple visualization mapping (R<-pos, B<-neg, G<-surface)
                    prev = out_tensor.detach().clone()
                    for c in range(prev.shape[0]):
                        mx = prev[c].abs().max()
                        if mx > 0:
                            prev[c] = prev[c] / mx
                    rgb = torch.stack([
                        torch.clamp(prev[0], 0, 1),
                        torch.clamp(prev[2].abs(), 0, 1),
                        torch.clamp(prev[1], 0, 1),
                    ], dim=0)
                    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(eventImage_dir, f"{current_image_timestamp}.png"),
                                cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

                # warped image: resize to match output H,W so dataloader stays simple
                if (warped.shape[0] != self.out_H) or (warped.shape[1] != self.out_W):  # DIMENSION ADAPT
                    warped = cv2.resize(warped, (self.out_W, self.out_H), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(warpped_dir, f"{current_image_timestamp}.png"), warped)

                # optional vis triplet (event-preview, warped, label)
                label = os.path.join(label_dir, f"{current_image_timestamp}.png")
                if self.save_png_preview and os.path.exists(label):
                    try:
                        lab = cv2.imread(label)
                        lab = ((lab - lab.min()) / (lab.max() - lab.min() + 1e-8)) * 255
                        lab = lab.astype(np.uint8)
                        if (lab.shape[0] != warped.shape[0]) or (lab.shape[1] != warped.shape[1]):
                            lab = cv2.resize(lab, (warped.shape[1], warped.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                        prev_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR) if self.save_png_preview else warped
                        vis_img = np.concatenate([prev_bgr, warped, lab], axis=1)
                        cv2.imwrite(os.path.join(vis_dir, f"{current_image_timestamp}.png"), vis_img)
                    except Exception:
                        pass

                # reset accumulators
                accu_x, accu_y, accu_p = [], [], []
                j += 1

    def _process_subfolder(self, subfolder):
        print("---------------------------------- processing folder:", subfolder)
        event_dir = os.path.join(self.events_root, subfolder, "events", "left", "events.h5")
        image_dir = os.path.join(self.image_root, subfolder, "images", "left", "rectified")
        image_timestamp_dir = os.path.join(self.image_root, subfolder, "images", "timestamps.txt")
        calib_dir = os.path.join(self.calib_root, subfolder, "calibration", "cam_to_cam.yaml")
        save_root = os.path.join(self.image_root, subfolder, "images", "left")
        warpped_dir = os.path.join(save_root, "warpped")
        vis_dir = os.path.join(save_root, "vis")
        eventImage_dir = os.path.join(save_root, "eventImage")
        label_dir = os.path.join(self.sementatic_root, subfolder, "11classes")

        # label_dir = None
        print(f"event_dir: {event_dir}")
        print(f"image_dir: {image_dir}")
        print(f"calib_dir: {calib_dir}")
        print(f"label_dir: {label_dir}")

        print("removing both event images and warpped images to have a clean start")
        shutil.rmtree(warpped_dir, ignore_errors=True)
        shutil.rmtree(eventImage_dir, ignore_errors=True)
        shutil.rmtree(vis_dir, ignore_errors=True)

        os.makedirs(warpped_dir, exist_ok=True)
        os.makedirs(eventImage_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        print(f"warpped images will be saved in:        {warpped_dir}")
        print(f"LADS hybrid tensors will be saved in:    {eventImage_dir}")  # NEW (.pt + .png preview)

        event_dict, image_dict = self.load_event_image(image_dir, event_dir, image_timestamp_dir)
        self.save_warpped_event_pair(event_dict, image_dict, calib_dir, eventImage_dir, warpped_dir, vis_dir, label_dir)
        return subfolder

    def run(self, subfolders=None, n_workers=1):
        target_subfolders = [s for s in sorted(os.listdir(self.image_root)) if (subfolders is None or s in subfolders)]
        filtered = []
        for sf in target_subfolders:
            filtered.append(sf)
        target_subfolders = filtered
        if n_workers is None or n_workers <= 1:
            for sf in target_subfolders:
                self._process_subfolder(sf)
        else:
            print(f"Parallel processing with {n_workers} workers over {len(target_subfolders)} folders")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(self._process_subfolder, sf): sf for sf in target_subfolders}
                for f in as_completed(futures):
                    sf = futures[f]
                    try:
                        _ = f.result()
                        print(f"Finished: {sf}")
                    except Exception as e:
                        print(f"Failed: {sf} with error: {e}")

    def compute_rgb_stats(self, subfolders=None, save_to: str | None = None, workers: int = 1):
        """Compute per-channel mean/std for event (.pt hybrid) and image (.png) modalities."""
        ev_sum = np.zeros(3, dtype=np.float64)
        ev_sq_sum = np.zeros(3, dtype=np.float64)
        ev_count = 0

        img_sum = np.zeros(3, dtype=np.float64)
        img_sq_sum = np.zeros(3, dtype=np.float64)
        img_count = 0

        folders = [s for s in sorted(os.listdir(self.image_root)) if (subfolders is None or s in subfolders)]

        if workers is None or workers <= 1:
            for subfolder in tqdm(folders, desc="stats folders"):
                ev_s, ev_ss, ev_c, im_s, im_ss, im_c = _compute_stats_for_subfolder(self.image_root, subfolder)
                ev_sum += ev_s; ev_sq_sum += ev_ss; ev_count += ev_c
                img_sum += im_s; img_sq_sum += im_ss; img_count += im_c
        else:
            print(f"Parallel stats with {workers} workers over {len(folders)} folders")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_compute_stats_for_subfolder, self.image_root, sf): sf for sf in folders}
                for f in tqdm(as_completed(futures), total=len(futures), desc="stats futures"):
                    try:
                        ev_s, ev_ss, ev_c, im_s, im_ss, im_c = f.result()
                        ev_sum += ev_s; ev_sq_sum += ev_ss; ev_count += ev_c
                        img_sum += im_s; img_sq_sum += im_ss; img_count += im_c
                    except Exception as e:
                        sf = futures[f]
                        print(f"Failed stats for {sf}: {e}")

        def finalize(sum_, sq_sum_, count_):
            if count_ == 0:
                return [float('nan')] * 3, [float('nan')] * 3
            mean = sum_ / count_
            var = np.maximum(sq_sum_ / count_ - mean * mean, 0.0)
            std = np.sqrt(var)
            return mean.tolist(), std.tolist()

        ev_mean, ev_std = finalize(ev_sum, ev_sq_sum, ev_count)
        img_mean, img_std = finalize(img_sum, img_sq_sum, img_count)

        print("==== Dataset stats (event = LADS hybrid .pt, image = warpped png 0..1) ====")
        print(f"Event  mean: {ev_mean}")
        print(f"Event  std : {ev_std}")
        print(f"Image  mean: {img_mean}")
        print(f"Image  std : {img_std}")

        if save_to is not None:
            stats = {
                "event": {"mean": ev_mean, "std": ev_std},
                "image": {"mean": img_mean, "std": img_std},
            }
            out_dir = os.path.dirname(save_to)
            os.makedirs(out_dir if out_dir != "" else ".", exist_ok=True)
            with open(save_to, "w") as f:
                yaml.safe_dump(stats, f)
            print(f"Saved stats to {save_to}")

    @torch.no_grad()
    def process_tokens(self, subfolders=None, workers: int = 1):
        """Tokenize event/image pairs with a frozen ViT pair.

        NEW: event side reads .pt hybrid frames (already 3-ch, already resized),
        so no PIL/Normalize transforms are applied to it. The image side keeps
        the original Normalize/CenterCrop chain so the teacher branch is unchanged.
        """
        self.train_preprocessor     = PairedProcessor([
                                            ToTensor(type=self.type),
                                            Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                            PadToMinSide(target=(self.DSEC_H, self.DSEC_W), pad_x1=0, pad_x2=0),
                                            CenterCrop((self.DSEC_H, self.DSEC_W)),
                                            ])
        self.valid_preprocessor     = PairedProcessor([
                                                    ToTensor(type=self.type),
                                                    Normalize(self.DSEC_ME, self.DSEC_SE, self.DSEC_MI, self.DSEC_SI, type=self.type),
                                                    PadToMinSide(target=(self.DSEC_H, self.DSEC_W), pad_x1=0, pad_x2=0),
                                                    CenterCrop((self.DSEC_H, self.DSEC_W)),
                                                    ])
        self.image_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(self.device)
        self.event_encoder = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1e-6).to(self.device)
        self.image_encoder.load_state_dict(torch.load("/data/storage/jianwen/cache/dinov2/dinov2_vits14_pretrain.pth", weights_only=True), strict=True)
        self.event_encoder.load_state_dict(torch.load("/data/storage/jianwen/cache/ckpt_matters/gra_mixture_16x.pt", weights_only=True)["event_encoder"], strict=True)
        self.image_encoder.eval()
        self.event_encoder.eval()

        # NEW: normalization stats for the event branch — applied directly on the (3,H,W) hybrid tensor
        ev_mean = torch.tensor(getattr(self, "EV_MEAN", [0.0, 0.0, 0.0]), device=self.device).view(1, 3, 1, 1)
        ev_std  = torch.tensor(getattr(self, "EV_STD",  [1.0, 1.0, 1.0]), device=self.device).view(1, 3, 1, 1)

        for subfolder in sorted(os.listdir(self.image_root)):
            if subfolders is not None and subfolder not in subfolders:
                continue
            print("---------------------------------- merging tokens in folder:", subfolder)
            save_root = os.path.join(self.image_root, subfolder, "images", "left")
            eventToken_dir = os.path.join(save_root, "eventToken")
            imageToken_dir = os.path.join(save_root, "imageToken")
            warpped_dir = os.path.join(save_root, "warpped")
            eventImage_dir = os.path.join(save_root, "eventImage")
            print("removing to have a clean start")
            shutil.rmtree(eventToken_dir, ignore_errors=True)
            shutil.rmtree(imageToken_dir, ignore_errors=True)
            os.makedirs(eventToken_dir, exist_ok=True)
            os.makedirs(imageToken_dir, exist_ok=True)

            # NEW: drive iteration off the warpped pngs and pair to .pt by stem
            names = sorted([f for f in os.listdir(warpped_dir) if f.lower().endswith(".png")])

            def _process_one(name: str):
                stem = os.path.splitext(name)[0]
                warped = Image.open(os.path.join(warpped_dir, name))
                # NEW: event tensor comes pre-built from save_warpped_event_pair
                event_pt = os.path.join(eventImage_dir, f"{stem}.pt")
                if not os.path.exists(event_pt):
                    return
                event_rgb_t = torch.load(event_pt, map_location="cpu").to(torch.float32)  # (3,H,W)

                if self.split == "train":
                    _, warped_t = self.train_preprocessor(Image.fromarray(np.zeros((event_rgb_t.shape[1], event_rgb_t.shape[2], 3), dtype=np.uint8)), warped)
                else:
                    _, warped_t = self.valid_preprocessor(Image.fromarray(np.zeros((event_rgb_t.shape[1], event_rgb_t.shape[2], 3), dtype=np.uint8)), warped)

                # DIMENSION ADAPT: pad/resize event tensor to (3, DSEC_H, DSEC_W) before normalization
                if (event_rgb_t.shape[-2] != self.DSEC_H) or (event_rgb_t.shape[-1] != self.DSEC_W):
                    event_rgb_t = F.interpolate(event_rgb_t.unsqueeze(0), size=(self.DSEC_H, self.DSEC_W),
                                                mode="bilinear", align_corners=False).squeeze(0)
                event_rgb_t = event_rgb_t.unsqueeze(0).to(self.device)
                event_rgb_t = (event_rgb_t - ev_mean) / ev_std

                image_tokens = self.image_encoder.forward_features(warped_t.unsqueeze(0).to(self.device))["x_norm_patchtokens"].squeeze(0).cpu()
                event_tokens = self.event_encoder.forward_features(event_rgb_t)["x_norm_patchtokens"].squeeze(0).cpu()
                torch.save(image_tokens, os.path.join(imageToken_dir, f"{stem}.pt"))
                torch.save(event_tokens, os.path.join(eventToken_dir, f"{stem}.pt"))

            if workers is None or workers <= 1:
                for name in tqdm(names):
                    _process_one(name)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(tqdm(ex.map(_process_one, names), total=len(names)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="D:/ANew_Stage/Achenteacher/generative_event_pretraining-master/data/DSEC", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--workers", default=2, type=int, help="number of parallel processes for run()")
    parser.add_argument("--stats_out", default=None, type=str, help="optional path to save stats as YAML")
    # NEW: LADS / gating CLI knobs
    parser.add_argument("--lads_decay_func", default="er", type=str)
    parser.add_argument("--lads_decay_param", default=0.2, type=float)
    parser.add_argument("--lads_patch_size", default=32, type=int)
    parser.add_argument("--lads_interpolate_patches", action="store_true")
    parser.add_argument("--lads_min_decay", default=0.0, type=float)
    parser.add_argument("--lads_ts_to_seconds_factor", default=1.0, type=float)
    parser.add_argument("--gate_ckpt", default=None, type=str, help="optional SmartRouterFrontEnd ckpt path")
    parser.add_argument("--gate_device", default="cpu", type=str)
    parser.add_argument("--out_h", default=224, type=int, help="DSEC output height (DIMENSION ADAPT)")
    parser.add_argument("--out_w", default=224, type=int, help="DSEC output width (DIMENSION ADAPT)")
    parser.add_argument("--save_png_preview", action="store_true")
    args = parser.parse_args()

    for split in ["train", "test"]:
        args.split = split

        processor = Processor(args)
        subfolders = os.listdir(processor.sementatic_root) if os.path.isdir(processor.sementatic_root) else None
        processor.run(subfolders=subfolders, n_workers=args.workers)
        # processor.process_tokens(workers=args.workers)
        # processor.compute_rgb_stats(save_to=args.stats_out, workers=args.workers)