"""VARN-LQG -- offline: the WFS's measurement-noise covariance, R_noise.npy and R_aliasing.npy.

Not run by rama_cup: run it once, on a machine with AI4AO (torch, the RAMA calibration), and keep
the two files in this folder. optimizer_varn_lqg.py uses R = R_noise + R_aliasing.

  R_noise     photon and read-out noise of the pyramid, propagated through the linear
              reconstructor (recon, [n_pix, n_modes]), with the intensity normalized to 1 photon:
                  R_noise = RON^2 / Nph^2 recon^T recon + 1 / Nph (I_ref * recon)^T recon
  R_aliasing  covariance of the modes the WFS reconstructs from a residual phase with every
              mode the DM can make removed (levelOfCorrection 1): the high orders seen as low
              ones. Computed without WFS noise, so that it does not count the noise twice.

Both are in the units of the reconstructed modes: they must be those of rama_cup's da.input
(RAMA's pyrModes, the same M2C) for R to be right.

    python make_R.py --params Rama_params.py --nphotons 1e5 --ron 2 --r0 0.1
"""
import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from AI4AO import DeformableMirror, PhaseDataset, PyramidWFS
from AI4AO.paths import DATA_DIR

HERE = Path(__file__).resolve().parent


def noise_covariance(recon, ref_intensity, nphotons, ron):
    """Photon + read-out noise of the modes (Kalman_tools_numpy.estimate_R), [n_modes, n_modes]."""
    cov_ron = ron ** 2 / nphotons ** 2 * recon.T @ recon
    cov_photon = 1 / nphotons * (ref_intensity.reshape(-1, 1) * recon).T @ recon
    return cov_ron + cov_photon


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", required=True, help="the experiment's parameter file (Rama_params.py)")
    parser.add_argument("--data-dir", default=str(DATA_DIR / "Rama"),
                        help="folder with M2C.npy, RamaWFS.pth, RamaDM.pth (default: %(default)s)")
    parser.add_argument("--n-modes", type=int, default=None, help="modes (default: all of M2C, 96 on RAMA)")
    parser.add_argument("--nphotons", type=float, default=None,
                        help="photons per frame in the WFS (default: the WFS model's)")
    parser.add_argument("--ron", type=float, default=None, help="read-out noise, e-/pixel (default: the WFS model's)")
    parser.add_argument("--r0", type=float, default=None,
                        help="r0 (m) of the aliasing phases (default: the params file's range)")
    parser.add_argument("--batches", type=int, default=1000, help="phase batches for R_aliasing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(HERE), help="where to write the files (default: this folder)")
    args = parser.parse_args()

    params = runpy.run_path(args.params)
    WFSParams, AtmosParams = dict(params["WFSParams"]), dict(params["AtmosParams"])
    LoopParams, DMParams = dict(params["LoopParams"]), dict(params["DMParams"])
    data_dir, device = Path(args.data_dir), args.device

    M2C = np.load(data_dir / "M2C.npy").astype(np.float32)
    n_modes = args.n_modes or M2C.shape[1]

    WFSParams["centralObstruction"] = 0.0
    wfs = PyramidWFS(WFSParams, device)
    wfs.LoadCalibration(data_dir / "RamaWFS.pth")
    wfs.eval()
    dm = DeformableMirror(WFSParams, DMParams, device)
    dm.LoadCalibration(data_dir / "RamaDM.pth")
    nphotons = args.nphotons if args.nphotons is not None else float(wfs.Nphotons)
    ron = args.ron if args.ron is not None else float(wfs.RON)

    with torch.no_grad():
        modes = dm(torch.from_numpy(M2C.T).to(device))[:n_modes]
        wfs.modulation = 0
        wfs.BuildReconstructionMatrix(modes)

        # --- noise
        R_noise = noise_covariance(wfs.reconstructionMatrix, wfs.reference_intensity, nphotons, ron)
        R_noise = R_noise.squeeze().double().cpu().numpy()

        # --- aliasing: residual phases with the DM's modes removed, through the noiseless WFS
        AtmosParams["Nphases"] = 16
        if args.r0 is not None:
            AtmosParams["r0"] = [args.r0, args.r0]
        LoopParams["levelOfCorrection"] = [1.0, 1.0]
        dataset = PhaseDataset(WFSParams, AtmosParams, LoopParams, DMParams, device)
        dataset.generateClosedLoop = True
        use_noise, wfs.useNoise = wfs.useNoise, False
        t0, outputs = time.time(), []
        for _ in range(args.batches):
            batch = dataset[0]
            frames = wfs(batch["opd"], batch["pupil"])
            outputs.append(wfs.GetReconstructedOPD(frames).reshape(-1, n_modes).double().cpu())
        wfs.useNoise = use_noise
        outputs = torch.cat(outputs).numpy()
        R_aliasing = np.cov(outputs.T)

    out = Path(args.out)
    np.save(out / "R_noise.npy", R_noise.astype(np.float64))
    np.save(out / "R_aliasing.npy", R_aliasing.astype(np.float64))
    info = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "params": str(args.params), "data_dir": str(data_dir),
            "n_modes": n_modes, "nphotons": nphotons, "ron": ron,
            "r0": args.r0 if args.r0 is not None else AtmosParams["r0"],
            "aliasing_samples": int(outputs.shape[0]),
            "R_noise_mean_variance": float(np.mean(np.diag(R_noise))),
            "R_aliasing_mean_variance": float(np.mean(np.diag(R_aliasing)))}
    (out / "R_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(f"R_noise.npy, R_aliasing.npy, R_info.json written to {out} ({time.time() - t0:.0f} s for the aliasing)")


if __name__ == "__main__":
    main()
