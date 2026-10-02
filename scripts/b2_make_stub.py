"""B2: write the STUB keypoint file data/derived/keypoints/stub_session_032.npz (contract v2).

Synthetic keypoints from Wuji FK with known joint angles (see wuji_cap/sim/stub_keypoints.py).
Used only until A2 writes the real data/derived/keypoints/session_032.npz.

Usage: uv run python scripts/b2_make_stub.py
"""

from __future__ import annotations

from wuji_cap.common import derived_io
from wuji_cap.common.provenance import REPO_ROOT, provenance
from wuji_cap.sim import stub_keypoints, wuji_model as wm

OUT = REPO_ROOT / "data/derived/keypoints/stub_session_032.npz"


def main() -> None:
    arrays, params = stub_keypoints.make_stub()
    derived_io.check_keypoints(arrays)
    sources = {
        str(wm.urdf_path(s)): wm.sha256(wm.urdf_path(s)) for s in ("left", "right")
    }
    meta = {**provenance("scripts/b2_make_stub.py"), "sources": sources,
            "wuji_description_commit": wm.wuji_description_commit(), "params": params}
    derived_io.write_npz(OUT, arrays, meta)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
