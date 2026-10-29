"""A45 disk guards. Import these in every E5 script that writes gradient data.

Three guards, none of which existed before (checked: no `statvfs` or
`shutil.disk_usage` call anywhere under EXPERIMENTS/, and refuse-to-overwrite
existed only as a skip-if-present in build_matchedK.py).

    check_free(min_gb)          stop cleanly BEFORE writing if space is short
    open_new(path)              refuse to overwrite an existing output
    assert_sketch_size(path, d) abort if the first sketch is not the size the
                                projection dimension implies

The third is the one that catches the failure shape we are actually afraid of:
a bug that writes FULL vectors instead of sketches fills 73 GB in about a
dozen prompts, silently, and is visible only afterwards. Checking the first
file written costs nothing and fails in one prompt instead of twelve.

Self-test:  $HOME/miniconda3/envs/vllm/bin/python guards.py
"""
import os
import shutil

FLOAT_BYTES = {"float16": 2, "float32": 4, "float64": 8}


class DiskGuard(Exception):
    pass


def check_free(min_gb, path=None, what=""):
    """Stop cleanly while there is still room to stop."""
    path = path or os.path.dirname(os.path.abspath(__file__))
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise DiskGuard(
            f"free space {free_gb:.1f} GB is below the {min_gb:.1f} GB floor"
            f"{' before ' + what if what else ''}. Stopping before writing "
            "rather than filling the disk.")
    return free_gb


def open_new(path, mode="wb"):
    """Refuse to overwrite. Same policy adopted after the variance-file bug."""
    if os.path.exists(path):
        raise DiskGuard(
            f"refusing to overwrite existing output {path}. Delete it "
            "deliberately if that is what you mean.")
    return open(path, mode)


def expected_sketch_bytes(d_proj, dtype="float32", n_vectors=1, overhead=0):
    return d_proj * FLOAT_BYTES[dtype] * n_vectors + overhead


def assert_sketch_size(path, d_proj, dtype="float32", n_vectors=1,
                       tol=2.0, overhead=4096):
    """Abort if the FIRST written sketch is not the size d_proj implies.

    tol is a multiplicative factor: the file must be within [1/tol, tol] of
    the expected size. A full 0.5B lm_head vector is 272 MB against a 65536-
    bucket sketch of 0.26 MB, i.e. ~1039x, so any sane tol catches it on the
    first prompt.
    """
    if not os.path.exists(path):
        raise DiskGuard(f"expected a sketch at {path}, found nothing")
    got = os.path.getsize(path)
    exp = expected_sketch_bytes(d_proj, dtype, n_vectors)
    if got > exp * tol + overhead or got * tol + overhead < exp:
        raise DiskGuard(
            f"first sketch {path} is {got / 1e6:.3f} MB but the projection "
            f"dimension {d_proj} implies {exp / 1e6:.3f} MB "
            f"({got / max(exp, 1):.1f}x). This is the shape of a bug that "
            "writes FULL gradients instead of sketches; aborting after one "
            "prompt instead of filling the disk.")
    return got, exp


if __name__ == "__main__":
    import tempfile

    D, DT = 65536, "float32"
    print(f"expected sketch at D_PROJ={D} {DT}: "
          f"{expected_sketch_bytes(D, DT) / 1e6:.3f} MB")
    print(f"free space now: {check_free(0.1):.1f} GB")

    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.bin")
        with open(good, "wb") as f:
            f.write(b"\0" * expected_sketch_bytes(D, DT))
        print("correct-size sketch:", assert_sketch_size(good, D, DT), "-> OK")

        bad = os.path.join(td, "bad.bin")
        with open(bad, "wb") as f:                    # 100x too big
            f.write(b"\0" * (expected_sketch_bytes(D, DT) * 100))
        try:
            assert_sketch_size(bad, D, DT)
            print("FAIL: oversized sketch was not caught")
        except DiskGuard as e:
            print("oversized sketch CAUGHT:", str(e)[:110], "...")

        try:
            open_new(good)
            print("FAIL: overwrite was not refused")
        except DiskGuard as e:
            print("overwrite REFUSED:", str(e)[:80], "...")

        try:
            check_free(10_000_000, td, "the gradient pass")
            print("FAIL: low-space was not caught")
        except DiskGuard as e:
            print("low free space CAUGHT:", str(e)[:90], "...")
    print("\nall three guards behave as specified")
