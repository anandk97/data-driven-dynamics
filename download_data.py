"""Download the six CTF datasets used here from the CTF-for-Science OSF project (osf.io/6rzhm) into ./data."""

import tarfile
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
FILES = {  # dataset -> OSF file id (see https://api.osf.io/v2/nodes/6rzhm/files/osfstorage/)
    "ODE_Lorenz": "yga95",  # 2 MB, public test set included
    "PDE_KS": "j8hrb",  # 0.7 GB, public test set included
    "seismo": "nrufb",  # 0.2 GB, test set withheld
    "ocean_das": "hc58g",  # 0.2 GB, test set withheld
    "sst": "jzc6t",  # 1.3 GB, test set withheld
    "msfr": "h5qc2",  # 1.0 GB, CTF4Nuclear molten salt fast reactor, test set withheld
}

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name, fid in FILES.items():
        if (DATA / name).exists():
            continue
        archive = DATA / f"{name}.tar.gz"
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"https://osf.io/download/{fid}/", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(DATA, filter="data")
        archive.unlink()
        for junk in list(DATA.rglob("._*")) + list(DATA.rglob(".DS_Store")):  # macOS metadata in the archives
            junk.unlink()
