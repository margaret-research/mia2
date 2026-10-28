import base64
import os
import sys
import zlib

#__FILES__


def extract(dest):
    for name, e in FILES.items():
        p = os.path.join(dest, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(zlib.decompress(base64.b64decode(e)))
    print(len(FILES), "files ->", os.path.abspath(dest))


if __name__ == "__main__":
    if "--list" in sys.argv:
        for n in FILES:
            print(n)
    else:
        extract(sys.argv[1] if len(sys.argv) > 1 else "T026_restored")
