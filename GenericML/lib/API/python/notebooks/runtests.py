# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import subprocess
import sys

if __name__ == "__main__":
    for file in os.listdir(sys.argv[1]):
        if file.endswith(".ipynb"):
            subprocess.run(["exec_nb", "--exc_stop", file])
