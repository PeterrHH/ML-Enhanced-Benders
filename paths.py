"""Resolve repo-anchored resources and root-anchored data/output trees.

Two kinds of path live in this project:

* Repo resources -- configs/ and inputs/ -- are version-controlled, read-only
  and small. They are anchored to the repository via __file__, so an entry
  point works from any working directory (a SLURM job starts in whatever
  directory sbatch inherited).
* Data and outputs are large and regenerable, and on a cluster belong under
  $SCRATCH. They are anchored to a root chosen on the command line.

Imports only os, so gep_benders.py can use it without pulling in main.py's
import chain.
"""

import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _expand(path):
    return os.path.expanduser(os.path.expandvars(path))


def _first(*candidates):
    """First non-empty candidate. Empty strings fall through, matching the
    `or`-chain the original get_data_root used."""
    for candidate in candidates:
        if candidate:
            return candidate
    return None


def under_repo(path):
    """Anchor a repo resource (configs/, inputs/) to the repo, not the CWD.

    Absolute paths pass through, so --config /scratch/me/custom.json works.
    """
    path = _expand(path)
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def under_root(path, root):
    """Put a relative path under `root`. Absolute paths pass through."""
    path = _expand(path)
    return path if os.path.isabs(path) else os.path.join(root, path)


#! main.py has always called this helper by this name; keep it working.
under_data_root = under_root


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def add_path_args(parser):
    """Register --home-path / --data-root / --output-root on an ArgumentParser."""
    parser.add_argument(
        "--home-path", "--home_path",
        dest="home_path",
        default=None,
        help=(
            "Master folder holding data/ and outputs/. Use '.' locally and "
            "$SCRATCH/<project> on a cluster."
        ),
    )
    parser.add_argument(
        "--data-root", "--data_root",
        dest="data_root",
        default=None,
        help=(
            "Parent folder for generated/loaded datasets. Overrides "
            "--home-path. Relative dataset paths such as data/ED_data/... "
            "become <data-root>/data/ED_data/...."
        ),
    )
    parser.add_argument(
        "--output-root", "--output_root",
        dest="output_root",
        default=None,
        help=(
            "Parent folder for outputs/ and the W&B run directory. "
            "Overrides --home-path."
        ),
    )
    return parser


def resolve_roots(config=None, cli=None):
    """Resolve data_root and output_root, first non-empty wins:

        CLI specific > CLI --home-path > config specific > env specific
                     > config home_path > $PDL_HOME > "." (cwd)

    The CLI beats the config deliberately: a stale relative "data_root" left
    in a JSON must not silently override a scratch path passed on the command
    line.

    Returns absolute paths for home_path, data_root and output_root.
    """
    config = config or {}
    cli_home = getattr(cli, "home_path", None)
    cfg_home = config.get("home_path")
    env_home = os.environ.get("PDL_HOME")

    data = _first(
        getattr(cli, "data_root", None),
        cli_home,
        config.get("data_root"),
        os.environ.get("DATA_ROOT"),
        cfg_home,
        env_home,
        ".",
    )
    output = _first(
        getattr(cli, "output_root", None),
        cli_home,
        config.get("output_root"),
        os.environ.get("OUTPUT_ROOT"),
        cfg_home,
        env_home,
        ".",
    )
    home = _first(cli_home, cfg_home, env_home, ".")

    return {
        "home_path": os.path.abspath(_expand(home)),
        "data_root": os.path.abspath(_expand(data)),
        "output_root": os.path.abspath(_expand(output)),
    }
