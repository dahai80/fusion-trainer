import logging
import re
from importlib.metadata import PackageNotFoundError, requires

logger = logging.getLogger(__name__)

# fusion-trainer is an HTTP client to fusion-mlx (localhost:11434).
# It must NOT pull in-process native MLX libs or a file:// engine pin.
# See issue #14: over-declared unused mlx/mlx-lm/mlx-vlm + file:// fusion-mlx.
FORBIDDEN_DEP_PATTERNS = [
    re.compile(r"^mlx$", re.IGNORECASE),
    re.compile(r"^mlx-lm$", re.IGNORECASE),
    re.compile(r"^mlx-vlm$", re.IGNORECASE),
    re.compile(r"fusion-mlx\s*@", re.IGNORECASE),
    re.compile(r"file://", re.IGNORECASE),
]

REQUIRED_DEP = "httpx"


def _declared_deps():
    try:
        reqs = requires("fusion-trainer")
    except PackageNotFoundError:
        logger.error("fusion-trainer not installed; run `pip install -e .`")
        return None
    if not reqs:
        return []
    deps = []
    for raw in reqs:
        # strip environment markers, extras, and version specifiers → base name
        name = raw.split(";")[0].split("[")[0].strip()
        name = re.split(r"[<>=!~]", name, maxsplit=1)[0].strip()
        if name:
            deps.append(name)
    logger.debug("declared deps: %s", deps)
    return deps


class TestDependencyManifest:
    def test_package_is_installed(self):
        deps = _declared_deps()
        assert deps is not None, "fusion-trainer must be installed for dep checks"

    def test_no_in_process_mlx_libs(self):
        deps = _declared_deps()
        assert deps is not None
        offenders = []
        for dep in deps:
            for pat in FORBIDDEN_DEP_PATTERNS:
                if pat.search(dep):
                    offenders.append(dep)
        assert not offenders, (
            f"forbidden native-MLX/file:// deps declared: {offenders} "
            "(fusion-trainer talks to fusion-mlx over HTTP, not in-process)"
        )

    def test_httpx_is_declared(self):
        deps = _declared_deps()
        assert deps is not None
        names_lower = [d.lower() for d in deps]
        assert REQUIRED_DEP in names_lower, (
            f"{REQUIRED_DEP} must be a declared dep "
            "(used by mlx_client/cli/grpo/sft trainer to reach fusion-mlx)"
        )

    def test_no_file_engine_pin(self):
        deps = _declared_deps()
        assert deps is not None
        for dep in deps:
            assert not dep.startswith("file://"), f"file:// pin is non-portable: {dep}"
