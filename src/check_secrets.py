"""Check Git content against local dotenv secrets without printing their values."""
import subprocess
from pathlib import Path
from dotenv import dotenv_values


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    secrets = [str(value).encode() for name, value in dotenv_values(".env").items()
               if value and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
               and str(value).lower() not in {"sample", "your_api_key_here", "your_ecos_api_key"}]
    if git("ls-files", "--", ".env").strip():
        raise SystemExit("STOP: .env is tracked")
    if subprocess.run(["git", "check-ignore", "-q", ".env"]).returncode:
        raise SystemExit("STOP: .env is not ignored")
    template = git("show", "HEAD:.env.example")
    if any(secret in template for secret in secrets):
        raise SystemExit("STOP: secret detected in template")
    if not Path(".env.example").exists():
        Path(".env.example").write_bytes(template)
    seen = set()
    for line in git("rev-list", "--objects", "HEAD").splitlines():
        oid = line.split()[0].decode()
        if oid in seen:
            continue
        seen.add(oid)
        if git("cat-file", "-t", oid).strip() == b"blob":
            if any(secret in git("cat-file", "blob", oid) for secret in secrets):
                raise SystemExit("STOP: local secret detected in Git history")
    paths = git("ls-files", "--cached", "-z").split(b"\0")
    for raw in paths:
        if not raw:
            continue
        path = raw.decode("utf-8")
        content = git("show", ":" + path)
        if any(secret in content for secret in secrets):
            raise SystemExit("STOP: local secret detected in Git index")
    print("PASS: .env ignored and untracked; local dotenv secrets absent from history and index")
    print("Checked", len(secrets), "local secret values without printing them")


if __name__ == "__main__":
    main()
