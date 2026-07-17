"""Create GitHub repo for game-digit-trainer using git credential helper."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request


def git_credential(host: str) -> dict[str, str]:
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input=f"url=https://{host}\n\n",
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def github_create_repo(name: str, desc: str, token: str) -> dict:
    url = "https://api.github.com/user/repos"
    body = json.dumps(
        {
            "name": name,
            "description": desc,
            "private": False,
            "auto_init": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    name = "game-digit-trainer"
    desc = "Game HUD multi-font digit trainer: segment, correct, train CNN, export ONNX"
    cred = git_credential("github.com")
    token = cred.get("password", "")
    if not token:
        print("NO_TOKEN")
        return 2
    try:
        repo = github_create_repo(name, desc, token)
        print("OK", repo.get("html_url", ""), repo.get("clone_url", ""))
        return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print("HTTP", exc.code, body[:500])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
