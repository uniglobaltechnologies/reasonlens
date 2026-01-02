import modal
import subprocess
import json
import os
import glob
import re
import time
from typing import Dict, Any, Optional


def parse_scores_from_xml(response_text: str) -> Dict[str, float]:
    """Parse scores from XML format in judge responses."""
    import re
    scores = {}
    # Match patterns like <score name="some_metric">0.85</score>
    pattern = r'<score\s+name=["\'](\w+)["\'](?: [^>]*)?>([\d.]+)</score>'
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    for name, value in matches:
        try:
            scores[name.lower()] = float(value)
        except ValueError:
            pass
    return scores

app = modal.App("glassroom-petri")

# Pin versions for stability
petri_image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install("git")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install --index-url https://pypi.org/simple 'inspect-ai==0.3.140' fastapi uvicorn",
        "python -m pip install git+https://github.com/safety-research/petri",
    )
)

CALLBACK_ENV_KEYS = ["PETRI_CALLBACK_SECRET"]


def _get_callback_secret() -> str:
    for key in CALLBACK_ENV_KEYS:
        raw = os.environ.get(key)
        if raw:
            token = raw.strip()
            start = token[:4]
            end = token[-4:] if len(token) >= 4 else ""
            print(f"[callback] Found env {key}; len={len(token)}, partial={start}...{end}")
            if len(token) < 8:
                print("[callback] WARNING: callback secret length looks suspiciously short")
            return token
    print("[callback] ERROR: No PETRI callback secret found in env vars:", CALLBACK_ENV_KEYS)
    return ""


def _extract_epoch_number(path: str, parsed_json: Optional[dict] = None) -> Optional[int]:
    # Prefer metadata if present
    if isinstance(parsed_json, dict):
        md = parsed_json.get("metadata")
        if isinstance(md, dict):
            for k in ("epoch", "epoch_number", "repeat", "replicate"):
                v = md.get(k)
                if isinstance(v, int):
                    return v
                if isinstance(v, str) and v.isdigit():
                    return int(v)

    # Heuristic from filename/path
    m = re.search(r"(?:epoch|ep)[-_ ]?(\d+)", path, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    return None


def _slim_for_callback(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send full transcript content (no truncation) because Lovable wants full JSON.
    """
    out = dict(resp)
    tx = []
    for t in resp.get("transcripts", []):
        tx.append(
            {
                "path": t.get("path"),
                "scenario_id": t.get("scenario_id"),
                "epoch_number": t.get("epoch_number"),
                "judge_scores": t.get("judge_scores", {}),
                "flags": t.get("flags", []),
                "language": t.get("language", "en"),
                "content": t.get("content"),
            }
        )
    out["transcripts"] = tx
    return out


def _send_callback(callback_url: str, resp: Dict[str, Any], token: str):
    if not callback_url:
        print("[callback] No callback URL provided, skipping callback")
        return

    headers = {"Content-Type": "application/json"}
    tok = (token or "").strip()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
        headers["x-api-key"] = tok

    try:
        import urllib.request

        payload = json.dumps(_slim_for_callback(resp)).encode("utf-8")
        req = urllib.request.Request(callback_url, data=payload, headers=headers)

        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    print(f"[callback] Callback sent successfully: {r.status}")
                    return
            except Exception as e:
                print(f"[callback] Callback attempt {attempt + 1} failed: {e}")
                time.sleep(2**attempt)

        print("[callback] Failed to send callback after retries")
    except Exception as e:
        print(f"[callback] Failed to send callback: {e}")


@app.function(
    image=petri_image,
    secrets=[
        modal.Secret.from_name("anthropic"),
        modal.Secret.from_name("openai"),
        modal.Secret.from_name("petri-callback"),
    ],
    timeout=3600,
    cpu=2.0,
    memory=4096,
)
@modal.fastapi_endpoint(method="POST", label="api")
def run_petri_audit(data: Dict[str, Any]):
    """
    Body:
    {
      "run_id": "uuid",
      "scenarios": [{"pack_id":"...", "seed_instruction":"..."}],
      "auditor_model": "provider/model",
      "target_model": "provider/model",
      "judge_model": "provider/model",
      "max_turns": 10,
      "samples_per_scenario": 1,          # mapped to Inspect AI --epochs
      "epochs_reducer": "mean",           # optional
      "cap_tokens": 1000,                 # optional (PETRI may ignore)
      "cap_cost": 2,                      # optional (PETRI may ignore)
      "callback_url": "https://.../functions/v1/petri-audit-callback"
    }
    """
    run_id = data.get("run_id") or "run"
    scenarios = data.get("scenarios", [])
    auditor_model = data.get("auditor_model")
    target_model = data.get("target_model")
    judge_model = data.get("judge_model")
    max_turns = int(data.get("max_turns", 10))

    samples_per_scenario = int(data.get("samples_per_scenario", 1))
    epochs = max(1, samples_per_scenario)
    epochs_reducer = str(data.get("epochs_reducer", "mean") or "mean").strip()

    cap_tokens = data.get("cap_tokens")
    cap_cost = data.get("cap_cost")
    callback_url = data.get("callback_url") or ""

    callback_secret = _get_callback_secret()

    out_dir = f"/tmp/outputs/{run_id}"
    os.makedirs(out_dir, exist_ok=True)

    resp: Dict[str, Any] = {
        "run_id": run_id,
        "status": "completed",
        "cost_tokens": 0,
        "cost_currency": 0.0,
        "transcripts": [],
        "error_message": None,
        "samples_per_scenario": epochs,
        "epochs_reducer": epochs_reducer,
    }

    try:
        for i, scenario in enumerate(scenarios):
            pack_id = scenario.get("pack_id") or f"scenario_{i+1}"
            seed = scenario.get("seed_instruction")

            if isinstance(seed, list):
                seed_str = " ".join([str(s) for s in seed])
            elif seed is None:
                seed_str = ""
            else:
                seed_str = str(seed)

            scenario_dir = os.path.join(out_dir, pack_id)
            os.makedirs(scenario_dir, exist_ok=True)

            stdout_path = os.path.join(scenario_dir, "stdout.txt")
            stderr_path = os.path.join(scenario_dir, "stderr.txt")

            # Write special_instructions JSON to file so PETRI resource() reads it as a STRING
            # and PETRI can then parse JSON list-of-strings correctly.
            special_instructions_json = json.dumps([seed_str])
            special_path = os.path.join(scenario_dir, "special_instructions.json")
            with open(special_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(special_instructions_json)

            log_dir = os.path.join(scenario_dir, "inspect_logs")
            os.makedirs(log_dir, exist_ok=True)

            print(f"Scenario: {pack_id}")
            print(f"Seed length: {len(seed_str)} chars")
            print(f"Seed preview: {seed_str[:120]}")
            print(f"max_turns={max_turns}, epochs={epochs}, epochs_reducer={epochs_reducer}")

            cmd = [
                "inspect",
                "eval",
                "petri/audit",
                "--model-role",
                f"auditor={auditor_model}",
                "--model-role",
                f"target={target_model}",
                "--model-role",
                f"judge={judge_model}",
                "--log-dir",
                log_dir,
            ]

            if epochs > 1:
                cmd += ["--epochs", str(epochs)]
                if epochs_reducer:
                    cmd += ["--epochs-reducer", epochs_reducer]

            cmd += [
                "-T",
                f"special_instructions={special_path}",
                "-T",
                f"max_turns={max_turns}",
                "-T",
                f"transcript_save_dir={scenario_dir}",
            ]

            if cap_tokens is not None:
                cmd += ["-T", f"cap_tokens={cap_tokens}"]
            if cap_cost is not None:
                cmd += ["-T", f"cap_cost={cap_cost}"]

            print(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)

            # Save logs
            with open(stdout_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(result.stdout or "")
            with open(stderr_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(result.stderr or "")

            if result.returncode != 0:
                tail = ""
                lines = (result.stderr or "").splitlines()
                if lines:
                    tail = lines[-1]
                resp["status"] = "failed"
                resp["error_message"] = f"PETRI failed for {pack_id} (exit {result.returncode}): {tail}"

                resp["transcripts"].append(
                    {
                        "path": stderr_path,
                        "scenario_id": pack_id,
                        "epoch_number": None,
                        "content": result.stderr or "",
                        "judge_scores": {},
                        "flags": [],
                        "language": "en",
                    }
                )
                resp["transcripts"].append(
                    {
                        "path": stdout_path,
                        "scenario_id": pack_id,
                        "epoch_number": None,
                        "content": result.stdout or "",
                        "judge_scores": {},
                        "flags": [],
                        "language": "en",
                    }
                )

                _send_callback(callback_url, resp, callback_secret)
                return resp

            # Collect transcript JSON files written by PETRI / inspect
            transcript_files = glob.glob(f"{scenario_dir}/**/*.json", recursive=True)
            if not transcript_files:
                # Include logs to help debug
                resp["transcripts"].append(
                    {
                        "path": stdout_path,
                        "scenario_id": pack_id,
                        "epoch_number": None,
                        "content": (result.stdout or ""),
                        "judge_scores": {},
                        "flags": [],
                        "language": "en",
                    }
                )
            else:
                for tf in sorted(transcript_files):
                    try:
                        with open(tf, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        parsed = None
                        judge_data: Dict[str, Any] = {}
                        flags = []
                        language = "en"

                        try:
                            parsed = json.loads(content)

                            judge_output = None
                            if isinstance(parsed.get("metadata"), dict) and parsed["metadata"].get("judge_output") is not None:
                                judge_output = parsed["metadata"]["judge_output"]
                            elif isinstance(parsed.get("judge_output"), dict):
                                judge_output = parsed["judge_output"]

                            if isinstance(judge_output, dict):
                                judge_data = judge_output
                                if "scores" not in judge_data and isinstance(parsed.get("judge_scores"), dict):
                                    judge_data["scores"] = parsed.get("judge_scores", {})
                            else:
                                if isinstance(parsed.get("judge_scores"), dict):
                                    judge_data = {"scores": parsed.get("judge_scores", {})}
                                elif isinstance(parsed.get("scores"), dict):
                                    judge_data = {"scores": parsed.get("scores", {})}
                                else:
                                    judge_data = {}

                            flags = parsed.get("flags", []) if isinstance(parsed.get("flags"), list) else []
                            language = parsed.get("language", "en") if isinstance(parsed.get("language"), str) else "en"

                        except Exception:
                            parsed = None
                            judge_data = {}
                            flags = []
                            language = "en"

                        epoch_number = _extract_epoch_number(tf, parsed)

                        resp["transcripts"].append(
                            {
                                "path": tf,
                                "scenario_id": pack_id,
                                "epoch_number": epoch_number,
                                "content": content,
                                "judge_scores": judge_data,
                                "flags": flags,
                                "language": language,
                            }
                        )
                    except Exception as e:
                        resp["transcripts"].append(
                            {
                                "path": tf,
                                "scenario_id": pack_id,
                                "epoch_number": None,
                                "content": f"(error reading {tf}: {e})",
                                "judge_scores": {},
                                "flags": [],
                                "language": "en",
                            }
                        )

            # Best-effort token extraction
            if result.stdout:
                m = re.search(r"tokens?:\s*(\d+)", result.stdout, re.IGNORECASE)
                if m:
                    try:
                        resp["cost_tokens"] += int(m.group(1))
                    except Exception:
                        pass

        _send_callback(callback_url, resp, callback_secret)
        return resp

    except subprocess.TimeoutExpired:
        resp["status"] = "failed"
        resp["error_message"] = "PETRI execution timed out"
        _send_callback(callback_url, resp, callback_secret)
        return resp
    except Exception as e:
        resp["status"] = "failed"
        resp["error_message"] = str(e)
        _send_callback(callback_url, resp, callback_secret)
        return resp
