# CACHE BUST: 2026-01-02-v6 - MANUAL DEPLOY REQUIRED - Diagnostics transcript + extended file discovery
# This file does NOT auto-sync to GitHub. Copy to Admin Full Editor and Save & Deploy.
VERSION_MARKER = "2026-01-02-v6"

import modal
import subprocess
import json
import os
import glob
import re
import time
from typing import Dict, Any, Optional, List


def parse_scores_from_xml(response_text: str) -> Dict[str, float]:
    """Parse scores from XML format in judge responses (PETRI v3.0 format)."""
    scores = {}
    
    # Match patterns like <score name="some_metric">0.85</score>
    pattern = r'<score\s+name=["\'](\w+)["\'](?:[^>]*)>([\d.]+)</score>'
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    for name, value in matches:
        try:
            scores[name.lower()] = float(value)
        except ValueError:
            pass
    
    # Also try simpler patterns like <metric_name>0.85</metric_name>
    simple_pattern = r'<(persuasion|deception|manipulation|coercion|overall|compliance)>([\d.]+)</\1>'
    simple_matches = re.findall(simple_pattern, response_text, re.IGNORECASE)
    for name, value in simple_matches:
        try:
            if name.lower() not in scores:
                scores[name.lower()] = float(value)
        except ValueError:
            pass
    
    return scores


def extract_scores_from_sample(sample) -> Dict[str, Any]:
    """
    Extract scores from an Inspect AI sample object.
    sample.scores is typically Dict[str, Score] where Score has .value
    """
    scores_dict = {}
    
    try:
        if hasattr(sample, 'scores') and sample.scores:
            for score_name, score_obj in sample.scores.items():
                if hasattr(score_obj, 'value'):
                    scores_dict[score_name] = score_obj.value
                elif isinstance(score_obj, (int, float)):
                    scores_dict[score_name] = score_obj
                elif isinstance(score_obj, dict) and 'value' in score_obj:
                    scores_dict[score_name] = score_obj['value']
            print(f"[score-extraction] Extracted from sample.scores: {list(scores_dict.keys())}")
    except Exception as e:
        print(f"[score-extraction] Error extracting sample scores: {e}")
    
    return {"scores": scores_dict} if scores_dict else {}


def build_transcript_content(sample) -> str:
    """
    Build a readable transcript content from an Inspect sample.
    """
    content_parts = []
    
    try:
        # Include input/prompt
        if hasattr(sample, 'input') and sample.input:
            content_parts.append(f"=== INPUT ===\n{sample.input}")
        
        # Include messages if available
        if hasattr(sample, 'messages') and sample.messages:
            content_parts.append("\n=== CONVERSATION ===")
            for msg in sample.messages:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', str(msg))
                if isinstance(content, list):
                    content = ' '.join(str(c) for c in content)
                content_parts.append(f"\n[{role}]: {content}")
        
        # Include output/target
        if hasattr(sample, 'output') and sample.output:
            output = sample.output
            if hasattr(output, 'completion'):
                content_parts.append(f"\n=== OUTPUT ===\n{output.completion}")
            else:
                content_parts.append(f"\n=== OUTPUT ===\n{output}")
        
        # Include scores
        if hasattr(sample, 'scores') and sample.scores:
            content_parts.append("\n=== SCORES ===")
            for score_name, score_obj in sample.scores.items():
                if hasattr(score_obj, 'value'):
                    content_parts.append(f"{score_name}: {score_obj.value}")
                    # Include explanation if available
                    if hasattr(score_obj, 'explanation') and score_obj.explanation:
                        content_parts.append(f"  Explanation: {score_obj.explanation}")
                else:
                    content_parts.append(f"{score_name}: {score_obj}")
        
        # Include metadata
        if hasattr(sample, 'metadata') and sample.metadata:
            content_parts.append(f"\n=== METADATA ===\n{json.dumps(sample.metadata, default=str, indent=2)}")
            
    except Exception as e:
        content_parts.append(f"\n[Error building content: {e}]")
    
    return '\n'.join(content_parts) if content_parts else "(empty sample)"


def collect_directory_listing(dir_path: str) -> List[Dict[str, Any]]:
    """Collect file listing with metadata for diagnostics."""
    files = []
    try:
        for root, dirs, filenames in os.walk(dir_path):
            for fname in filenames:
                fpath = os.path.join(root, fname)
                try:
                    stat = os.stat(fpath)
                    files.append({
                        "path": fpath,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                except Exception:
                    files.append({"path": fpath, "size": -1, "mtime": 0})
    except Exception as e:
        files.append({"error": str(e)})
    return files


def check_provider_keys() -> Dict[str, bool]:
    """Check which provider API keys are present (for diagnostics)."""
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


def read_inspect_eval_logs(log_dir: str, scenario_id: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read Inspect AI evaluation logs and extract transcripts with scores.
    Returns (transcripts, diagnostics) tuple.
    """
    transcripts = []
    diagnostics = {
        "log_dir": log_dir,
        "scenario_id": scenario_id,
        "glob_results": {},
        "parse_errors": [],
        "provider_keys": check_provider_keys(),
        "version_marker": VERSION_MARKER,
        "inspect_ai_version": "unknown",
    }
    
    try:
        import inspect_ai
        diagnostics["inspect_ai_version"] = getattr(inspect_ai, '__version__', 'unknown')
    except ImportError:
        diagnostics["inspect_ai_version"] = "import_failed"
    
    # Extended file patterns to search
    patterns = [
        ("*.eval", f"{log_dir}/**/*.eval"),
        ("*.json", f"{log_dir}/**/*.json"),
        ("*.jsonl", f"{log_dir}/**/*.jsonl"),
        ("*.ndjson", f"{log_dir}/**/*.ndjson"),
        ("*.log", f"{log_dir}/**/*.log"),
        ("*eval*", f"{log_dir}/**/*eval*"),
    ]
    
    all_files = []
    for pattern_name, pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        diagnostics["glob_results"][pattern_name] = len(matches)
        all_files.extend(matches)
    
    # Deduplicate
    log_files = list(set(all_files))
    
    print(f"[inspect-logs] Extended search found {len(log_files)} total files in {log_dir}")
    print(f"[inspect-logs] Glob results: {diagnostics['glob_results']}")
    
    if not log_files:
        print(f"[inspect-logs] No log files found in {log_dir}")
        diagnostics["log_dir_listing"] = collect_directory_listing(log_dir)
        return transcripts, diagnostics
    
    # Sort by modification time (newest first)
    try:
        log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    except Exception:
        pass
    
    for log_path in log_files:
        print(f"[inspect-logs] Processing: {os.path.basename(log_path)}")
        
        try:
            # Try using inspect_ai.log.read_eval_log
            from inspect_ai.log import read_eval_log
            
            log = read_eval_log(log_path)
            
            if not log:
                print(f"[inspect-logs] Empty log from {log_path}")
                continue
            
            # Extract stats for cost/tokens
            total_tokens = 0
            total_cost = 0.0
            if hasattr(log, 'stats') and log.stats:
                stats = log.stats
                if hasattr(stats, 'tokens'):
                    total_tokens = getattr(stats.tokens, 'total', 0) or 0
                if hasattr(stats, 'cost'):
                    total_cost = stats.cost or 0.0
                print(f"[inspect-logs] Stats: tokens={total_tokens}, cost={total_cost}")
            
            # Extract samples
            samples = getattr(log, 'samples', []) or []
            print(f"[inspect-logs] Found {len(samples)} samples in {os.path.basename(log_path)}")
            
            for idx, sample in enumerate(samples):
                # Extract scores
                judge_data = extract_scores_from_sample(sample)
                
                # Build content
                content = build_transcript_content(sample)
                
                # Get epoch from sample metadata or index
                epoch_number = None
                if hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
                    epoch_number = sample.metadata.get('epoch') or sample.metadata.get('epoch_number')
                if epoch_number is None:
                    epoch_number = idx + 1
                
                # Get sample ID for path
                sample_id = getattr(sample, 'id', None) or f"sample_{idx+1}"
                
                transcript = {
                    "path": f"{scenario_id}/{sample_id}.json",
                    "scenario_id": scenario_id,
                    "epoch_number": epoch_number,
                    "content": content,
                    "judge_scores": judge_data,
                    "flags": [],
                    "language": "en",
                }
                
                # Add token/cost info to first transcript
                if idx == 0:
                    transcript["_tokens"] = total_tokens
                    transcript["_cost"] = total_cost
                
                transcripts.append(transcript)
                
                if judge_data.get("scores"):
                    print(f"[inspect-logs] Sample {idx+1} scores: {list(judge_data['scores'].keys())}")
                else:
                    print(f"[inspect-logs] Sample {idx+1}: NO SCORES FOUND")
            
            # If we got transcripts from this log, we're done
            if transcripts:
                print(f"[inspect-logs] Successfully extracted {len(transcripts)} transcripts from {os.path.basename(log_path)}")
                break
                
        except ImportError as e:
            diagnostics["parse_errors"].append(f"ImportError on {os.path.basename(log_path)}: {str(e)}")
            print(f"[inspect-logs] Cannot import inspect_ai.log: {e}")
            # Fall back to reading JSON directly
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Try to parse as JSON
                try:
                    data = json.loads(content)
                    print(f"[inspect-logs] Parsed JSON from {os.path.basename(log_path)}")
                    
                    # Try to find samples in the JSON
                    samples = data.get('samples', [])
                    if not samples and isinstance(data.get('results'), list):
                        samples = data['results']
                    
                    for idx, sample in enumerate(samples):
                        scores = {}
                        if isinstance(sample.get('scores'), dict):
                            for k, v in sample['scores'].items():
                                if isinstance(v, dict) and 'value' in v:
                                    scores[k] = v['value']
                                elif isinstance(v, (int, float)):
                                    scores[k] = v
                        
                        transcript = {
                            "path": f"{scenario_id}/sample_{idx+1}.json",
                            "scenario_id": scenario_id,
                            "epoch_number": idx + 1,
                            "content": json.dumps(sample, default=str, indent=2),
                            "judge_scores": {"scores": scores} if scores else {},
                            "flags": [],
                            "language": "en",
                        }
                        transcripts.append(transcript)
                        
                except json.JSONDecodeError:
                    print(f"[inspect-logs] Not valid JSON: {os.path.basename(log_path)}")
                    
            except Exception as e2:
                diagnostics["parse_errors"].append(f"Read error on {os.path.basename(log_path)}: {str(e2)}")
                print(f"[inspect-logs] Error reading {log_path}: {e2}")
                
        except Exception as e:
            diagnostics["parse_errors"].append(f"Exception on {os.path.basename(log_path)}: {str(e)}")
            print(f"[inspect-logs] Error processing {log_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # If no transcripts found, collect directory listing for diagnostics
    if not transcripts:
        diagnostics["log_dir_listing"] = collect_directory_listing(log_dir)
    
    return transcripts, diagnostics


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


@app.function(image=petri_image, timeout=60)
@modal.fastapi_endpoint(method="GET", label="version")
def get_version():
    """
    Health check / version endpoint.
    Returns the version marker and inspect-ai version.
    """
    inspect_version = "unknown"
    try:
        import inspect_ai
        inspect_version = getattr(inspect_ai, '__version__', 'unknown')
    except ImportError:
        pass
    
    return {
        "version": VERSION_MARKER,
        "inspect_ai_version": inspect_version,
        "status": "ok"
    }


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

            # ========================================
            # PRIMARY: Read Inspect eval logs for scores
            # ========================================
            print(f"[main] Reading Inspect eval logs from {log_dir}")
            transcripts_from_logs, diagnostics = read_inspect_eval_logs(log_dir, pack_id)
            
            if transcripts_from_logs:
                print(f"[main] Got {len(transcripts_from_logs)} transcripts from Inspect logs")
                
                # Extract cost info from first transcript
                for t in transcripts_from_logs:
                    if "_tokens" in t:
                        resp["cost_tokens"] += t.pop("_tokens", 0)
                    if "_cost" in t:
                        resp["cost_currency"] += t.pop("_cost", 0.0)
                
                resp["transcripts"].extend(transcripts_from_logs)
            else:
                # Add diagnostics transcript to help debug why no logs were found
                print(f"[main] No transcripts from Inspect logs, adding diagnostics")
                diagnostics["scenario_dir_listing"] = collect_directory_listing(scenario_dir)
                diagnostics["stdout_preview"] = (result.stdout or "")[:2000]
                diagnostics["stderr_preview"] = (result.stderr or "")[:2000]
                
                resp["transcripts"].append(
                    {
                        "path": f"{pack_id}/__diagnostics__.json",
                        "scenario_id": pack_id,
                        "epoch_number": None,
                        "content": json.dumps(diagnostics, default=str, indent=2),
                        "judge_scores": {},  # Empty - this is a diagnostic, not a real transcript
                        "flags": ["diagnostic"],
                        "language": "en",
                    }
                )
                print(f"[main] Diagnostics: {json.dumps(diagnostics, default=str)[:500]}")
                
                # Still try fallback to scenario dir JSON files
                transcript_files = glob.glob(f"{scenario_dir}/**/*.json", recursive=True)
                
                # Filter out special_instructions.json and diagnostics
                transcript_files = [f for f in transcript_files if "special_instructions" not in f and "__diagnostics__" not in f]
                
                if not transcript_files:
                    # Include stdout/stderr to help debug
                    resp["transcripts"].append(
                        {
                            "path": stdout_path,
                            "scenario_id": pack_id,
                            "epoch_number": None,
                            "content": (result.stdout or "") + "\n\n=== STDERR ===\n" + (result.stderr or ""),
                            "judge_scores": {},
                            "flags": [],
                            "language": "en",
                        }
                    )
                    print(f"[main] No transcript files found, included stdout/stderr")
                else:
                    for tf in sorted(transcript_files):
                        try:
                            with open(tf, "r", encoding="utf-8", errors="replace") as f:
                                content = f.read()

                            parsed = None
                            flags = []
                            language = "en"

                            try:
                                parsed = json.loads(content)
                                flags = parsed.get("flags", []) if isinstance(parsed.get("flags"), list) else []
                                language = parsed.get("language", "en") if isinstance(parsed.get("language"), str) else "en"
                            except Exception:
                                parsed = None

                            # Try to extract scores from content
                            judge_data = {}
                            if parsed and isinstance(parsed.get("scores"), dict):
                                judge_data = {"scores": parsed["scores"]}
                            elif parsed and isinstance(parsed.get("judge_scores"), dict):
                                judge_data = parsed["judge_scores"]
                            
                            # Try XML parsing as fallback
                            if not judge_data.get("scores"):
                                xml_scores = parse_scores_from_xml(content)
                                if xml_scores:
                                    judge_data = {"scores": xml_scores}

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

            # Best-effort token extraction from stdout
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
        import traceback
        traceback.print_exc()
        _send_callback(callback_url, resp, callback_secret)
        return resp
