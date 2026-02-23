import argparse
import json
import asyncio
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


import torch
from faster_whisper import WhisperModel
from shazamio import Shazam
from deezer import Client as DeezerClient
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
PROCESS_FILE = "process.json"
LYRICS_CACHE_FILE = "deezer_lyrics_cache.json"
LANGUAGE = "en"

# ---------------- UTILS ----------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# ---------------- AUDIO ----------------
def extract_segment(path: Path, start=None, duration=None) -> Path | None:
    tmp = path.with_suffix(f".{int(start or 0)}.wav")
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-ac", "1", "-ar", "44100", str(tmp)]
    if run(cmd).returncode != 0:
        return None
    return tmp

# ---------------- SHAZAM ----------------
async def shazam_detect(path: Path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(str(path))
        track = out.get("track")
        if not track:
            return None
        return {"title": track.get("title"), "artist": track.get("subtitle")}
    except Exception:
        return None

async def fingerprint_track(audio: Path, seconds: float):
    tmp = extract_segment(audio, 0, seconds)
    if not tmp:
        return None
    res = await shazam_detect(tmp)
    tmp.unlink(missing_ok=True)
    return res

# ---------------- WHISPER ----------------
def detect_first_word(model, path: Path, max_seconds=30) -> float | None:
    tmp = extract_segment(path, 0, max_seconds)
    if not tmp:
        return None
    segments, _ = model.transcribe(str(tmp), language=LANGUAGE, word_timestamps=True)
    tmp.unlink(missing_ok=True)
    for seg in segments:
        if seg.words:
            return seg.words[0].start
    return None

def transcribe_from_start(model, path: Path, start: float, seconds: float) -> str:
    tmp = extract_segment(path, start, seconds)
    if not tmp:
        return ""
    segments, _ = model.transcribe(str(tmp), language=LANGUAGE)
    text = " ".join(s.text.strip() for s in segments)
    tmp.unlink(missing_ok=True)
    return text.strip()

# ---------------- DEEZER ----------------
_deezer = DeezerClient()


def load_cache(base: Path) -> dict:
    p = base / LYRICS_CACHE_FILE
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_cache(base: Path, cache: dict):
    (base / LYRICS_CACHE_FILE).write_text(json.dumps(cache))


def fetch_lyrics(track_id: int, cache: dict) -> str:
    if str(track_id) in cache:
        return cache[str(track_id)]
    try:
        lyrics = _deezer.get_track_lyrics(track_id) or ""
        cache[str(track_id)] = lyrics
        return lyrics
    except Exception:
        return ""

def search_deezer_direct(query, limit=10):
    """Direct Deezer API call - always returns data if available"""
    url = "https://api.deezer.com/search"
    params = {
        'q': query,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract tracks from response
        tracks = data.get('data', [])
        
        if not tracks:
            print(f"No tracks found for: '{query}'")
            return []
        
        print(f"API returned {len(tracks)} tracks")
        return tracks
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []


def deezer_search_parallel(text: str, cache: dict, limit=1):
    # results = _deezer.search(text)
    # log(results)
    results = search_deezer_direct(text,limit)
    out = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(fetch_lyrics, r['id'], cache): r for r in results
        }
        for fut, r in futures.items():
            log(fut)
            lyrics = fut.result()
            if lyrics:
                out.append({"title": r.title, "artist": r.artist.name, "lyrics": lyrics})
    return out



# ---------------- SCORING ----------------
def score_candidates(source: str, candidates: list):
    if not source or not candidates:
        return []

    fuzz = PolyFuzz(TFIDF())
    fuzz.match([source], [c["lyrics"] for c in candidates])
    similarity = fuzz.get_matches().loc[0, "Similarity"] * 100

    scored = []
    for c in candidates:
        title_hit = c["title"].lower() in source.lower()
        artist_hit = c["artist"].lower() in source.lower()
        bonus = (10 if title_hit else 0) + (5 if artist_hit else 0)
        scored.append({
            "title": c["title"],
            "artist": c["artist"],
            "similarity": round(similarity, 2),
            "bonus": bonus,
            "final_score": round(similarity + bonus, 2),
            "title_hit": title_hit,
            "artist_hit": artist_hit
        })
    return scored

# ---------------- TAGGING ----------------
def tag_file(path: Path, title: str, artist: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    r = run([
        "ffmpeg", "-y", "-i", str(path),
        "-metadata", f"title={title}",
        # "-metadata", f"artist={artist}",
        "-codec", "copy", str(tmp)
    ])
    if r.returncode == 0:
        tmp.replace(path)
        return True
    tmp.unlink(missing_ok=True)
    return False

# ---------------- MAIN ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--shazam-seconds", type=float, default=10)
    p.add_argument("--deezer-seconds", type=float, default=10)
    p.add_argument("--force-clean", action="store_true")
    args = p.parse_args()

    directory = Path(args.dir).resolve()
    process_path = directory / PROCESS_FILE

    if args.force_clean and process_path.exists():
        process_path.unlink()

    data = json.loads(process_path.read_text()) if process_path.exists() else {}
    cache = load_cache(directory)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper = WhisperModel("base", device=device, compute_type="float16" if device == "cuda" else "int8")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for f in directory.iterdir():
        log(f"Processing {f.name}")

        if f.suffix.lower() not in AUDIO_EXTENSIONS:
            # log(f"System: skipping {f.name}, file not supported")
            continue
        if data.get(f.name, {}).get("status") == "done":
            log(f"System: skipping {f.name}, already processed")
            continue
        
        log(f"System: Shazam...")
        shazam_res = loop.run_until_complete(fingerprint_track(f, args.shazam_seconds))
        if shazam_res:
            tag_file(f, shazam_res["title"], shazam_res["artist"])
            data[f.name] = {"track": {"title": shazam_res['title'], "artist": shazam_res['artist'] },"method": "shazam", "status": "done"}
            process_path.write_text(json.dumps(data, indent=2))
            continue

        log(f"System: Shazam -> Whisper+Deezer...")
        start = detect_first_word(whisper, f)
        if start is None:
            data[f.name] = {"status": "failed"}
            log(f"System: Whispher first word detection failed, try increading the detection time.")
            continue

        log(f"System: Whisper...")
        lyrics = transcribe_from_start(whisper, f, start, args.deezer_seconds)
        log(f"System: Deezer...")
        candidates = deezer_search_parallel(lyrics, cache)
        log(candidates)
        data[f.name] = {"deezer_candidates":candidates}
        log(f"System: Processing scores")
        scores = score_candidates(lyrics, candidates)

        if scores:
            best = max(scores, key=lambda x: x["final_score"])
            threshold = 90 if not (best["title_hit"] or best["artist_hit"]) else 80
            if best["final_score"] >= threshold:
                log(f"System: Applying metadata")
                tag_file(f, best["title"])
                data[f.name] = {
                  "track": {
                    "title": best["title"],
                    "artist": best["artist"]
                  },
                }
                status = "done"

            else:
                log(f"System: Scoring did not match criteria, seting to manual review.")
                status = "review"
        else:
            log(f"System: Failed scoring")
            status = "failed"

        data[f.name] = {
            "method": "deezer",
            "lyrics": lyrics,
            "scores": scores,
            "status": status
        }

        process_path.write_text(json.dumps(data, indent=2))
        save_cache(directory, cache)

    log("All processing complete")

if __name__ == "__main__":
    main()
