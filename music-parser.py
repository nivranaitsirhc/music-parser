# Smart Music Title Parser ver 3

import argparse
import json
import asyncio
import subprocess
import time
import re
import math
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Iterable,  Callable, Awaitable, Optional

from lrclib import AsyncLRCLibClient
from deezerlib import AsyncDeezerClient
from genius_api import GeniusSearch
from genius_driver import fetch_genius_lyrics

from search_engines import Aol
import torch
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.fuzz import partial_ratio
from faster_whisper import WhisperModel
from shazamio import Shazam
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
from lrcup import LRCLib
import numpy as np


# Config
# ------------------------------------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
PROCESS_FILE = "process.json"
LANGUAGE = "en"


# Helpers
# ------------------------------------------------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def tag_file(path: Path, title: str, artist: str):
    tmp = path.with_suffix(".tmp." + path.suffix)
    r = run([
        "ffmpeg", "-y", "-i", str(path),
        "-metadata", f"title={title}",
        "-codec", "copy", str(tmp)
    ])
    
    if r.returncode != 0:
        print(f"Error running ffmpeg on {path}:")
        print("STDERR:", r.stderr)
        print("STDOUT:", r.stdout)
        tmp.unlink(missing_ok=True)
        return False
    
    tmp.replace(path)
    return True

def is_list_of_dicts(variable):
    # First, check if the variable is a list
    if not isinstance(variable, list):
        return False
    
    # Second, check if every item within the list is a dictionary
    # The all() function ensures all elements satisfy the condition
    if not all(isinstance(item, dict) for item in variable):
        return False
        
    # If both conditions are met, it is a list of dictionaries
    return True


# extract audio segments
def extract_segment(path: Path, start=None, duration=None) -> Path | None:
    tmp = path.with_suffix(f".{start or 'full'}.wav")
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-ac", "1", "-ar", "44100", str(tmp)]

    r = run(cmd)
    if r.returncode != 0:
        return None
    return tmp

def get_duration(path: Path) -> float:
    r = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0

def save_to_process(data, filepath, indent=2, max_compact=10):
    # First, generate standard JSON
    json_str = json.dumps(data, indent=indent, sort_keys=False)
    
    # Post-process to compact small lists
    lines = json_str.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for array start
        if line.rstrip().endswith('['):
            # Find the matching closing bracket
            start_idx = i
            bracket_count = 1
            i += 1
            
            while i < len(lines) and bracket_count > 0:
                bracket_count += lines[i].count('[')
                bracket_count -= lines[i].count(']')
                i += 1
            
            # Get all array lines
            array_lines = lines[start_idx:i]
            
            # Count actual data lines (not brackets or whitespace)
            data_lines = [
                l for l in array_lines[1:-1]  # Exclude opening/closing
                if l.strip() and not l.strip() in ['[', ']']
            ]
            
            if len(data_lines) <= max_compact:
                # Compact the array
                indent_str = array_lines[0][:len(array_lines[0]) - len(array_lines[0].lstrip())]
                compact = indent_str + ' '.join(l.strip() for l in array_lines)
                result.append(compact)
            else:
                result.extend(array_lines)
            
            continue
        
        result.append(line)
        i += 1
    
    Path(filepath).write_text('\n'.join(result) + '\n', encoding='utf-8')

def save_update(updates, track_data, process_data, process_path):
    """Update track data and save to process file"""
    track_data.update(updates)
    save_to_process(process_data, process_path)


def update_candidates_with_keys(
    main_list: List[Dict[str, Any]],
    partial_list: List[Dict[str, Any]],
    keys: Iterable[str],
) -> List[Dict[str, Any]]:
    """
    Merge items from partial_list into main_list by matching one or more keys.

    Args:
        main_list: Base list of dictionaries.
        partial_list: List containing partial updates.
        keys: Key or keys used to match items (e.g. ["id"], ["id", "name"]).

    Returns:
        New merged list of dictionaries.
    """

    keys = tuple(keys)

    def make_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(item.get(k) for k in keys)

    partial_map = {make_key(item): item for item in partial_list}

    return [
        {**item, **partial_map.get(make_key(item), {})}
        for item in main_list
    ]

# Method Shared
# ------------------------------------------------------------------------------------------------------
async def lrclib_fetch_lyrics(
    track: str,
    artist: str,
    album: Optional[str] = None,
    duration: Optional[int] = None
) -> Optional[str]:

    client = AsyncLRCLibClient(max_retries=3, ssl_mode="httpx")

    lyrics = await client.get(track, artist, album, duration)

    if lyrics and lyrics.get("plainLyrics"):
        return " ".join(lyrics["plainLyrics"].split())

    return ""

async def deezer_fetch_tracks(query:str)->List[Dict]:
    client = AsyncDeezerClient(max_retries=3)
    
    data = await client.search(query,limit=2)

    results = []
    for d in data.get("data", []):
        results.append({
            "track_id": d.get("id"),
            "title": d.get("title_short") if "title_short" in d else d.get("title",""),
            "artist": d.get("artist", {}).get("name",""),
            "album": d.get("album",{}).get("title",""),
            "duration": d.get("duration"),
            "isrc": d.get("isrc"),
            "search_kword":query
        })
    
    return results

async def fetch_and_flatten_deezer_candidates(
    queries: List[str],
    deezer_fetch_tracks: Callable[[str], Awaitable[List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """
    Fetch Deezer results for multiple queries concurrently and flatten the results.

    Args:
        queries: List of search query strings.
        deezer_fetch_tracks: Async function that fetches Deezer tracks for a single query.

    Returns:
        A single flat list of track dictionaries with duplicates removed (by track_id).
    """

    if not queries:
        return []

    # Run all queries concurrently
    tasks = [deezer_fetch_tracks(q) for q in queries]
    results_nested = await asyncio.gather(*tasks)  # List[List[Dict]]

    # Flatten list
    all_results: List[Dict[str, Any]] = [track for sublist in results_nested for track in sublist]

    # # Optional: remove duplicates by track_id
    # seen_ids = set()
    # unique_results: List[Dict[str, Any]] = []
    # for track in all_results:
    #     tid = track.get("track_id")
    #     if tid and tid not in seen_ids:
    #         unique_results.append(track)
    #         seen_ids.add(tid)

    # return unique_results
    return all_results


async def fetch_candidates_lyrics(
    candidates: List[Dict[str, Any]],
    lrclib_fetch_lyrics: Optional[
        Callable[[str, str, Optional[str], Optional[int]], Awaitable[Optional[str]]]
    ] = None,
) -> List[Dict[str, Any]]:
    """
    Concurrently fetch and attach lyrics for candidate tracks.

    Required fields per candidate:
        - title: str
        - artist: str

    Optional fields:
        - album: str | None
        - duration: int | None

    This function does NOT mutate input. It returns a new enriched list.

    Args:
        candidates:
            List of track dictionaries.

        lrclib_fetch_lyrics:
            Optional async function with signature:
                (title, artist, album?, duration?) -> lyrics

    Returns:
        A new list of candidate dictionaries enriched with a "lyrics" field.
    """

    if not candidates:
        return []

    if lrclib_fetch_lyrics is None:
        return [dict(c) for c in candidates]

    async def fetch(candidate: Dict[str, Any]) -> Dict[str, Any]:
        title = candidate.get("title")
        artist = candidate.get("artist")

        if not title or not artist:
            return {**candidate, "lyrics": ""}

        album = candidate.get("album")
        duration = candidate.get("duration")

        lyrics = await lrclib_fetch_lyrics(
            title,
            artist,
            album if isinstance(album, str) else None,
            int(duration) if isinstance(duration, (int, float)) else None,
        )

        return {**candidate, "lyrics": lyrics}

    tasks = [fetch(c) for c in candidates]
    return await asyncio.gather(*tasks)

async def genius_get_track_data(
    queries: List[Dict[str, Any]],
    fetch_fn: Optional[Callable[[str], Awaitable[dict]]] = None,
    *,
    keyword_keys: Iterable[str] = ("title",),
    joiner: str = " ",
    concurrency: int = 10,
) -> List[dict]:
    """
    Concurrently fetch Genius track data using multiple keys
    from each query dict to construct the search keyword.

    Args:
        queries: List of dicts containing search metadata.
        fetch_fn: Async function (ex: fetch_genius_lyrics).
        keyword_keys: Iterable of keys to extract & combine as query.
        joiner: String used to join extracted keywords.
        concurrency: Max concurrent tasks.

    Returns:
        List of results (dict), preserving input order.
        Errors are returned as: []
    """
    if not queries:
        return []

    if fetch_fn is None:
        return [dict(c) for c in queries]

    sem = asyncio.Semaphore(concurrency)

    def build_query(item: Dict[str, Any]) -> str:
        parts = [
            str(item.get(k, "")).strip()
            for k in keyword_keys
            if item.get(k)
        ]
        return joiner.join(parts)

    async def runner(item: Dict[str, Any]) -> dict:
        keyword = build_query(item)

        if not keyword:
            return {**item}

        async with sem:
            try:
                results = await fetch_fn(keyword) 
                # get the first result, get the lyrics and merge back to the item
                return {**item, "lyrics": results[0]["lyrics"]}
            except Exception as e:
                return {**item}

    tasks = [runner(item) for item in queries]
    return await asyncio.gather(*tasks)

def boost_confidence_by_occurrence(
    input_list: List[Dict[str, Any]],
    key_fields: Optional[List[str]] = None,
    boost_percent: float = 3.0
) -> List[Dict[str, Any]]:
    """
    Merge duplicate track entries and boost confidence using probabilistic asymptotic scaling.

    This function groups tracks by a composite identity key (default: title + artist),
    merges duplicates, and boosts confidence based on the number of occurrences using:

        boosted = 1 - (1 - base) ^ occurrences

    followed by a mild linear gain multiplier:

        boosted *= (1 + boost_percent / 100)

    This ensures:
      - Diminishing returns (asymptotic growth toward 100%)
      - Stability against large duplicate counts
      - Prevention of runaway confidence inflation

    The function automatically detects if boosting has already been applied by checking
    for duplicate identity keys. If no duplicates are detected, the input is returned
    unchanged.

    Args:
        input_list:
            List of track dictionaries. Each item must contain at least:
            - "confidence" (float, 0–100)
            - Fields defined in `key_fields`

        key_fields:
            Fields used to uniquely identify tracks.
            Default: ["title", "artist"]

        boost_percent:
            Additional gain multiplier applied after probabilistic merge.
            Recommended range: 1.0 – 5.0

    Returns:
        A deduplicated list of tracks with boosted confidence and occurrence count.
        Sorted by descending confidence.
    """

    if not input_list:
        return []

    if key_fields is None:
        key_fields = ["title", "artist"]

    grouped: Dict[tuple, Dict[str, Any]] = {}
    duplicate_detected = False

    for item in input_list:
        key = tuple(item.get(k) for k in key_fields)
        if not all(key):
            continue

        confidence = float(item.get("confidence", 0.0))

        if key not in grouped:
            grouped[key] = {
                **item,
                "confidence": confidence,
                "occurrences": max(1, int(item.get("occurrences", 1))),
                "_base_confidence": confidence / 100.0,
            }
        else:
            duplicate_detected = True
            grouped[key]["occurrences"] += max(1, int(item.get("occurrences", 1)))

            # Merge segments if present
            if "segments" in grouped[key] or "segments" in item:
                grouped[key]["segments"] = [
                    *grouped[key].get("segments", []),
                    *item.get("segments", [])
                ]

    # If no duplicates exist, assume boosting already applied
    if not duplicate_detected:
        return sorted(input_list, key=lambda x: -x.get("confidence", 0.0))

    results: List[Dict[str, Any]] = []

    for data in grouped.values():
        base = data["_base_confidence"]
        occurrences = data["occurrences"]

        # Probabilistic asymptotic merge
        merged = 1.0 - (1.0 - base) ** occurrences

        # Gentle linear gain boost
        merged *= (1.0 + boost_percent / 100.0)

        boosted_confidence = min(merged * 100.0, 100.0)

        data["confidence"] = round(boosted_confidence, 4)
        data.pop("_base_confidence", None)

        results.append(data)

    return sorted(results, key=lambda x: -x.get("confidence", 0.0))

def whisper_detect_first_word(model, path: Path, max_seconds=30) -> float | None:
    tmp = extract_segment(path, 0, max_seconds)
    if not tmp:
        return None
    segments, _ = model.transcribe(str(tmp), language=LANGUAGE, word_timestamps=True)
    tmp.unlink(missing_ok=True)
    for seg in segments:
        if seg.words:
            return seg.words[0].start
    return None

def whisper_transcribe_from_start(model, path: Path, start: float, seconds: float) -> str:
    tmp = extract_segment(path, start, seconds)
    if not tmp:
        return ""
    segments, _ = model.transcribe(str(tmp), language=LANGUAGE)
    tmp.unlink(missing_ok=True)
    text = " ".join(s.text.strip() for s in segments)
    return text.strip()

def split_lyrics_to_chunks(long_string: str, max_words: int, max_chunks: int =3) -> list[str]:
    pattern = r'(?:\s*,\s*|\s*\n\s*|\s+and\s+)'
    parts = re.split(pattern, long_string)
    words = [w for part in parts for w in part.strip().split()]
    results = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return results[:max_chunks]

def score_candidates_by_lyrics_match(
    lyric_sample: str,
    song_list: List[Dict[str, Any]],
    *,
    tfidf_weight: float = 0.90,
    partial_weight: float = 0.90,
    jaccard_weight: float = 0.90
) -> List[Dict[str, Any]]:
    """
    Score each track in `song_list` against `lyric_sample` using a hybrid
    matching strategy: TF-IDF + Jaccard + fuzzy partial matching.

    Each track is enriched with a new property:

        track["match_scores"] = {
            "final": float,      # weighted ensemble score (0–1)
            "confidence": float, # final score expressed as percentage (0–100)
            "tfidf": float,      # TF-IDF cosine similarity
            "jaccard": float,    # Jaccard similarity
            "partial": float     # fuzzy partial match score
        }

    Args:
        lyric_sample: Short lyric segment to match.
        song_list: List of track objects. Each must contain a "lyrics" property.
        tfidf_weight: Weight assigned to TF-IDF similarity.
        partial_weight: Weight assigned to fuzzy partial similarity.
        jaccard_weight: Weight assigned to Jaccard similarity.

    Returns:
        Updated song_list with per-track scoring metadata.
    """

    if not lyric_sample or not song_list:
        return song_list

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

    def jaccard_similarity(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(len(sa | sb), 1)

    normalized_sample = normalize(lyric_sample)
    normalized_lyrics = [normalize(t.get("lyrics", "")) for t in song_list]

    if not any(normalized_lyrics):
        return song_list

    # ---------- TF-IDF similarity ----------
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    tfidf_matrix = vectorizer.fit_transform([normalized_sample] + normalized_lyrics)

    tfidf_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    ).flatten()

    # ---------- Per-track hybrid scoring ----------
    for idx, track in enumerate(song_list):
        lyrics_text = normalized_lyrics[idx]

        tfidf_score = float(tfidf_scores[idx])
        jaccard_score = jaccard_similarity(normalized_sample, lyrics_text)
        partial_score = partial_ratio(normalized_sample, lyrics_text) / 100.0

        final_score = min((
            partial_weight * partial_score +
            tfidf_weight * tfidf_score +
            jaccard_weight * jaccard_score
        ),100)

        track.update({
            "confidence": round(final_score * 100, 2),
            "lyrics_match_scoring_results": {
                "final": round(final_score, 4),
                "scores": {
                    "tfidf": round(tfidf_score, 4),
                    "jaccard": round(jaccard_score, 4),
                    "partial": round(partial_score, 4)
                }
            }
        })

    return song_list

# Method 1
# ------------------------------------------------------------------------------------------------------
async def shazam_detect(path: Path):
    shazam = Shazam()
    try:
        out = await shazam.recognize(str(path))
        track = out.get("track")
        if not track:
            return None
        return {
            "title": track.get("title"),
            "artist": track.get("subtitle"),
            "confidence": 95.0
        }
    except Exception:
        return None

async def fingerprint_track(audio: Path, seconds: float)->List[Dict]:
    duration = get_duration(audio)
    # segment track by seconds
    segments = [(i, min(i+seconds, duration)) for i in (seconds * j for j in range(int(duration // seconds) + 1)) if i < duration]
    
    results = []
    for start, dur in segments:
        tmp = extract_segment(audio, start, dur)
        if not tmp:
            continue
            
        result = await shazam_detect(tmp)
        tmp.unlink(missing_ok=True)
        
        if result:
            result.update({"segments":[start,dur]})
            results.append(result)
            # break when confidence is match
            if result.get("confidence", 0) >= 98.0:
                break
    
    return sorted(results, key=lambda x: -x.get("confidence", 0)) if results else []

def review_track_data(
    process_data: Dict[str, Dict[str, Any]],
    page_size: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Review all tracks with status == 'review'.

    Commands:
        [1-N]   → Select candidate
        <       → Previous page
        >       → Next page
        c|custom→ Manually enter title & artist
        Enter   → Skip track
        q       → Quit review

    Returns:
        Updated process_data dict.
    """

    def paginate(items: List[Dict[str, Any]], page: int, size: int):
        start = page * size
        end = start + size
        return items[start:end]

    review_items = [
        track for track in process_data.values()
        if track.get("status") == "review"
    ]

    if not review_items:
        print("✔️ No tracks pending review.")
        return process_data

    for track in review_items:
        path = Path(track.get("path", ""))
        candidates = track.get("candidates") or []

        total_pages = max(1, math.ceil(len(candidates) / page_size))
        page = 0

        while True:
            page_candidates = paginate(candidates, page, page_size)

            print("\n" + "=" * 70)
            print(f"Track Name   : {path.name}")
            print(f"Track Lyrics: {track.get('lyrics_sample', '')[:200]}")
            print(f"\nPage {page + 1} of {total_pages}\n")

            for i, c in enumerate(page_candidates, start=1):
                print(f"{i:02d} -> Confidence : {c.get('confidence', 0):.1f}")
                print(f"      Occurrences: {c.get('occurrences', 0)}")
                print(f"      Title      : {c.get('title', '')}")
                print(f"      Artist     : {c.get('artist', '')}")
                print(f"      Method     : {c.get('method', '')}")
                print(f"      Lyrics     : {(c.get('lyrics') or '')[:120]}")
                print(f"      Keyword    : {c.get('search_kword', '')}")
                print(f"      Segment    : {c.get('segments', '')}")
                print()

            cmd = input(
                "Select [1-N], < prev, > next, c custom, Enter skip, Q quit: "
            ).strip().lower()

            # skip track
            if not cmd:
                break

            # quit review session
            if cmd == "q":
                print("Exiting Review Process..")
                return process_data

            # pagination
            if cmd == "<":
                page = max(0, page - 1)
                continue

            if cmd == ">":
                page = min(total_pages - 1, page + 1)
                continue

            # manual override
            if cmd in {"c", "custom"}:
                print("\nManual Tagging:")
                title = input("  Enter Title : ").strip()
                artist = input("  Enter Artist: ").strip()

                if not title:
                    print("⚠️ Title and artist are required.")
                    continue

                tag_file(path, title, artist)

                track.update({
                    "status": "done",
                    "title": title,
                    "artist": artist,
                    "method": "manual",
                    "status_message": "User manual override",
                })

                print(f"✔️ Tagged (manual): {title} – {artist}")
                break

            # candidate selection
            if cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(page_candidates):
                    chosen = page_candidates[idx]

                    title = chosen.get("title") or ""
                    artist = chosen.get("artist") or ""

                    if not title:
                        print("⚠️ Invalid selection (missing title or artist).")
                        continue

                    tag_file(path, title, artist)

                    track.update({
                        "status": "done",
                        "title": title,
                        "artist": artist,
                        "method": chosen.get("method", ""),
                        "status_message": "User reviewed",
                    })

                    print(f"✔️ Tagged: {title} – {artist}")
                    break

                print("⚠️ Invalid selection index.")
                continue

            print("⚠️ Unknown command.")

    return process_data

# Method 0
# ------------------------------------------------------------------------------------------------------
def search_track_by_lyrcs(sample_lyrics:str) -> Dict[str, Any] | None:
    try:
        if not sample_lyrics:
            return {}
        
        # define search engine
        engine = Aol()
        results = engine.search(f"site: genius.com {sample_lyrics}",1)

        if not results:
            return {}
        
        result = results[0]["title"]

        # extract the title and artist
        artist, title = re.split(r' [–-] ', re.sub(r' Lyrics(?: [–-] Genius| \| Genius Lyrics)?$', '', result))[0:2]

        # fetch the lyrics
        genius_track = asyncio.run(fetch_genius_lyrics(f"{title} {artist}"))

        track = {
            "title": title,
            "artist": artist,
            "lyrics": genius_track["lyrics"],
        }
        
        return track
    except:
        return {}

# Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--target-filename", type=str, default=None)
    p.add_argument("--duration-sample-fingerprint", type=float, default=30)
    p.add_argument("--duration-sample-lyrics", type=float, default=60)
    p.add_argument("--duration-first-word", type=float, default=60)
    p.add_argument("--force-clean", action="store_true")
    p.add_argument("--reset-status",action="store_true")
    p.add_argument("--reset-candidates-fingerprint",action="store_true")
    p.add_argument("--reset-candidates-lyrics",action="store_true")
    p.add_argument("--reset-track-lyrics",action="store_true")
    p.add_argument("--reset-track-lyrics-chunks",action="store_true")
    p.add_argument("--review", action="store_true")
    args = p.parse_args()

    # Preprocess Initialization
    # --------------------------------------------------------------------------------------------------
    
    # process.json
    directory = Path(args.dir).resolve()
    # process file
    process_path = directory / PROCESS_FILE
    # list of tracks path
    process_lists = directory.iterdir()

    # default Lyrics Scoring Data Structure
    default_lyrics_scoring = {
        "final": 0.0,
        "scores": {
            "jaccard": [0.0, 0.90],
            "partial": [0.0, 0.90],
            "tfidf": [0.0, 0.90]
        }
    }

    # default Candidate Data Structure
    default_candidate = {
        "confidence":0,
        "occurrence":1,
        "title": "",
        "artist": "",
        "album" : "",
        "lyrics": "",
        "lyrics_match_scoring_results": default_lyrics_scoring,
    }

    # default Method 1 Candidates Data Structure
    default_candidate_method1 = {
        **default_candidate,
        "segments": []
    }
    
    # default Method 2 Candidates Data Structure
    default_candidate_method2 = {
        **default_candidate,
        "track_id": 0,
        "isrc": "",
        "duration":""
    }

    # default Candidates Data Structure
    default_candidates = {
        "title":"",
        "confidence": 0.0,
        "lyrics":"",
        "occurrences":1,
        "scores":{
            "method1": 0,
            "method2": 0,
        },
        "method1": [],
        "method2": []
    }


    # default Track Data Structure
    default_TrackData = {
        "path":"",
        "title":"",
        "artist":"",
        "status":"",
        "status_message":"",
        "method":"",
        "fingerprint_samples_duration": args.duration_sample_fingerprint,
        "lyrics_sample":"",
        "lyrics_sample_chunks":[],
        "lyrics_sample_duration": args.duration_sample_lyrics,
        "lyrics_1st_word":0.0,
        "candidates":[],
        "candidates_method0":[],
        "candidates_method1":[],
        "candidates_method2":[]
    }

    # clean up process.json
    if args.force_clean and process_path.exists():
        process_path.unlink(missing_ok=True) # delete the file


    # get data from process.json
    process_data = json.loads(content) if process_path.exists() and (content := process_path.read_text()) else {}

    # ensure that process data is of type dict
    if not isinstance(process_data, dict):
        process_data = {}

    # process only one file
    if args.target_filename:
        process_lists = [x for x in process_lists if x.name == args.target_filename]

    # revew
    marked_for_review = False
    if args.review:
        if isinstance(process_data,dict) and process_data:
            save_to_process(
                review_track_data(
                    process_data
                ),
                process_path
            )
            return None
        else:
            marked_for_review = True


    # data manipulation
    updates = {}

    if args.reset_status:
        updates['status'] = ""

    if args.reset_candidates_fingerprint:
        updates['candidates_method1'] = []

    if args.reset_candidates_lyrics:
        updates['candidates_lyrics'] = []

    # Apply updates if any exist
    if updates:
        process_data = {track: {**track_data, **updates} 
                    for track, track_data in process_data.items()}

    # Handle lyrics separately (modifies in-place)
    if args.reset_track_lyrics:
        for track_data in process_data.values():
            track_data.update({"lyrics_sample": "", "lyrics_sample_duration": 0})

    # Handle lyrics chunks
    if args.reset_track_lyrics_chunks:
        for track_data in process_data.values():
            track_data.update({"lyrics_sample_chunks":[]})

    # Save only if changes were made
    if updates or args.reset_track_lyrics:
        save_to_process(process_data, process_path)

    # init whisper model
    model = WhisperModel(
        "base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8"
    )

    # main loop
    for f in process_lists:
        # skip non audio files
        if f.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        log(f"Processing {f.name}..")

        # get the track entry from process_data if any
        track_item = process_data.get(f.name)

        if not track_item:
            track_item = dict(default_TrackData)
            track_item["path"] = str(f)
            process_data[f.name] = track_item
            save_to_process(process_data, process_path)

        # skip with done status
        if track_item.get("status") == "done":
            log(f"✔️ {f.name} -> already done! title:{track_item.get('title')}")
            continue
        
        # Method Init
        Method1_Error = ValueError
        Method2_Error = ValueError
        Method0_Error = ValueError
        candidates_method0 = track_item.get("candidates_method0",[])
        candidates_method1 = track_item.get("candidates_method1",[])
        candidates_method2 = track_item.get("candidates_method2",[])
        fingerprint_samples_duration = track_item.get("fingerprint_samples_duration",0)
        lyrics_sample = track_item.get("lyrics_sample","")
        lyrics_1st_word = track_item.get("lyrics_1st_word")
        lyrics_sample_chunks = track_item.get("lyrics_sample_chunks",[])
        lyrics_sample_duration = track_item.get("lyrics_sample_duration",0)

        # Method 0 : Lyrics Search using Search Engine + Webscraper
        try:
            log("Method 0 : Lyrics Search (Search Engine + Webscraper)")

            # Step 1
            if not lyrics_sample or lyrics_sample_duration != args.duration_sample_lyrics:
                # Step 0 - Use whisper determine the time of the 1st word of the track
                whisper_first_word: float

                if not lyrics_1st_word:
                    log("Method 0 -> Step 1A: Whisper analyzing track first word...")
                    detected = whisper_detect_first_word(model, f,args.duration_first_word)

                    if detected is None:
                        raise Method1_Error(
                            "Method 0 -> Step 1A: Failed to detect first word of track."
                        )

                    whisper_first_word = float(detected)

                else:
                    log("Method 0 -> Step 1A: Reusing cached track first word duration")
                    whisper_first_word = float(lyrics_1st_word)
                
                if not lyrics_sample or lyrics_sample_duration != args.duration_sample_lyrics:
                    log(f"Method 0 -> Step 1B: Whisper transcribing to generate track lyrics sample...")
                    
                    lyrics_sample_duration = args.duration_sample_lyrics

                    lyrics_sample = whisper_transcribe_from_start(model, f, whisper_first_word, lyrics_sample_duration)
                    
                    if not lyrics_sample:
                        raise Method1_Error("Method 0 -> Step 1B: Failed to transcribe lyrics")

                else:
                    log(f"Method 0 -> Step 1B: Reusing track lyrics sample...")

                save_update(
                    {
                        "lyrics_sample" : lyrics_sample,
                        "lyrics_1st_word": whisper_first_word,
                        "lyrics_sample_duration": lyrics_sample_duration
                    },
                    track_item, process_data, process_path
                )

            else:
                log(f"Method 0 -> Setp 1 : Reusing cached sample lyrics")


            # Step 2
            # Generate lyrics chunk from lyrics sample
            if not lyrics_sample_chunks:
                log(f"Method 0 -> Step 2 : Generating sample lyrics chunk..")
                # split lyrics to chuncks for deezer search
                lyrics_sample_chunks = split_lyrics_to_chunks(lyrics_sample,max_words=7,max_chunks=20)
                if not lyrics_sample_chunks:
                    raise Method2_Error("Method 0 -> Step 2 : Failed to generate sample lyrics chunk")
                
                save_update(
                    {
                        "lyrics_sample_chunks":lyrics_sample_chunks,
                    },
                    track_item,process_data,process_path
                )
            else:
                log(f"Method 0 -> Setp 2 : Reusing generated sample lyrics chunk")

            # Step 3 Identify the track
            track = search_track_by_lyrcs(lyrics_sample)
            
            if not track:
                raise Method0_Error("Method 0 -> Step 3 : Failed to identify the track")
            
            candidates_method0 =[track]

            # score candidates by lyrics match
            log(f"Method 0 -> Step 4 : Scoring by Lyrics Match..")
            candidates_method1 = score_candidates_by_lyrics_match(lyrics_sample, candidates_method0)

            # check if we have 99% match
            if candidates_method0[0].get("confidence",0.0) >= 99:
                # tag file
                success = tag_file(f, candidates_method0[0]["title"], candidates_method0[0]["artist"])
                
                if not success:
                    raise Method0_Error("Failed to tag file")

                # update process file
                save_update(
                    {
                        "title":candidates_method0[0]["title"],
                        "artist":candidates_method0[0]["artist"],
                        "status": "done",
                        "method": "Fingerprint",
                        "candidates_method1":candidates_method0
                    },
                    track_item,process_data,process_path
                )
                
                log(f"✔️ {f.name} -> title:{candidates_method0[0]['title']}")
                continue

        except Method0_Error as e:
            log(e)

        # Start Method 1
        try:
            log(f"Method 1 : Fingerprint")
            # Step 1 - Use shazam to identify track
            # load fingerprint sample, update only if empty or not the same
            if not candidates_method1 or fingerprint_samples_duration != args.duration_sample_fingerprint:
                log(f"Method 1 -> Step 1 : Shazam Analyzing Track..")
                candidates_method1_identify_track = asyncio.run(fingerprint_track(f, args.duration_sample_fingerprint))
                
                # check if method 1 is successfull
                if not candidates_method1_identify_track:
                    raise Method1_Error("Method 1 -> Step 1 : Failed to identify track, proceeding to Method 2..")
                
                candidates_method1 = candidates_method1_identify_track

                # update candidates to match default dict
                candidates_method1 = [{**default_candidate_method1, **track} for track in candidates_method1]
                
                save_update(
                    {
                        "candidates_method1":candidates_method1
                    },
                    track_item,process_data,process_path
                )
            else:
                log(f"Method 1 -> Step 1 : Skipping Shazam, Reusing cached data")
            

            # boost by re-occurance using title,artist properties
            candidates_method1=boost_confidence_by_occurrence(candidates_method1,["title","artist"],20)
            
            # check if we have 99% match
            if candidates_method1[0].get("confidence",0.0) >= 99:
                # tag file
                success=tag_file(f, candidates_method1[0]["title"], candidates_method1[0]["artist"])
                
                if not success:
                    raise Method1_Error ("Failed to write tag")

                # update process file
                save_update(
                    {
                        "title":candidates_method1[0]["title"],
                        "artist":candidates_method1[0]["artist"],
                        "status": "done",
                        "method": "Fingerprint",
                        "candidates_method1":candidates_method1
                    },
                    track_item,process_data,process_path
                )
                
                log(f"✔️ {f.name} -> title:{candidates_method1[0]['title']}")
                continue

            # Filter only candidates needing lyrics
            candidates_method1_needing_lyrics = [
                c for c in candidates_method1
                if ("lyrics" not in c or not c["lyrics"]) and 
                all(key in c for key in ["title", "artist"])
            ]

            # Fetch lyrics using Genius
            log(f"Method 1 -> Step 2 : Genius searching for shazam Lyrics")
            candidates_method1_with_lyrics = asyncio.run(genius_get_track_data(
                    candidates_method1_needing_lyrics,
                    fetch_genius_lyrics,
                    keyword_keys=("title", "artist")
                )
            )

            if not candidates_method1_with_lyrics:
                log("Method 1 -> Step 2 : Failed to match lyrics.")

            # Return back to candidates_method1
            candidates_method1 = candidates_method1_with_lyrics
            
            # ---------------------------------------------------------------------------
            # # Fetch Lyrics using LRClib
            # log(f"Method 1 -> Step 2 : LRCLib searching for shazam Lyrics...")

            # candidates_method1 = update_candidates_with_keys(
            #     candidates_method1,
            #     candidates_method1_with_lyrics,
            #     ["title","artist"]
            # )
            
            # # fetch lyrics
            # candidates_method1_with_lyrics = asyncio.run(fetch_candidates_lyrics(
            #         candidates_method1_needing_lyrics,
            #         lrclib_fetch_lyrics
            #     )
            # )

            # if not candidates_method1_with_lyrics:
            #     raise Method1_Error("Method 1 -> Step 2 : Failed to match lyrics.")

            # candidates_method1 = update_candidates_with_keys(
            #     candidates_method1,
            #     candidates_method1_with_lyrics,
            #     ["title","artist"]
            # )

            # ---------------------------------------------------------------------------

            save_update(
                {
                    "candidates_method1":candidates_method1
                },
                track_item,process_data,process_path
            )

            if not lyrics_sample or lyrics_sample_duration != args.duration_sample_lyrics:
                # Step 1 - Use whisper determine the time of the 1st word of the track
                whisper_first_word: float

                if not lyrics_1st_word:
                    log("Method 1 -> Step 3A: Whisper analyzing track first word...")
                    detected = whisper_detect_first_word(model, f,args.duration_first_word)

                    if detected is None:
                        raise Method1_Error(
                            "Method 1 -> Step 3A: Failed to detect first word of track."
                        )

                    whisper_first_word = float(detected)

                    save_update(
                        {
                            "lyrics_1st_word": whisper_first_word
                        },
                        track_item, process_data, process_path,
                    )
                else:
                    log("Method 1 -> Step 3A: Whisper reusing cached track first word duration")
                    whisper_first_word = float(lyrics_1st_word)
                    
                log(f"Method 1 -> Step 3B: Whisper transcribing to generate track lyrics sample...")
                lyrics_sample = whisper_transcribe_from_start(model, f, whisper_first_word, args.duration_sample_lyrics)

                if not lyrics_sample:
                    raise Method1_Error("Method 1 -> Step 3B: Failed to transcribe lyrics")

                # save lyrics_sample
                save_update(
                    {
                        "lyrics_sample":lyrics_sample
                    },
                    track_item,process_data,process_path
                )
            else:
                log(f"Method 1 -> Setp 3 : Whisper reusing cached sample lyrics")

            # score candidates by lyrics match
            log(f"Method 1 -> Step 4 : Scoring by Lyrics Match..")
            candidates_method1 = score_candidates_by_lyrics_match(lyrics_sample, candidates_method1)

            # check if we have 99% match
            if candidates_method1[0].get("confidence",0.0) >= 99:
                # tag file
                success = tag_file(f, candidates_method1[0]["title"], candidates_method1[0]["artist"])
                
                if not success:
                    raise Method1_Error("Failed to tag file")

                # update process file
                save_update(
                    {
                        "title":candidates_method1[0]["title"],
                        "artist":candidates_method1[0]["artist"],
                        "status": "done",
                        "method": "Fingerprint",
                        "candidates_method1":candidates_method1
                    },
                    track_item,process_data,process_path
                )
                
                log(f"✔️ {f.name} -> title:{candidates_method1[0]['title']}")
                continue

        except Method1_Error as e:
            log(e)

        finally:
            # cleanup method 1 before method 2
            save_update(
                {
                    "status":"review"
                },
                track_item,process_data,process_path
            )

        # Method 2 : Lyrics Search
        try:
            log(f"Method 2 : Lyrics Search")

            # Step 1
            if not lyrics_sample or lyrics_sample_duration != args.duration_sample_lyrics:
                # Step 1 - Use whisper determine the time of the 1st word of the track
                whisper_first_word: float

                if not lyrics_1st_word:
                    log("Method 2 -> Step 1A: Whisper analyzing track first word...")
                    detected = whisper_detect_first_word(model, f,args.duration_first_word)

                    if detected is None:
                        raise Method1_Error(
                            "Method 2 -> Step 1A: Failed to detect first word of track."
                        )

                    whisper_first_word = float(detected)

                    save_update(
                        {
                            "lyrics_1st_word": whisper_first_word
                        },
                        track_item, process_data, process_path,
                    )
                else:
                    log("Method 2 -> Step 1A: Whisper reusing cached track first word duration")
                    whisper_first_word = float(lyrics_1st_word)
                    
                log(f"Method 2 -> Step 1B: Whisper transcribing to generate track lyrics sample...")
                lyrics_sample = whisper_transcribe_from_start(model, f, whisper_first_word, args.duration_sample_lyrics)

                if not lyrics_sample:
                    raise Method1_Error("Method 2 -> Step 1B: Failed to transcribe lyrics")

                # save lyrics_sample
                save_update(
                    {
                        "lyrics_sample":lyrics_sample
                    },
                    track_item,process_data,process_path
                )
            else:
                log(f"Method 2 -> Setp 1 : Whisper reusing cached sample lyrics")


            # Step 2
            # Generate lyrics chunk from lyrics sample
            if not lyrics_sample_chunks:
                log(f"Method 2 -> Step 2 : Generating sample lyrics chunk..")
                # split lyrics to chuncks for deezer search
                lyrics_sample_chunks = split_lyrics_to_chunks(lyrics_sample,max_words=7,max_chunks=20)
                if not lyrics_sample_chunks:
                    raise Method2_Error("Method 2 -> Step 2 : Failed to generate sample lyrics chunk")
                
                save_update(
                    {
                        "lyrics_sample_chunks":lyrics_sample_chunks,
                    },
                    track_item,process_data,process_path
                )
            else:
                log(f"Method 2 -> Setp 2 : Reusing generated sample lyrics chunk")

            # Step 3
            # Use deezer to find tracks matching lyrics chunks
            if not candidates_method2:
                log(f"Method 2 -> Step 3 : Deezer searching for tracks matching the sample lyrics chunks...")
                candidates_method2_matching_tracks = asyncio.run(
                    fetch_and_flatten_deezer_candidates(
                      lyrics_sample_chunks,
                      deezer_fetch_tracks
                    )
                  )

                if not candidates_method2_matching_tracks:
                    raise Method2_Error("Method 2 -> 3 : Deezer failed to find Tracks.")

                candidates_method2 = candidates_method2_matching_tracks

                # update candidates to match default dict
                candidates_method2 = [{**default_candidate_method2, **track} for track in candidates_method2]
                
                save_update(
                    {
                        "candidates_method2":candidates_method2,
                    },
                    track_item,process_data,process_path
                )
                
            else:
                log(f"Method 2 -> Step 3 : Reusing Method 2 Candidates...")
            

            # Step 4
            log(f"Method 2 -> Step 4 : LRCLib searching for deezer Lyrics...")
            
            # combine similar keys and boost by re-occurance using track_id
            candidates_method2=boost_confidence_by_occurrence(candidates_method2,["title","artist"],20)

            # Find all candidates needing lyrics
            candidates_method2_needing_lyrics = [
                c for c in candidates_method2
                if ("lyrics" not in c  or not c["lyrics"]) and 
                all(key in c for key in ["title", "artist"])
            ]

            # fetch lyrics
            candidates_method2_with_lyrics = asyncio.run(fetch_candidates_lyrics(
                    candidates_method2_needing_lyrics,
                    lrclib_fetch_lyrics
                )
            )

            if not candidates_method2_with_lyrics:
                raise Method2_Error("Method 2 -> Step 5 : Failed to fetch lyrics")

            candidates_method2 = update_candidates_with_keys(
                candidates_method2,
                candidates_method2_with_lyrics,
                ["track_id"]
            )

            save_update(
                {
                    "candidates_method2" : candidates_method2
                },
                track_item,process_data,process_path
            )


            # score candidates by lyrics match
            log(f"Method 2 -> Step 5 : Scoring by Lyrics Match..")
            candidates_method2_with_lyrics_score_match = score_candidates_by_lyrics_match(lyrics_sample, candidates_method2)

            if not candidates_method2_with_lyrics_score_match:
                raise Method2_Error("Method 2 -> Step 5 : Failed to score candidates.")

            candidates_method2 = candidates_method2_with_lyrics_score_match

            # check if we have 99% match
            if candidates_method2[0].get("confidence",0.0) >= 99:
                # tag file
                success=tag_file(f, candidates_method2[0]["title"], candidates_method2[0]["artist"])

                if not success:
                    raise Method2_Error("Failed to tag file.")
                
                # update process file
                save_update(
                    {
                        "title":candidates_method2[0]["title"],
                        "artist":candidates_method2[0]["artist"],
                        "status": "done",
                        "method": "Lyrics Search",
                        "candidates_method2":candidates_method2
                    },
                    track_item,process_data,process_path
                )
                
                log(f"✔️ {f.name} -> title:{candidates_method2[0]['title']}")
                continue

        except Method2_Error as e:
            log(e)
        
        finally:
            # cleanup method 1 before method 2
            save_update(
                {
                    "status":"review",
                    "candidates_method2": candidates_method2
                },
                track_item,process_data,process_path
            )
        
        # Scoring Methods

        # compute all candidates scores
        log(f"Calculating Score Results...")
        # combine candidates
        candidates_casual_merged = [*candidates_method1, *candidates_method2]

        # extract candidates specific data
        # keys_to_extract = list(default_candidates.keys())
        # candidates_filtered = [
        #     {key: candidate.get(key) for key in keys_to_extract}
        #     for candidate in candidates_casual_merged
        # ]

        candidates_filtered = [
            {
                "confidence": candidate.get("confidence", 0.0),
                "occurrences": candidate.get("occurrences", 1),
                "title": candidate.get("title", ""),
                "artist": candidate.get("artist", ""),
                "lyrics": candidate.get("lyrics", ""),
                "method": "Fingerprint" if "segments" in candidate else "Search Lyrics",
                **({} if candidate.get("segments") is None else {"segments": candidate["segments"]}),
                **({} if candidate.get("search_kword") is None else {"search_kword": candidate["search_kword"]}),
            }
            for candidate in candidates_casual_merged
        ]


        # boost score candidates by occurences
        candidates_boosted = boost_confidence_by_occurrence(
            candidates_filtered,
            ["title"]
        )
        
        # filter candidates by confidence
        filtered_candidates = sorted([c for c in candidates_boosted if c.get('confidence',0.0) >= 50],key=lambda x: -x.get('confidence',0.0))

        if not filtered_candidates:
            log(f"❌ {f.name} -> Failed to find Matches")
            save_update(
                {
                    "status":"failed",
                    "status_message":"Failed to find Matches"
                },
                track_item,process_data,process_path
            )
            continue

        # check if we have 99% match
        if filtered_candidates[0].get("confidence",0.0) >= 99:
            # tag file
            success = tag_file(f, filtered_candidates[0]["title"], filtered_candidates[0]["artist"])
            
            if not success:
                log("Failed to Tag File.")
                continue 

            # update process file
            save_update(
                {
                    "title":filtered_candidates[0]["title"],
                    "artist":filtered_candidates[0]["artist"],
                    "status": "done",
                    "method": "Calculated Candidates",
                    "candidates":filtered_candidates
                },
                track_item,process_data,process_path
            )
            
            log(f"✔️ {f.name} -> title:{filtered_candidates[0]['title']}")
            continue
        else:
            log(f"📒 {f.name} -> Is Marked For Review, possible title:{filtered_candidates[0]['title']} with confidence of {filtered_candidates[0]['confidence']}")


        save_update(
            {
                "candidates" : filtered_candidates
            },
            track_item,process_data,process_path
        )
    if marked_for_review:
        save_to_process(
            review_track_data(
                process_data
            ),
            process_path
        )

if __name__ == "__main__":
    main()