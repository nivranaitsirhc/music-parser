# Smart Music Title Parser ver 4

import tempfile
import argparse
import json
import asyncio
import subprocess
import time
import shutil
import re
import os
import math
# from contextlib import contextmanager
import sys
from pathlib import Path
from datetime import datetime
from typing import cast, Tuple, List, Dict, Any, Iterable,  Callable, Awaitable, Optional, Union, Literal

# from lrclib import AsyncLRCLibClient
# from deezerlib import AsyncDeezerClient
from genius_api import GeniusSearch

import vlc
# from search_engines import Aol
import torch
# import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.fuzz import partial_ratio
from faster_whisper import WhisperModel
from shazamio import Shazam
# from polyfuzz import PolyFuzz
# from polyfuzz.models import TFIDF
# from lrcup import LRCLib
# import numpy as np


# Config
# ------------------------------------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".aac",
                    ".ogg", ".opus"}  # wav is reserve for processing files
PROCESS_FILE = "process.json"
LANGUAGE = None  # "en"

H_TITLE = "Smart Music Parser"
H_SUBTITLE = "GreenLeaf Labworks"
H_VERSION = "5"
H_AUTHOR = "Christian Arvin Cabo"

# Helpers
# ------------------------------------------------------------------------------------------------------


class HeaderDisplay:
    """A class to handle formatted header display"""

    def __init__(self, title: str, subtitle: str, version: str, author: str, box_width: int = 50):
        # box_width = 50  # Width of the box interior

        title_line = f"{title:^{box_width}}"
        subtitle_line = f"{subtitle:^{box_width}}"
        version_line = f"{'Ver. ' + str(version):^{box_width}}"
        author_line = f"{author:^{box_width}}"

        header = f"""
        ╔══════════════════════════════════════════════════╗
        ║     ███╗   ███╗██╗   ██╗███████╗██╗ ██████╗ 
        ║     ████╗ ████║██║   ██║██╔════╝██║██╔════╝ 
        ║     ██╔████╔██║██║   ██║███████╗██║██║      
        ║     ██║╚██╔╝██║██║   ██║╚════██║██║██║      
        ║     ██║ ╚═╝ ██║╚██████╔╝███████║██║╚██████╗ 
        ║     ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝ 
        ║{title_line} 
        ║                                           
        ║{subtitle_line}  
        ║{version_line}     
        ║{author_line}          
        ╚══════════════════════════════════════════════════╝
        """
        self.header = header

    def __call__(self):
        print(self.header)

    def print(self):
        print(self.header)

    def get(self) -> str:
        return self.header

    def get_height(self) -> int:
        return self.header.count('\n')


# init the header
header = HeaderDisplay(
    title=H_TITLE,
    subtitle=H_SUBTITLE,
    version=H_VERSION,
    author=H_AUTHOR
)


class AudioPlayer:
    def __init__(self):
        instance = vlc.Instance()
        if instance is None:
            raise RuntimeError("VLC instance could not be created")

        self.instance = cast(vlc.Instance, instance)
        self.player = self.instance.media_player_new()

    def play(self, path: str):
        media = self.instance.media_new(path)
        self.player.set_media(media)
        self.player.play()

    def stop(self):
        self.player.stop()

    def is_playing(self) -> bool:
        return bool(self.player.is_playing())


def log(msg, status: Union[str, Literal["d", "f", "l", "s", "p", "r", "w"]] = ""):
    icon: str
    match status:
        case "d":  # done
            icon = "✔️ "
        case "f":  # failed
            icon = "❌ "
        case "l":  # loading
            icon = "⏳ "
        case "s":  # skipping
            icon = "⏭️ "
        case "p":  # processing
            icon = "🔄 "
        case "r":  # review
            icon = "📒 "
        case "w":  # warning
            icon = "⚠️ "
        case _:
            icon = ""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {icon}{msg}")


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def sorted_by_confidence(lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort List of Dict by Confidence Descending"""
    return sorted(
        lists,
        key=lambda x: (
            -round(float(x.get("confidence", 0)), 6),
            -int(float(x.get("occurrences", 0))),
            (x.get("title") or "").casefold(),
            (x.get("artist") or "").casefold(),
        )
    ) if lists else []


def save_to_process(data, filepath, indent=2, max_compact=10):
    def is_primitive(x):
        return isinstance(x, (str, int, float, bool, type(None)))

    def is_primitive_list(lst):
        return all(is_primitive(x) for x in lst)

    def format_value(value: Any, level: int = 0) -> str:
        pad = ' ' * (indent * level)
        next_pad = ' ' * (indent * (level + 1))

        if isinstance(value, dict):
            if not value:
                return '{}'

            lines = ['{']
            items = list(value.items())

            for i, (k, v) in enumerate(items):
                comma = ',' if i < len(items) - 1 else ''
                rendered = format_value(v, level + 1)
                lines.append(f'{next_pad}"{k}": {rendered}{comma}')

            lines.append(f'{pad}}}')
            return '\n'.join(lines)

        if isinstance(value, list):
            if not value:
                return '[]'

            # Primitive lists always inline
            if is_primitive_list(value):
                return json.dumps(value, ensure_ascii=False)

            # Object lists always multiline
            lines = ['[']
            items = value if len(value) > max_compact else value

            for i, item in enumerate(items):
                comma = ',' if i < len(items) - 1 else ''
                rendered = format_value(item, level + 1)
                lines.append(f'{next_pad}{rendered}{comma}')

            lines.append(f'{pad}]')
            return '\n'.join(lines)

        return json.dumps(value, ensure_ascii=False)

    out = format_value(data) + '\n'
    Path(filepath).write_text(out, encoding='utf-8')


def save_update(updates, track_data, process_data, process_path):
    """Update track data and save to process file"""
    track_data.update(updates)
    save_to_process(process_data, process_path)


def tag_file(path: Path, title: str, artist: Optional[str] = None):
    tmp = path.with_suffix(".tmp." + path.suffix)
    metadata = ["-metadata", f"title={title}"]
    if artist:
        metadata = [*metadata, "-metadata", f"artist={artist}"]
    r = run([
        "ffmpeg", "-y", "-i", str(path),
        *metadata,
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


def ffmpeg_extract_segment(path: Path, start=None, duration=None) -> Path | None:
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


def ffmpeg_get_duration(path: Path) -> float:
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


def boost_confidence_by_occurrence(
    input_list: List[Dict[str, Any]],
    key_fields: Optional[List[str]] = None,
    boost_percent: float = 0.0,  # 0% default
    assume_sorted: bool = False,
) -> List[Dict[str, Any]]:
    """
    Merge duplicate track entries and boost confidence using full probabilistic fusion.

    Enhancements:
      - True probabilistic merge across *all* confidence values.
      - Occurrence-based asymptotic reinforcement.
      - Adds `confidence_list` capturing all contributing confidence sources.
      - Preserves provenance and prevents runaway inflation.
    """
    if not input_list:
        return []

    if key_fields is None:
        key_fields = ["title", "artist"]

    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    # Preserve order if already sorted
    items = input_list if assume_sorted else sorted(
        input_list,
        key=lambda x: (
            -round(float(x.get("confidence", 0.0)), 6)
        )
    )

    def clamp_prob(p: float) -> float:
        return min(max(p, 0.000001), 0.999999)

    def method_of(item: Dict[str, Any]) -> str:
        method = item.get("method")
        if isinstance(method, str) and method:
            return method
        else:
            return (
                "Fingerprint" if "segments" in item
                else "Lyrics Search"
            )

    def confidence_signature(conf: float, method: str) -> Tuple[float, str]:
        return (round(conf, 6), method)

    for item in items:
        key = tuple(item.get(k) for k in key_fields)
        if not all(key):
            continue

        conf = float(item.get("confidence", 0.0))
        conf = min(max(conf, 0.0), 100.0)
        p = clamp_prob(conf / 100.0)

        occurrences = max(1, int(item.get("occurrences", 1)))
        method = method_of(item)

        confidence_entry = {
            "confidence": conf,
            "occurrences": occurrences,
            "method": method,
            "probability": p,
        }

        sig = confidence_signature(conf, method)

        if key not in grouped:
            grouped[key] = {
                **item,
                "confidence_list": [confidence_entry],
                "_confidence_signatures": {sig: confidence_entry},
            }

            # Remove unsafe raw confidence to enforce exclusivity
            grouped[key].pop("confidence", None)
            continue

        bucket = grouped[key]

        # MUTUAL EXCLUSIVITY:
        # If incoming confidence not represented inside confidence_list,
        # treat as new independent evidence.
        if sig in bucket["_confidence_signatures"]:
            bucket["_confidence_signatures"][sig]["occurrences"] += occurrences
        else:
            bucket["_confidence_signatures"][sig] = confidence_entry
            bucket["confidence_list"].append(confidence_entry)

        # Merge segments safely
        if "segments" in bucket or "segments" in item:
            bucket["segments"] = list(dict.fromkeys([
                *bucket.get("segments", []),
                *item.get("segments", []),
            ]))

        # Merge kwords
        if "kwords" in bucket or "kowrds" in item:
            bucket["kwords"] = [
                *bucket.get("kwords", []),
                *item.get("kwords", []),
            ]

    # Bayesian fusion + calibrated reinforcement
    results: List[Dict[str, Any]] = []

    for data in grouped.values():
        conf_list = data["confidence_list"]

        # Bayesian independent fusion:
        #   P = 1 - Π (1 - pᵢ)
        product = 1.0
        total_occurrences = 0
        method_list = []

        for c in conf_list:
            p = c["probability"]
            occ = c.get("occurrences", 1)
            total_occurrences += occ
            method_list = [*method_list, c["method"]]

            # Controlled repetition reinforcement (log dampened)
            reinforced_p = 1.0 - ((1.0 - p) ** math.log2(occ + 1))
            product *= (1.0 - reinforced_p)

        fused_prob = 1.0 - product

        # Gentle linear boost (optional) only if occurrences happens
        if total_occurrences > 1 and boost_percent > 0:
            fused_prob *= (1.0 + boost_percent / 100.0)

        fused_prob = clamp_prob(fused_prob)
        final_conf = round(fused_prob * 100.0, 4)

        _method = "Mixed" if set(method_list) >= {
            "Fingerprint", "Lyrics Search"} else method_list[0] if method_list else ''

        # preserve structure
        data = {
            "confidence": final_conf,  # return the confidence that was poped
            **data,
            "occurrences": total_occurrences,
            "method": _method,
        }

        # Cleanup
        data.pop("_confidence_signatures", None)

        results.append(data)

    return sorted_by_confidence(results)


def filter_candidates_needing_lyrics(candidates: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter Candidates Needing Lyrics Keys"""
    if not keys:
        keys = ["title", "artist"]

    candidates_needing_lyrics = [
        c for c in candidates
        if ("lyrics" not in c or not c["lyrics"]) and
        all(key in c for key in keys)
    ]
    return candidates_needing_lyrics


def merge_lists_concise(
    original: List[Dict],
    updated: List[Dict],
    key_fields: Optional[Union[str, List[str]]] = None
) -> List[Dict]:
    """
    Merge updated items into original list based on a key.

    Args:
        original: Original list of dictionaries
        updated: Updated list with new data
        key_fields: How to identify matching items:
            - None: default to ['title', 'artist']
            - str: single field name (e.g., 'id')
            - List[str]: multiple field names (e.g., ['first', 'last'])

    Returns:
        Merged list
    """
    if key_fields is None:
        key_fields = ['title', 'artist']

    # Convert single field to list for uniform handling
    if isinstance(key_fields, str):
        key_fields = [key_fields]

    # Create lookup
    lookup = {}
    for item in updated:
        key = tuple(item.get(field) for field in key_fields)
        lookup[key] = item

    # Merge
    return [
        lookup.get(tuple(item.get(field) for field in key_fields), item)
        for item in original
    ]


# Shared Method Functions
# ------------------------------------------------------------------------------------------------------
async def genius_get_track_data(
    queries: List[Dict[str, Any]],
    fetch_fn: Optional[Callable[[str], Awaitable[dict]]] = None,
    *,
    keyword_keys: Iterable[str] = ("title",),
    joiner: str = " - ",
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
                result = await fetch_fn(keyword)
                # get the first result, get the lyrics and merge back to the item
                return {**item, "lyrics": result["lyrics"]}
            except Exception as e:
                return {**item}

    tasks = [runner(item) for item in queries]
    return await asyncio.gather(*tasks)


def whisper_detect_first_word(model, path: Path, max_seconds: Union[float, Literal['full']] = 30) -> float | None:
    duration = ffmpeg_get_duration(path)
    tmp = None
    if max_seconds == 'full':
        tmp = ffmpeg_extract_segment(path, 0.0, duration)
    else:
        tmp = ffmpeg_extract_segment(path, 0.0, max_seconds)
    if not tmp:
        return None
    segments, _ = model.transcribe(
        str(tmp), language=LANGUAGE, word_timestamps=True)
    tmp.unlink(missing_ok=True)
    for seg in segments:
        if seg.words:
            return seg.words[0].start
    return None


def whisper_transcribe_from_start(model, path: Path, start: float, seconds: Union[float, Literal['full']]) -> str:
    duration = ffmpeg_get_duration(path)
    tmp = None
    if seconds == 'full':
        tmp = ffmpeg_extract_segment(path, 0.0, duration)
    else:
        tmp = ffmpeg_extract_segment(path, start, seconds)
    if not tmp:
        return ""
    segments, _ = model.transcribe(str(tmp), language=LANGUAGE)
    tmp.unlink(missing_ok=True)
    text = " ".join(s.text.strip() for s in segments)
    return text.strip()


def score_candidates_by_lyrics_match(
    lyric_sample: str,
    song_list: List[Dict[str, Any]],
    *,
    tfidf_weight: float = 0.20,
    jaccard_weight: float = 0.20,
    partial_weight: float = 0.60
) -> List[Dict[str, Any]]:
    """
    Score each track in `song_list` against `lyric_sample` using a hybrid
    matching strategy: TF-IDF + Jaccard + fuzzy partial matching.

    If a track already contains a "confidence" value, the new score is merged
    with the existing one using probabilistic fusion instead of overwriting.

    This preserves evidence accumulation and prevents destructive overwrites.
    """

    if not lyric_sample or not song_list:
        return song_list

    def normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()

    def jaccard_similarity(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(len(sa | sb), 1)

    def probabilistic_merge(a: float, b: float) -> float:
        """Merge two confidence values in [0,1] probabilistically."""
        return 1.0 - ((1.0 - a) * (1.0 - b))

    normalized_sample = normalize(lyric_sample)
    normalized_lyrics = [normalize(t.get("lyrics", "")) for t in song_list]

    if not any(normalized_lyrics):
        return song_list

    # ---------- TF-IDF similarity ----------
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(
        [normalized_sample] + normalized_lyrics
    )

    tfidf_scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    ).flatten()

    # ---------- Per-track hybrid scoring ----------
    for idx, track in enumerate(song_list):
        lyrics_text = normalized_lyrics[idx]

        # do nothing if track has no lyrics
        if not lyrics_text:
            continue

        tfidf_score = float(tfidf_scores[idx])
        jaccard_score = jaccard_similarity(normalized_sample, lyrics_text)
        partial_score = partial_ratio(normalized_sample, lyrics_text) / 100.0

        lyric_conf = (
            (partial_weight * partial_score) +
            (tfidf_weight * tfidf_score) +
            (jaccard_weight * jaccard_score)
        )

        lyric_conf = min(max(lyric_conf, 0.0), 1.0)

        # Existing confidence merge (probabilistic fusion)
        old_conf = float(track.get("confidence", 0.0)) / 100.0
        merged_conf = probabilistic_merge(old_conf, lyric_conf)

        # Track Update
        track.update({
            **track,
            "confidence": round(merged_conf * 100.0, 2),
            "method": "Lyrics Match",
            "lyrics_match_scoring_results": {
                "final": round(lyric_conf, 4),
                "scores": {
                    "tfidf": round(tfidf_score, 4),
                    "jaccard": round(jaccard_score, 4),
                    "partial": round(partial_score, 4),
                }
            }
        })

    return song_list


# Method 1 Functions
# ------------------------------------------------------------------------------------------------------

async def shazam_detect(path: Path, segment_seconds: int = 10) -> Dict[str, Any] | None:
    shazam = Shazam(
        language="en-US",
        endpoint_country="US",
        segment_duration_seconds=segment_seconds
    )
    try:
        out = await shazam.recognize(str(path))
        track = out.get("track")
        if not track:
            return None
        return {
            "title": track.get("title"),
            "artist": track.get("subtitle")
        }
    except Exception:
        return None


async def fingerprint_track(audio: Path, seconds: Union[float, Literal['full']]) -> List[Dict]:
    """Async Wrapper for Shazam Detection"""
    duration = ffmpeg_get_duration(audio)
    shazam_segment_seconds = 10
    if seconds == 'full':
        # divide the whole duration that will fit the shazam detection
        segments = [
            (i, min(i+shazam_segment_seconds, duration))
            for i in (shazam_segment_seconds * j for j in range(int(duration // shazam_segment_seconds) + 1))
            if i < duration
        ]
    else:
        # segment track by seconds
        segments = [
            (i, min(i+seconds, duration))
            for i in (seconds * j for j in range(int(duration // seconds) + 1))
            if i < duration
        ]

    results = []
    for start, dur in segments:
        tmp = ffmpeg_extract_segment(audio, start, dur)
        if not tmp:
            continue

        result = await shazam_detect(tmp, shazam_segment_seconds)
        tmp.unlink(missing_ok=True)

        if result:
            if seconds == 'full':
                result.update({
                    "confidence": 99,
                })
            else:
                result.update({
                    "confidence": 95,
                })
            result.update({
                "segments": [start, dur]
            })
            results.append(result)

    return results


# Method 2 Functions
# ------------------------------------------------------------------------------------------------------

def split_lyrics_to_chunks(long_string: str, max_words: int, max_chunks: int = 3) -> list[str]:
    pattern = r'(?:\s*,\s*|\s*\n\s*|\s+and\s+)'
    parts = re.split(pattern, long_string)
    words = [w for part in parts for w in part.strip().split()]
    results = [' '.join(words[i:i + max_words])
               for i in range(0, len(words), max_words)]
    return results[:max_chunks]


async def fetch_song(query: str) -> Dict[str, Any]:
    if not query:
        return {}

    client = GeniusSearch()
    result = await client.fetch_song(query)
    return result


async def search_tracks(lyrics: List[str]) -> List[Dict[str, Any]]:

    if not lyrics:
        return []

    async def search_chunk(chunk):
        client = GeniusSearch()
        result = await client.search(chunk)
        return await result.all()

    tasks = [search_chunk(chunk) for chunk in lyrics]
    results = await asyncio.gather(*tasks)

    # Flatten the list of lists
    all_tracks = []
    for result in results:
        for track in result:
            if track.get('title', ''):
                all_tracks.extend([{
                    "title": track.get('title', ''),
                    "artist": track.get('artist', ''),
                    "album": track.get('album', ''),
                    "lyrics": track.get('plainLyrics', ''),
                }])
    return all_tracks

# Args Custom Type
# ------------------------------------------------------------------------------------------------------


def type_positive_float_or_full(value):
    if value.lower() == 'full':
        # or return a special value like None, float('inf'), etc.
        return 'full'
    try:
        f = float(value)
        if f <= 0:
            raise argparse.ArgumentTypeError(
                f"'{value}' must be greater than 0")
        return f
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a positive float or 'full'")


def type_method(value):
    try:
        if value == "1":
            return 1
        elif value == "2":
            return 2
        elif value.lower() == 'all':
            return 'all'
        raise argparse.ArgumentTypeError(
            f"'{value}' must be 1, 2  or all")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid input, input 1, 2, all")


# Review Logic
# ------------------------------------------------------------------------------------------------------
class CliReviewPanel:
    def __init__(self):
        self.queue = Path(tempfile.NamedTemporaryFile(
            delete=False, suffix=".queue.json").name)
        self.result = Path(tempfile.NamedTemporaryFile(
            delete=False, suffix=".result.json").name)
        self.control = Path(tempfile.NamedTemporaryFile(
            delete=False, suffix=".ctrl").name)

        # Start clean
        self.queue.unlink(missing_ok=True)
        self.result.unlink(missing_ok=True)
        self.control.unlink(missing_ok=True)

        self._spawn()

    def _spawn(self):
        python = str(sys.executable)
        script = str(Path(__file__).resolve())

        cmd = [
            "wt.exe",
            "-w", "0",
            "-M",
            "nt",  # "split-pane", # "--vertical",
            "--",
            python,
            script,
            "--cli-review-worker",
            str(self.queue),
            "--cli-result-file",
            str(self.result),
            "--cli-control-file",
            str(self.control),
        ]

        subprocess.Popen(cmd, shell=False)

    def send(self, track: dict, page_size: int):
        payload = {
            "track": track,
            "page_size": page_size,
        }

        tmp = self.queue.with_suffix(".tmp")

        tmp.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8"
        )

        os.replace(tmp, self.queue)   # atomic on Windows

        while not self.result.exists():
            time.sleep(0.05)

        result = json.loads(self.result.read_text(encoding="utf-8"))
        self.result.unlink(missing_ok=True)
        return result

    def close(self):
        self.control.touch()


def save_cursor():
    sys.stdout.write("\033[s")
    sys.stdout.flush()


def restore_cursor():
    sys.stdout.write("\033[u")
    sys.stdout.flush()


def set_scroll_region(top: int, bottom: int):
    """Restrict terminal scrolling to [top, bottom] rows (1-indexed)."""
    sys.stdout.write(f"\033[{top};{bottom}r")
    sys.stdout.flush()


def reset_scroll_region():
    sys.stdout.write("\033[r")
    sys.stdout.flush()


def soft_clear():
    """Clear the terminal and set the cursor to 1,1"""
    sys.stdout.write("\033[2J\033[H")  # Clear and home cursor
    sys.stdout.flush()


def clear_below_cursor():
    """Clear from cursor to bottom of scroll region."""
    sys.stdout.write("\033[J")
    sys.stdout.flush()


def terminal_height() -> int:
    return shutil.get_terminal_size((80, 24)).lines


def compute_page_size(term_height: int, header_lines: int) -> int:
    """Compute the available candidate item per page"""
    track_info = 6     # track info + pagination
    footer_lines = 2   # command prompt + spacing
    per_candidate = 7  # per-candidate block height

    usable = term_height - (header_lines + track_info + footer_lines)

    # return the max candidate item to display
    return max(3, usable // per_candidate)


def cli_review_loop(track: Dict[str, Any], _: int) -> Optional[Dict[str, Any]]:
    def paginate(items, page, size):
        start = page * size
        return items[start:start + size]

    player = AudioPlayer()
    path = Path(track.get("path", ""))
    candidates = track.get("candidates", [])

    if not candidates:
        return {"action": "no_valid", "reason": "No Valid Candidates"}

    prev_track = ""
    page = 0

    # Clear the Terminal
    soft_clear()

    # Print banner once
    h_width = 120
    header = "═" * (h_width - 1) + "╗"
    footer = "╚" + "═" * (h_width - 1)
    print(header)
    print(f"{H_TITLE + ' --- Track Review':^{h_width}}")
    header_lines = 2
    # footer_lines = 2

    # Position cursor BELOW header
    sys.stdout.write(f"\033[{header_lines + 1};1H")
    sys.stdout.flush()

    # Save cursor for CLI redraw
    save_cursor()
    # Restrict scroll region to below header
    term_h = terminal_height()
    # Restrict scrolling to body only (below header)
    set_scroll_region(header_lines + 1, term_h)

    try:
        while True:
            # Compute page size dynamically
            page_size = compute_page_size(term_h, header_lines)
            total_pages = max(1, math.ceil(len(candidates) / page_size))
            page = max(0, min(page, total_pages - 1))
            page_candidates = paginate(candidates, page, page_size)

            # Redraw CLI region only
            restore_cursor()
            clear_below_cursor()

            if prev_track != str(path):
                prev_track = str(path)
                player.play(str(path))

            print(
                f"Track Name     : {path.name}")
            print(
                f"Method 1 Sample: {track.get('m1_lyrics_sample','N/A')[:120]}")
            print(
                f"Method 2 Sample: {track.get('m2_lyrics_sample','N/A')[:120]}")
            print(
                f"\nPage {page + 1}/{total_pages}  |  Showing {len(page_candidates)} results\n")

            for i, c in enumerate(page_candidates, start=1):
                print(f"{i:02d} -> Confidence : {c.get('confidence', 0):.1f}")
                print(f"      Occurrences: {c.get('occurrences', 0)}")
                print(f"      Title      : {c.get('title', '')}")
                print(f"      Artist     : {c.get('artist', '')}")
                print(f"      Method     : {c.get('method', '')}")
                print(f"      Lyrics     : {(c.get('lyrics') or 'N/A')[:120]}")
                print()

            footer_text = f"{footer}\nSelect [1-N], < prev, > next, c custom, p play, s stop, Enter skip, Q quit: "
            cmd = input(
                footer_text
            ).strip().lower()

            if not cmd:
                player.stop()
                return {"action": "skip"}

            if cmd == "q":  # quit
                player.stop()
                return {
                    "action": "quit",
                    "reason": "User Terminated the Review Process"
                }

            if cmd == "p":  # play sound
                player.play(str(path))
                continue

            if cmd == "s":  # stop sound
                player.stop()
                continue

            if cmd == "<":
                page -= 1
                continue

            if cmd == ">":
                page += 1
                continue

            if cmd in {"c", "custom"}:
                restore_cursor()
                clear_below_cursor()
                print("\nManual Tagging:")
                title = input("  Enter Title : ").strip()
                artist = input("  Enter Artist: ").strip()
                if not title:  # or not artist:
                    continue
                player.stop()
                return {
                    "action": "manual",
                    "title": title,
                    "artist": artist,
                }

            if cmd.isdigit():
                idx = int(cmd) - 1
                if 0 <= idx < len(page_candidates):
                    player.stop()
                    return {
                        "action": "select",
                        "candidate": page_candidates[idx],
                    }

    finally:
        reset_scroll_region()


def create_ipc_files(payload: dict):
    payload_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".json",
        encoding="utf-8"
    )
    json.dump(payload, payload_file, ensure_ascii=False)
    payload_file.close()

    result_file = Path(payload_file.name).with_suffix(".result.json")

    return Path(payload_file.name), result_file


def run_cli_worker_persistent(queue_path: Path, result_path: Path, control_path: Path):
    import time
    import json

    # print banner
    header()

    print("\n🎛️ Review Panel Ready\n")
    print(f"Queue   : {queue_path}")
    print(f"Result  : {result_path}")
    print(f"Control : {control_path}\n")

    while True:
        if control_path.exists():
            print("\n👋🏻 Review panel shutting down...\n")
            time.sleep(3)
            break

        if not queue_path.exists():
            time.sleep(0.05)
            continue

        try:
            payload = json.loads(queue_path.read_text(encoding="utf-8"))
            queue_path.unlink(missing_ok=True)
        except Exception as e:
            print("IPC Read Error:", e)
            time.sleep(0.05)
            continue

        track = payload["track"]
        page_size = payload.get("page_size", 3)

        result = cli_review_loop(track, page_size)

        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(result or {"action": "quit"}, ensure_ascii=False),
            encoding="utf-8"
        )
        os.replace(tmp, result_path)

        # result_path.write_text(
        #     json.dumps(result or {"action": "quit"}, ensure_ascii=False),
        #     encoding="utf-8"
        # )

        if not result or result.get("action") == "quit":
            break


def review_track_data(
    process_list: List[Path],
    process_data: Dict[str, Dict[str, Any]],
    process_path: Path,
    page_size: int = 3,
) -> bool:

    process_names = {p.name for p in process_list}
    review_items = [
        track_data for track, track_data in process_data.items()
        if track in process_names
        and track_data.get("status") == "review"
    ]

    if not review_items:
        print("✔️ No tracks pending review.")
        return False

    panel = CliReviewPanel()

    # return False to stop processing of more tracks
    do_not_process = True

    try:
        for idx, track in enumerate(review_items, 1):
            path = Path(track["path"])

            print(f"\n🔎 [{idx}/{len(review_items)}] Reviewing: {path.name}")

            result = panel.send(track, page_size)

            if not result or result["action"] == "quit":
                print(f"⏹️ {result['reason']}")
                break

            if result["action"] == "no_valid":
                print(f"⏭️ Skipped: {path.name} --- {result['reason']}")
                do_not_process = False
                continue

            if result["action"] == "skip":
                print(f"⏭️ Skipped: {path.name} --- {result['reason']}")
                continue

            if result["action"] == "manual":
                tag_file(path, result["title"], result["artist"])
                save_update(
                    {
                        "status": "done",
                        "title": result["title"],
                        "artist": result["artist"],
                        "method": "manual",
                        "status_message": "User manual override",
                    },
                    track,
                    process_data,
                    process_path,
                )
                print(
                    f"✔️ Tagged (manual): {path.name} -> Title: {result['title']} Artist: {result.get('artist','')}")

            elif result["action"] == "select":
                chosen = result["candidate"]
                tag_file(path, chosen["title"], chosen["artist"])
                save_update(
                    {
                        "status": "done",
                        "title": chosen["title"],
                        "artist": chosen["artist"],
                        "method": chosen.get("method", ""),
                        "status_message": "User reviewed",
                    },
                    track,
                    process_data,
                    process_path,
                )
                print(
                    f"✔️ Tagged: {path.name} -> Title: {chosen['title']} Artist: {chosen.get('artist','')}")
    except:
        do_not_process = False

    finally:
        panel.close()

    return do_not_process

# Main


def main():
    header()

    class ResetCleanAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Check if any reset-* args are present
            reset_args = [arg for arg, value in vars(namespace).items()
                          if arg.startswith('reset_') and value == True]
            if reset_args:
                parser.error(
                    "--force-clean cannot be used with any --reset-* arguments")
            setattr(namespace, self.dest, True)

    p = argparse.ArgumentParser(
        description="""
        
        Determine the Title and Artist of a Track by using multiple methods.
        
            Method 1 - Fingerprint Track using ShazamIO,
            Methid 2 - Determine the Track title using Lyrics Matching

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    # target directory
    p.add_argument("--dir",
                   required=True,
                   metavar='PATH',
                   help="Path to the directory tracks to process."
                   )
    # select target directory
    p.add_argument("--target-filename",
                   type=str,
                   metavar='FILE.EXTENSION',
                   default=None,
                   help="Single-out the track to process in the target directory."
                   )
    # select method
    p.add_argument("--method",
                   type=type_method,
                   metavar='METHOD',
                   default='all',
                   help="Select method to process the file, 1, 2, 'ALL'."
                   )
    p.add_argument("--m1-fingerprint-sample-duration",
                   type=type_positive_float_or_full,
                   metavar='FLOAT',
                   default=15.0,
                   help="Method 1 - The Sample size of each segment of the track to fingerprint. (default: 15)"
                   )
    p.add_argument("--m1-lyrics-sample-duration",
                   type=type_positive_float_or_full,
                   metavar='FLOAT',
                   default=60.0,
                   help="Method 1 - The duration of the track to sample and transcribe the lyrics. (default: 45)"
                   )
    p.add_argument("--m2-lyricsearch-sample-duration",
                   type=type_positive_float_or_full,
                   metavar='FLOAT',
                   default=60.0,
                   help="Method 2 - The duration of the track to sample and transcribe the lyrics. (default: 60)"
                   )
    p.add_argument("--lyrics-firstword-detection-duration",
                   type=type_positive_float_or_full,
                   metavar='FLOAT',
                   default=60.0,
                   help="Shared   - The duration of the track to sample and detect the first word of the track. (default: 60)"
                   )
    p.add_argument("--force-clean",
                   action=ResetCleanAction,
                   nargs=0,
                   const=None,  # or just omit nargs and const
                   default=False,
                   help="Reset the process by removing the PROCESS_FILE before starting the operation."
                   )
    p.add_argument("--reset-status",
                   action="store_true",
                   help="Reset the status of each track in the PROCESS_FILE"
                   )
    p.add_argument("--reset-m1-candidates",
                   action="store_true",
                   help="Reset the Method 1 candidates for each Track."
                   )
    p.add_argument("--reset-m2-candidates",
                   action="store_true",
                   help="Reset the Method 2 candidates for each Track"
                   )
    # p.add_argument("--reset-track-lyrics",action="store_true")
    # p.add_argument("--reset-track-lyrics-chunks",action="store_true")
    p.add_argument("--review",
                   action="store_true",
                   help="Initiate the reivew for Track's with review status."
                   )
    args = p.parse_args()

    # default Lyrics Scoring Data Structure
    DEFAULT_LYRICS_SCORING = {
        "final": 0.0,
        "scores": {
            "jaccard": 0.0,
            "partial": 0.0,
            "tfidf": 0.0
        }
    }

    # default Candidate Data Structure
    DEFAULT_CANDIDATE = {
        "confidence": 0,
        "occurrences": 1,
        "title": "",
        "artist": "",
        "method": "",
        "album": "",
        "lyrics": "",
        "lyrics_match_scoring_results": {},
    }

    # default Method 1 Candidates Data Structure
    DEFAULT_CANDIDATE_METHOD1 = {
        **DEFAULT_CANDIDATE,
        "segments": []
    }

    # default Method 2 Candidates Data Structure
    DEFAULT_CANDIDATE_METHOD2 = {
        **DEFAULT_CANDIDATE,
        "track_id": 0,
        "isrc": "",
        "duration": ""
    }

    # default Track Data Structure
    DEFAULT_TRACK_DATA = {
        "path": "",
        "title": "",
        "artist": "",
        "status": "",
        "status_message": "",
        "method": "",
        "detected_firstword_time": 0.0,
        "lyrics_firstword_detection_duration": args.lyrics_firstword_detection_duration,
        "m1_fingerprint_sample_duration": args.m1_fingerprint_sample_duration,
        "m1_lyrics_sample_duration": args.m1_lyrics_sample_duration,
        "m1_lyrics_sample": "",
        "m2_lyricsearch_sample_duration": args.m2_lyricsearch_sample_duration,
        "m2_lyrics_sample": "",
        "m2_lyrics_sample_chunks": [],
        "candidates": [],
        "m1_candidates": [],
        "m2_candidates": [],
    }

    # Preprocess Initialization
    # --------------------------------------------------------------------------------------------------

    # process.json
    DIRECTORY = Path(args.dir).resolve()
    # process file
    PROCESS_PATH = DIRECTORY / PROCESS_FILE
    # list of tracks path
    PROCESS_LIST = list(DIRECTORY.iterdir())  # eager loading

    # clean up process.json
    if args.force_clean and PROCESS_PATH.exists():
        PROCESS_PATH.unlink(missing_ok=True)  # delete the file

    # get data from process.json
    PROCESS_DATA = (
        json.loads(content)
        if PROCESS_PATH.exists() and (content := PROCESS_PATH.read_text(encoding="utf-8"))
        else {}
    )

    # ensure that process data is of type dict
    if not isinstance(PROCESS_DATA, dict):
        PROCESS_DATA = {}

    # process only one file
    if args.target_filename:
        _filtered_list = [
            x for x in PROCESS_LIST if x.name == args.target_filename
        ]
        # Check if target exists in PROCESS_DATA
        if _filtered_list:
            log(f"Targeting only : {args.target_filename}", "l")
            PROCESS_LIST = _filtered_list
        else:
            log(f"Warning! '{args.target_filename}' not found in the traget directory", "w")

    # data manipulation
    updates = {}

    if args.reset_status:
        updates['status'] = ""

    if args.reset_m1_candidates:
        updates['m1_candidates'] = []

    if args.reset_m2_candidates:
        updates['m2_candidates'] = []

    # Apply updates if any exist
    if updates:
        # PROCESS_DATA = {track: {**track_data, **updates}
        #                 for track, track_data in PROCESS_DATA.items()}
        process_set = set({p.name for p in PROCESS_LIST})
        PROCESS_DATA = {
            track: {**track_data, **updates}
            if track in process_set else track_data
            for track, track_data in PROCESS_DATA.items()
        }

    # # Handle lyrics separately (modifies in-place)
    # if args.reset_track_lyrics:
    #     for track_data in PROCESS_DATA.values():
    #         track_data.update({"lyrics_sample": "", "lyrics_sample_duration": 0})

    # # Handle lyrics chunks
    # if args.reset_track_lyrics_chunks:
    #     for track_data in PROCESS_DATA.values():
    #         track_data.update({"lyrics_sample_chunks":[]})

    # Save only if changes were made
    if updates:
        save_to_process(PROCESS_DATA, PROCESS_PATH)

    # review
    marked_for_review = False
    if args.review:
        if isinstance(PROCESS_DATA, dict) and PROCESS_DATA:
            result = review_track_data(
                PROCESS_LIST,
                PROCESS_DATA,
                PROCESS_PATH
            )
            if result:
                return None
            else:
                marked_for_review = True

        else:
            marked_for_review = True

    # init whisper model
    MODEL = WhisperModel(
        "base",  # default to base for faster operation
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8"
    )

    # Main process loop
    for TRACK in PROCESS_LIST:
        # skip non audio files
        if TRACK.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        log(f"⏳ {TRACK.name} -> Processing..")

        # get the track entry from process_data if any
        TRACK_ITEM = PROCESS_DATA.get(TRACK.name)

        if not TRACK_ITEM:
            TRACK_ITEM = dict(DEFAULT_TRACK_DATA)
            TRACK_ITEM["path"] = str(TRACK)
            PROCESS_DATA[TRACK.name] = TRACK_ITEM
            save_to_process(PROCESS_DATA, PROCESS_PATH)

        # skip with done tracks
        if TRACK_ITEM.get("status") == "done":
            log(f"⏭️ {TRACK.name} -> Skiping already done! title:{TRACK_ITEM.get('title')}")
            continue

        # Method Init
        STATUS_M1 = 'processing'
        STATUS_M2 = 'processing'
        ERROR_METHOD1 = ValueError
        ERROR_METHOD2 = ValueError
        METHOD = args.method
        M1_CANDIDATES = TRACK_ITEM.get("m1_candidates", [])
        M2_CANDIDATES = TRACK_ITEM.get("m2_candidates", [])
        DETECTED_FIRSTWORD_TIME = TRACK_ITEM.get(
            "detected_firstword_time", 0.0)
        LYRICS_FIRSTWORD_DETECTION_DURATION = TRACK_ITEM.get(
            "lyrics_firstword_detection_duration", 0.0)
        M1_FINGERPRINT_SAMPLE_DURATION = TRACK_ITEM.get(
            "m1_fingerprint_sample_duration", 0.0)
        M1_LYRICS_SAMPLE_DURATION = TRACK_ITEM.get(
            "m1_lyrics_sample_duration", 0.0)
        M1_LYRICS_SAMPLE = TRACK_ITEM.get("m1_lyrics_sample", "")
        M2_LYRICSEARCH_SAMPLE_DURATION = TRACK_ITEM.get(
            "m2_lyricsearch_sample_duration", 0.0)
        M2_LYRICS_SAMPLE = TRACK_ITEM.get("m2_lyrics_sample", "")
        M2_LYRICS_SAMPLE_CHUNKS = TRACK_ITEM.get("m2_lyrics_sample_chunks", "")

        # Method 1
        if METHOD == 1 or METHOD == 'all':
            log(f"🔄 Method 1: Track Fingerprinting")
            try:
                # Redo Fingerprinting if duration changed
                if M1_FINGERPRINT_SAMPLE_DURATION != args.m1_fingerprint_sample_duration or not M1_CANDIDATES:
                    log(f"🔄 Method 1 -> Step 1 : Shazam Analyzing Track..")
                    M1_CANDIDATES_IDENTIFIED_TRACKS = asyncio.run(
                        fingerprint_track(TRACK, args.m1_fingerprint_sample_duration))

                    if not M1_CANDIDATES_IDENTIFIED_TRACKS:
                        raise ERROR_METHOD1(
                            f"🔄 Method 1 -> Step 1: Failed To Analyze Track!")

                    M1_CANDIDATES = M1_CANDIDATES_IDENTIFIED_TRACKS

                    # update candidates to match default dict
                    M1_CANDIDATES = [{**DEFAULT_CANDIDATE_METHOD1, **track}
                                     for track in M1_CANDIDATES]

                else:
                    log(f"🔄 Method 1 -> Step 1 : Reusing Cached Fingerprints")

                # boost confidence by re-occurrences
                M1_CANDIDATES = boost_confidence_by_occurrence(
                    M1_CANDIDATES, boost_percent=2)

                # Tag if we are 100 sure.
                if M1_CANDIDATES[0].get("confidence", 0.0) >= 99.99:
                    if len(M1_CANDIDATES) > 1 and M1_CANDIDATES[1].get("confidence", 0.0) >= 99.99:
                        STATUS_M1 = "review"
                    else:
                        # Single 99% match - tag both title and artist
                        tag_title = M1_CANDIDATES[0]['title']
                        tag_artist = M1_CANDIDATES[0]['artist']
                        STATUS_M1 = "done"
                        log(f"{TRACK.name} -> Title: {tag_title} Artist: {tag_artist}", "d")

                        # Call with correct arguments
                        success = tag_file(TRACK, tag_title, tag_artist)

                        if not success:
                            log(f"Failed to tag file", "w")

                        continue

                # Search for the Track Lyrics of each candidates and match with the track Lyrics.
                log(f"🔄 Method 1 -> Step 2 : Genius searching for shazam Lyrics")
                # Filter candidates Needing Lyrics
                M1_CANDIDATES_NEEDING_LYRICS = filter_candidates_needing_lyrics(
                    M1_CANDIDATES)
                M1_CANDIDATES_FETCHED_LYRICS = asyncio.run(genius_get_track_data(
                    M1_CANDIDATES_NEEDING_LYRICS,
                    fetch_song,
                    keyword_keys=("title", "artist")
                ))

                # check if we have atleast one valid lyrics
                if not any(d.get('lyrics') for d in M1_CANDIDATES_FETCHED_LYRICS):
                    raise ERROR_METHOD1(
                        f"🔄 Method 1 -> Step 2 : Failed search lyrics data!")

                # Merge Back the Candidates
                M1_CANDIDATES = merge_lists_concise(
                    M1_CANDIDATES, M1_CANDIDATES_FETCHED_LYRICS, key_fields=['title', 'artist'])

                # Check for Sample Lyrics and Generate if none
                if M1_LYRICS_SAMPLE_DURATION != args.m1_lyrics_sample_duration or not M1_LYRICS_SAMPLE:
                    log(f"🔄 Method 1 -> Step 3 : Transcribing Track Lyrics")
                    # Step 1 - Use whisper determine the time of the 1st word of the track
                    if not DETECTED_FIRSTWORD_TIME or LYRICS_FIRSTWORD_DETECTION_DURATION != args.lyrics_firstword_detection_duration:
                        log("🔄 Method 1 -> Step 3A: Whisper analyzing track first word...")
                        detected = whisper_detect_first_word(
                            MODEL, TRACK, args.lyrics_firstword_detection_duration)

                        if detected is None:
                            raise ERROR_METHOD1(
                                "🔄 Method 1 -> Step 3A: Failed to detect first word of track."
                            )

                        DETECTED_FIRSTWORD_TIME = float(detected)

                    log(f"🔄 Method 1 -> Step 3B: Whisper transcribing to generate track lyrics sample...")
                    M1_LYRICS_SAMPLE = whisper_transcribe_from_start(
                        MODEL, TRACK, DETECTED_FIRSTWORD_TIME, args.m1_lyrics_sample_duration)

                    if not M1_LYRICS_SAMPLE:
                        raise ERROR_METHOD1(
                            "🔄 Method 1 -> Step 3B: Failed to transcribe lyrics")
                else:
                    log(f"🔄 Method 1 -> Step 3 : Reusing Lyrics Sample")

                # score candidates by lyrics match
                log(f"🔄 Method 1 -> Step 4 : Scoring by Lyrics Match..")
                M1_CANDIDATES = score_candidates_by_lyrics_match(
                    M1_LYRICS_SAMPLE, M1_CANDIDATES)

                # check if we have 99% match
                if M1_CANDIDATES[0].get("confidence", 0.0) >= 99.99:
                    if len(M1_CANDIDATES) > 1 and M1_CANDIDATES[1].get("confidence", 0.0) >= 99.99:
                        STATUS_M1 = "review"
                    else:
                        # Single 99% match - tag both title and artist
                        tag_title = M1_CANDIDATES[0]['title']
                        tag_artist = M1_CANDIDATES[0]['artist']
                        STATUS_M1 = "done"
                        log(f"{TRACK.name} -> Title: {tag_title} Artist: {tag_artist}", "d")

                        # Call with correct arguments
                        success = tag_file(TRACK, tag_title, tag_artist)

                        if not success:
                            log(f"Failed to tag file", "w")

                        continue

                if METHOD == 1:
                    if M1_CANDIDATES[0].get("confidence", 0.0) <= 0:
                        raise ERROR_METHOD1(
                            f"🔄 Method 1 -> 4 : Failed to find candidate with matching score")

                # Set status for review
                STATUS_M1 = "review"

            except ERROR_METHOD1 as e:
                log(e)
                STATUS_M1 = 'failed'

            finally:
                save_update(
                    {
                        "title": M1_CANDIDATES[0]["title"] if STATUS_M1 == "done" else "",
                        "artist": M1_CANDIDATES[0]["artist"] if STATUS_M1 == "done" else "",
                        "status": STATUS_M1,
                        "method": "Fingerprint",
                        "detected_firstword_time": DETECTED_FIRSTWORD_TIME,
                        "lyrics_firstword_detection_duration": args.lyrics_firstword_detection_duration,
                        "m1_lyrics_sample": M1_LYRICS_SAMPLE,
                        "m1_lyrics_sample_duration": args.m1_lyrics_sample_duration,
                        "m1_candidates": M1_CANDIDATES  # do not clip
                    },
                    TRACK_ITEM, PROCESS_DATA, PROCESS_PATH
                )

            if METHOD == 1:  # standalone
                if STATUS_M1 == "failed":
                    log(f"❌ {TRACK.name} -> Failed to find Matches")
                elif STATUS_M1 == "review":
                    log(
                        f"📒 {TRACK.name} -> Is marked for review, possible title:{M1_CANDIDATES[0]['title']} with confidence of {M1_CANDIDATES[0]['confidence']}")
                    candidates_filtered = [
                        {
                            "confidence": candidate.get("confidence", 0.0),
                            "occurrences": candidate.get("occurrences", 1),
                            "title": candidate.get("title", ""),
                            "artist": candidate.get("artist", ""),
                            "lyrics": candidate.get("lyrics", ""),
                            "method": "Fingerprint",
                            **({} if candidate.get("segments") is None else {"segments": candidate["segments"]}),
                        }
                        for candidate in M1_CANDIDATES
                    ]
                    save_update(
                        {
                            "candidates": candidates_filtered
                        },
                        TRACK_ITEM, PROCESS_DATA, PROCESS_PATH
                    )
                # next loop
                continue

            if STATUS_M1 == "done":
                # next loop
                continue

        # Method 2
        if METHOD == 2 or METHOD == 'all':
            log(f"🔄 Method 2: Lyrics Search & Matching")
            try:
                if M2_LYRICSEARCH_SAMPLE_DURATION != args.m2_lyricsearch_sample_duration or not M2_CANDIDATES:
                    log(f"🔄 Method 2 -> Step 1 : Transcribing Track Lyrics")

                    if not DETECTED_FIRSTWORD_TIME or LYRICS_FIRSTWORD_DETECTION_DURATION != args.lyrics_firstword_detection_duration:
                        log("🔄 Method 2 -> Step 1A: Whisper analyzing track first word...")
                        detected = whisper_detect_first_word(
                            MODEL, TRACK, args.lyrics_firstword_detection_duration)

                        if detected is None:
                            raise ERROR_METHOD1(
                                "🔄 Method 2 -> Step 1A: Failed to detect first word of track."
                            )

                        DETECTED_FIRSTWORD_TIME = float(detected)

                    log(f"🔄 Method 2 -> Step 1B: Whisper transcribing to generate track lyrics sample...")
                    M2_LYRICS_SAMPLE = whisper_transcribe_from_start(
                        MODEL, TRACK, DETECTED_FIRSTWORD_TIME, args.m2_lyricsearch_sample_duration)

                    if not M2_LYRICS_SAMPLE:
                        raise ERROR_METHOD1(
                            "🔄 Method 2 -> Step 1B: Failed to transcribe lyrics")

                    # clear M2_LYRICS_SAMPLE_CHUNKS because M2_LYRICS_SAMPLE changed
                    if M2_LYRICS_SAMPLE_CHUNKS:
                        M2_LYRICS_SAMPLE_CHUNKS = []
                    # clear M2_CANDIDATES because M2_LYRICS_SAMPLE changed
                    if M2_CANDIDATES:
                        M2_CANDIDATES = []

                else:
                    log(f"🔄 Method 2 -> Step 1 : Reusing cached Lyrics Sample")

                # Split Lyrics Sample into chunks
                if not M2_LYRICS_SAMPLE_CHUNKS:
                    log(f"🔄 Method 2 -> Step 2 : Generating Lyrics Sample Chunks")
                    M2_LYRICS_SAMPLE_CHUNKS = split_lyrics_to_chunks(
                        M2_LYRICS_SAMPLE, max_words=8, max_chunks=10)
                else:
                    log(f"🔄 Method 2 -> Step 2 : Reusing cached Lyrics Sample Chunks")

                # Search the web for possible Track info using the lyrics chunks
                if not M2_CANDIDATES:
                    log(f"🔄 Method 2 -> Step 3 : Querying the Web for possible Track matches")
                    search_tracks_results = asyncio.run(
                        search_tracks(M2_LYRICS_SAMPLE_CHUNKS))

                    if not search_tracks_results:
                        raise ERROR_METHOD2(
                            f"🔄 Method 2 -> Step 3 : Failed to find Matches from the Web")

                    # update candidates to match default dict
                    M2_CANDIDATES = [{**DEFAULT_CANDIDATE_METHOD2, **track}
                                     for track in search_tracks_results]

                else:
                    log(f"🔄 Method 2 -> Step 3 : Reusing cached candidates")

                # score candidates by lyrics match
                log(f"🔄 Method 2 -> Step 4 : Scoring by Lyrics Match..")
                M2_CANDIDATES = score_candidates_by_lyrics_match(
                    M2_LYRICS_SAMPLE, M2_CANDIDATES)

                # boost confidence by re-occurrences
                M2_CANDIDATES = boost_confidence_by_occurrence(M2_CANDIDATES)

                # check if we have 99% match
                if M2_CANDIDATES[0].get("confidence", 0.0) >= 99.99:
                    if len(M2_CANDIDATES) > 1 and M2_CANDIDATES[1].get("confidence", 0.0) >= 99.99:
                        STATUS_M2 = "review"
                    else:
                        # Single 99% match - tag both title and artist
                        tag_title = M2_CANDIDATES[0]['title']
                        tag_artist = M2_CANDIDATES[0]['artist']
                        STATUS_M2 = "done"
                        log(f"{TRACK.name} -> Title: {tag_title} Artist: {tag_artist}", "d")

                        # Call with correct arguments
                        success = tag_file(TRACK, tag_title, tag_artist)

                        if not success:
                            log(f"Failed to tag file", "w")

                        continue

                if METHOD == 2:
                    if M2_CANDIDATES[0].get("confidence", 0.0) <= 0:
                        raise ERROR_METHOD2(
                            f"🔄 Method 2 -> 4 : Failed to find candidate with matching score")

                # Set status for review
                STATUS_M2 = "review"

            except ERROR_METHOD2 as e:
                log(e)
                STATUS_M2 = 'failed'
            finally:
                save_update(
                    {
                        "title": M2_CANDIDATES[0]["title"] if STATUS_M2 == "done" else "",
                        "artist": M2_CANDIDATES[0]["artist"] if STATUS_M2 == "done" else "",
                        "status": STATUS_M2,
                        "method": "Lyrics Search",
                        "detected_firstword_time": DETECTED_FIRSTWORD_TIME,
                        "lyrics_firstword_detection_duration": args.lyrics_firstword_detection_duration,
                        "m2_lyrics_sample": M2_LYRICS_SAMPLE,
                        "m2_lyrics_sample_chunks": M2_LYRICS_SAMPLE_CHUNKS,
                        "m2_lyricsearch_sample_duration": args.m2_lyricsearch_sample_duration,
                        "m2_candidates": M2_CANDIDATES
                    },
                    TRACK_ITEM, PROCESS_DATA, PROCESS_PATH
                )

                if METHOD == 2:  # standalone
                    if STATUS_M2 == "failed":
                        log(f"❌ {TRACK.name} -> Failed to find Matches")
                    elif STATUS_M2 == "review":
                        log(
                            f"📒 {TRACK.name} -> Is marked for review, possible title:{M2_CANDIDATES[0]['title']} with confidence of {M2_CANDIDATES[0]['confidence']}")
                        candidates_filtered = [
                            {
                                "confidence": candidate.get("confidence", 0.0),
                                "occurrences": candidate.get("occurrences", 1),
                                "title": candidate.get("title", ""),
                                "artist": candidate.get("artist", ""),
                                "lyrics": candidate.get("lyrics", ""),
                                "method": "Lyrics Search",
                            }
                            for candidate in M2_CANDIDATES
                        ]
                        save_update(
                            {
                                "candidates": candidates_filtered
                            },
                            TRACK_ITEM, PROCESS_DATA, PROCESS_PATH
                        )
                    # next loop
                    continue

                if STATUS_M2 == "done":
                    # next loop
                    continue
        log(f"🔃 Final    -> Calcutating Score Results from Method 1 & 2")

        # check if it is possible to Calculate from Two Methods
        if not M1_CANDIDATES and not M2_CANDIDATES:
            log(
                f"🔃 Final    -> Failed to Calculate! \nMethod 1: {len(M1_CANDIDATES)} \nMethod 2: {len(M2_CANDIDATES)}")
            continue

        # combine candidates
        candidates_casual_merged = [*M1_CANDIDATES, *M2_CANDIDATES]

        # create a unified candidates format
        candidates_filtered = [
            {
                "confidence": candidate.get("confidence", 0.0),
                "occurrences": candidate.get("occurrences", 1),
                "title": candidate.get("title", ""),
                "artist": candidate.get("artist", ""),
                "lyrics": candidate.get("lyrics", ""),
                "method": "Fingerprint" if "segments" in candidate else "Lyrics Search",
            }
            for candidate in candidates_casual_merged
        ]

        # boost confidence
        candidates_boosted = boost_confidence_by_occurrence(
            candidates_filtered)

        # filter out zero confidence
        candidates_filter = 40.00  # filter 50.00 and above
        candidates_filtered_non_zero = [
            c for c in candidates_boosted if c.get("confidence", 0.0) >= candidates_filter]

        if candidates_filtered_non_zero:
            STATUS = "review"
            # check if we have 99% match
            if candidates_filtered_non_zero[0].get("confidence", 0.0) >= 99.99:
                if not len(candidates_filtered_non_zero) > 1 or not candidates_filtered_non_zero[1].get("confidence", 0.0) >= 99.99:
                    # Single 99% match - tag both title and artist
                    tag_title = candidates_filtered_non_zero[0]['title']
                    tag_artist = candidates_filtered_non_zero[0]['artist']
                    STATUS_M2 = "done"
                    log(f"{TRACK.name} -> Title: {tag_title} Artist: {tag_artist}", "d")

                    # Call with correct arguments
                    success = tag_file(TRACK, tag_title, tag_artist)

                    if not success:
                        log(f"Failed to tag file", "w")

                    STATUS = "done"

                STATUS = "review"

            if STATUS == "review":
                log(
                    f"📒 {TRACK.name} -> Marked For Review, possible title:{candidates_filtered_non_zero[0]['title']} with confidence of {candidates_boosted[0]['confidence']}")

            save_update(
                {
                    "title": candidates_filtered_non_zero[0]["title"] if STATUS == "done" else "",
                    "artist": candidates_filtered_non_zero[0]["artist"] if STATUS == "done" else "",
                    "status": STATUS,
                    "method": "Candidate Scoring",
                    "candidates": candidates_filtered_non_zero
                },
                TRACK_ITEM, PROCESS_DATA, PROCESS_PATH
            )
        else:
            log(f"{TRACK.name} Failed to match.", "f")
            save_update(
                {
                    "status": "failed",
                    "status_message": "Failed to match"
                },
                TRACK_ITEM, PROCESS_DATA, PROCESS_PATH
            )

    if marked_for_review:
        review_track_data(
            PROCESS_LIST,
            PROCESS_DATA,
            PROCESS_PATH
        )


if __name__ == "__main__":
    if "--cli-review-worker" in sys.argv:
        i = sys.argv.index("--cli-review-worker")

        queue = Path(sys.argv[i + 1])
        result = Path(sys.argv[i + 3])
        control = Path(sys.argv[i + 5])

        run_cli_worker_persistent(queue, result, control)
        sys.exit(0)
    main()
