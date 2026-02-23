# Smart Music Title Parser

import argparse
import json
import asyncio
import subprocess
import time
import re
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Any

from lrclib import AsyncLRCLibClient
from deezerlib import AsyncDeezerClient 

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

# ---------------- CONFIG ----------------
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
PROCESS_FILE = "process.json"
LANGUAGE = "en"

# Helper
# ---------------- LOGGING ----------------
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Helper Functions
def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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

def combine_candidates(candidates):
    grouped = {}
    
    for c in candidates:
        # Try track_id first, then (title, artist) fallback
        key = c.get("track_id") or (c.get("title"), c.get("artist"))
        
        if key is None:
            continue  # Skip if no grouping key available
        
        if key not in grouped:
            grouped[key] = {**c, "occurrences": 1}
        else:
            grouped[key]["occurrences"] += 1
    
    return list(grouped.values())

def boost_confidence_by_occurrence(
    input_list: List[Dict[str, Any]], 
    key_fields: List[str] = None,
    boost_percent: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Returns a unique list where confidence is boosted by boost_percent 
    for each occurrence of the same (title, artist).
    
    Args:
        input_list: List of dictionaries with at least 'title', 'artist', and 'confidence' keys
        key_fields: List of field names to use for grouping.
                    If multiple fields, creates tuple key.
                    Example: ['track_id'] or ['title', 'artist']
        boost_percent: Percentage to boost confidence per occurrence (default: 10%)
    
    Returns:
        List of unique dictionaries with boosted confidence and updated occurrences count
    """
    # Default to track_id or title+artist
    if key_fields is None:
        key_fields = ['track_id', 'title', 'artist']  # Priority order

    def create_key(item: Dict[str, Any]) -> Union[str, tuple, None]:
        """Create key from specified fields"""
        values = []
        for field in key_fields:
            if (value := item.get(field)) is not None:
                values.append(value)
                # If track_id is found, use it alone (priority)
                if field == 'track_id' and value:
                    return value
        
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    # Dictionary to group by (title, artist)
    grouped_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    # First pass: group items and calculate base values
    for item in input_list:
        #key = item.get('track_id') or (item.get('title'), item.get('artist'))
        key = create_key(item)
        if key is None:
            continue
        if key not in grouped_data:
            # First occurrence - start with the item's values
            confidence = float(item.get('confidence',0))
            grouped_data[key] = {
                'title': item['title'],
                'artist': item['artist'],
                'confidence': confidence,
                'occurrences': 1,  # Count this occurrence
                'base_confidence': confidence,  # Keep original for calculation
                # Preserve other keys from first occurrence
                **{k: v for k, v in item.items() if k not in ['title', 'artist', 'confidence', 'occurrences']}
            }
        else:
            # Additional occurrence - update count and confidence
            grouped_data[key]['occurrences'] += 1

            # merge segments
            if 'segment' in grouped_data[key] or 'segment' in item:
                grouped_data[key]['segment'] = [*grouped_data[key].get('segment',[]), *item.get('segment',[])]
            
            # If item has occurrences count, add it (preserve previous value if exists)
            if 'occurrences' in item:
                grouped_data[key]['occurrences'] = max(
                    grouped_data[key]['occurrences'],
                    item.get('occurrences', 1)
                )
            
            # Update other keys if they exist in current item (optional - depends on your needs)
            for k, v in item.items():
                if k not in ['title', 'artist', 'confidence', 'occurrences'] and k not in grouped_data[key]:
                    grouped_data[key][k] = v
    
    # Second pass: apply confidence boost based on occurrences
    result = []
    for data in grouped_data.values():
        # Calculate boosted confidence
        occurrences = data['occurrences']
        boost_multiplier = 1.0 + (boost_percent / 100.0) * (occurrences - 1)
        boosted_confidence = data['base_confidence'] * boost_multiplier
        
        # Cap confidence at 1.0 (100%) if using normalized confidence
        boosted_confidence = min(boosted_confidence, 100.0)
        
        # Create result item
        result_item = {
            'title': data['title'],
            'artist': data['artist'],
            'confidence': round(boosted_confidence, 4),
            'occurrences': occurrences,
            # Include all other keys except internal ones
            **{k: v for k, v in data.items() if k not in ['confidence']}
        }
        
        result.append(result_item)
    
    return sorted(result, key=lambda x: -x.get("confidence", 0)) if result else []


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


# Methods
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
            result.update({"segment":[start,dur]})
            results.append(result)
            # break when confidence is match
            if result.get("confidence", 0) >= 98.0:
                break
    
    return sorted(results, key=lambda x: -x.get("confidence", 0)) if results else []



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


async def deezer_search(query: str,debug=False) -> List[Dict]:
    client = AsyncDeezerClient(max_retries=3)
    
    data = await client.search(query,limit=4)

    out = []
    for d in data.get("data", []):
        out.append({
            "track_id": d.get("id"),
            "title": d.get("title_short") if "title_short" in d else d.get("title"),
            "artist": d.get("artist", {}).get("name"),
            "album": d.get("album",{}).get("title"),
            "duration": d.get("duration"),
            "isrc": d.get("isrc"),
            "search_kword":query
        })

    return out

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
    pattern = r'(?:\s*\n\s*|\s+and\s+)'
    parts = re.split(pattern, long_string)
    words = [w for part in parts for w in part.strip().split()]
    results = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return results[:max_chunks]



def split_lyrics_to_chunks_old(text: str, max_words: int = 10, max_chunks: int = 3) -> List[str]:
    """
    Split lyrics into chunks using commas and newlines.
    Intelligently combine small chunks while respecting max_words as a soft limit.
    Returns at most max_chunks chunks.

    Args:
        text: Lyrics text to split.
        max_words: Soft word limit per chunk.
        max_chunks: Maximum number of chunks to return.

    Returns:
        List of processed text chunks.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Initial split
    split_pattern = r'(?:\s*\n\s*|\s+and\s+)'
    raw_chunks = [c.strip() for c in re.split(split_pattern, text) if c.strip()]

    parts: List[str] = []
    current_words = 0

    for chunk in raw_chunks:
        chunk_words = len(chunk.split())

        if not parts:
            parts.append(chunk)
            current_words = chunk_words
            continue

        # Soft-limit combining logic
        if current_words + chunk_words <= max_words or current_words < max_words // 2:
            parts[-1] = f"{parts[-1]}, {chunk}"
            current_words += chunk_words
        else:
            parts.append(chunk)
            current_words = chunk_words

    # Remove extremely short parts
    parts = [p for p in parts if len(p.split()) >= 3]

    return parts[:max_chunks]


async def lrclib_fetch_lyrics(track:str,artist:str,album:str,duration:int, debug=False) -> str:
    client = AsyncLRCLibClient(max_retries=3,ssl_mode="httpx")
    
    lyrics = await client.get(track,artist,album,duration)

    if lyrics and "plainLyrics" in lyrics and lyrics["plainLyrics"] != None:
        # print(lyrics["plainLyrics"])
        return " ".join(lyrics["plainLyrics"].split())
    return ""

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--target-filename", type=str, default=None)
    p.add_argument("--duration-method-fingerprint", type=float, default=30)
    p.add_argument("--duration-method-lyrics", type=float, default=30)
    p.add_argument("--force-clean", action="store_true")
    p.add_argument("--reset-status",action="store_true")
    p.add_argument("--reset-candidates-fingerprint",action="store_true")
    p.add_argument("--reset-candidates-lyrics",action="store_true")
    p.add_argument("--reset-track-lyrics",action="store_true")
    p.add_argument("--reset-track-lyrics-chunks",action="store_true")
    p.add_argument("--review", action="store_true")
    args = p.parse_args()

    # process.json
    directory = Path(args.dir).resolve()
    process_path = directory / PROCESS_FILE

    process_lists = directory.iterdir()

    default_processData = {
        "path":"",
        "title":"",
        "artist":"",
        "status":"",
        "status_message":"",
        "method":"",
        "fingerprint_sample_duration": args.duration_method_fingerprint,
        "lyrics_sample":"",
        "lyrics_sample_duration": args.duration_method_lyrics,
        "lyrics_sample_chunks":[],
        "candidates":[],
        "candidates_fingerprint":[],
        "candidates_lyricsSearch":[]
    }

    # clean up process.json
    if args.force_clean and process_path.exists():
        process_path.unlink(missing_ok=True) # delete the file


    # get data from process.json
    process_data = json.loads(content) if process_path.exists() and (content := process_path.read_text()) else {}

    # 
    if not isinstance(process_data, dict):
        process_data = {}

    # data manipulation
    updates = {}

    if args.reset_status:
        updates['status'] = ""

    if args.reset_candidates_fingerprint:
        updates['candidates_fingerprint'] = []

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

    # process only one file
    if args.target_filename:
        process_lists = [x for x in process_lists if x.name == args.target_filename]

    # main loop
    for f in process_lists:
        # skip non audio files
        if f.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        log(f"Processing {f.name}..")

        # get the track entry from process_data if any
        track_item = process_data.get(f.name, {})
        if not track_item:
            track_item = process_data[f.name] = {
                **default_processData,
                "path": str(f)
            }
            save_to_process(process_data,process_path)

        # skip with done status
        if track_item.get("status") == "done":
            log(f"✔️ {f.name} -> already done! title:{track_item.get('title')}")
            continue
        

        # Start Method 1
        log(f"Method 1 : Fingerprinting")

        candidates_fingerprint = track_item.get("candidates_fingerprint",[])

        # Fingerprint Track
        log(f"Method 1 -> Shazam analying fingerprint...")
        if not candidates_fingerprint and not track_item.get("fingerprint_sample_duratioin",0) == args.duration_method_fingerprint:
            candidates_fingerprint = asyncio.run(fingerprint_track(f, args.duration_method_fingerprint))
            save_update({
            "candidates_fingerprint":candidates_fingerprint
            },track_item,process_data,process_path)
        else:
            log(f"Method 1 -> Reusing cached fingerprint data")
        

        # Process Results
        boosted_candidates_fingerprint = []
        if candidates_fingerprint:
            # boost by re-occurance
            boosted_candidates_fingerprint=boost_confidence_by_occurrence(candidates_fingerprint,["title","aritist"],20) # boost by re-occurances
            
            # check if we have 99% match
            if boosted_candidates_fingerprint[0].get("confidence") >= 99:
                log(f"✔️ {f.name} -> title:{boosted_candidates_fingerprint[0]['title']}")
                tag_file(f, boosted_candidates_fingerprint[0]["title"], boosted_candidates_fingerprint[0]["artist"])
                # update process file
                save_update({
                    "title":boosted_candidates_fingerprint[0]["title"],
                    "artist":boosted_candidates_fingerprint[0]["artist"],
                    "status": "done",
                    "method": "fingerprint",
                    "candidates":boosted_candidates_fingerprint,
                    "candidates_fingerprint":candidates_fingerprint
                },track_item,process_data,process_path)
                continue
        
        # update process to review
        save_update({"status":"review"},track_item,process_data,process_path)


        # Method 2 : Lyrics Search
        log(f"Method 2 : Lyrics Search")

        # re-use cached lyrics
        sample_lyrics = track_item.get("lyrics_sample", "")
        if sample_lyrics and track_item.get("lyrics_sample_duration") == args.duration_method_lyrics:
            log(f"Method 2 -> Reusing Cached Whisper Lyrics with duration of {args.duration_method_lyrics}")
        else:
            log(f"Method 2 -> Whisper analysing sample...")
            whisper_first_word = whisper_detect_first_word(model, f)
            if not whisper_first_word:
                if not track_item.get("status","") == "review":
                    log(f"❌ {f.name} -> Failed to detect first word of track")
                    save_update({
                        "status":"failed",
                        "status_message":"Whisper Failed to detect first word track"
                    },track_item,process_data,process_path)
                    continue
                log(f"📒 {f.name} -> Failed Method 2 First word detection, Marked for Method 1 review")
                save_update({
                    "status":"review",
                    "status_message":"Failed Method 2, For Method 1 review"
                },track_item,process_data,process_path)
                continue

            # transcrib lyrics sample
            log(f"Method 2 -> Whisper transcribing sample...")
            sample_lyrics = whisper_transcribe_from_start(model, f, whisper_first_word, args.duration_method_lyrics)

            if not sample_lyrics:
                if not track_item.get("status","") == "review":
                    log(f"❌ {f.name} -> Failed to transcribe lyrics sample..")
                    save_update({
                        "status":"failed",
                        "status_message":"Whisper Failed to transcribe lyrics"
                    },track_item,process_data,process_path)
                    continue
                log(f"📒 {f.name} -> Failed Method 2 Lyrics Transcribe, Marked for Method 1 review")
                save_update({
                    "status":"review",
                    "status_message":"Failed Method 2, For Method 1 review"
                },track_item,process_data,process_path)
                continue

            # save lyrics to process
            save_update({
                "lyrics_sample" : sample_lyrics,
                "lyrics_sample_duration" : args.duration_method_lyrics
            },track_item,process_data,process_path)

        # Method 2 : Deezer Lyrics Search
        
        # init variables
        sample_lyrics_chunks = track_item.get("lyrics_sample_chunks",[])
        candidates_lyricsSearch = track_item.get("candidates_lyricsSearch") or []


        # search cache for deezer search candidates or init
        if sample_lyrics_chunks:
            log(f"Method 2 -> Reusing sample lyrics chunks...")
        else:
            # split lyrics to chuncks for deezer search
            sample_lyrics_chunks = split_lyrics_to_chunks(sample_lyrics,max_words=10,max_chunks=20)
            save_update({
                "lyrics_sample_chunks":sample_lyrics_chunks,
            },track_item,process_data,process_path)

        # search deezer for lyrics match
        if candidates_lyricsSearch:
            log(f"Method 2 -> Reusing Lyrics Search Candidates...")
        else:
            log(f"Method 2 -> Deezer searching for track matching lyrics chunks...")
            async def _search_deezer_tracks():
                tasks = [deezer_search(c) for c in sample_lyrics_chunks]
                results = await asyncio.gather(*tasks)
                return [item for sublist in results for item in sublist]

            candidates_lyricsSearch = asyncio.run(_search_deezer_tracks())

            save_update({
                "candidates_lyricsSearch":candidates_lyricsSearch,
            },track_item,process_data,process_path)
            
            if not candidates_lyricsSearch:
                if candidates_fingerprint:
                    log(f"📒 {f.name} -> Failed to find deezer matches with lyrics chunks, Marked for Method 1 Review")
                    save_update({
                        "status_message":"Failed to find deezer matches with lyrics chunks, Marked for Method 1 Review"
                    },track_item,process_data,process_path)
                    continue

                log(f"❌ {f.name} -> Failed to find deezer matches with lyrics chunks")
                save_update({
                    "status":"failed",
                    "status_message":"Deezer Failed to find deezer matches with lyrics chunks"
                },track_item,process_data,process_path)
                continue

            # boost by re-occurance
            #candidates_lyricsSearch=boost_confidence_by_occurrence(candidates_lyricsSearch,5.0) # boost by re-occurances
        
        # combine similar keys
        candidates_lyricsSearch = combine_candidates(candidates_lyricsSearch)

        log(f"Method 2 -> LRCLib searching for Deezer track lyrics.. ")
        # fetch lyrics using titles, artist, album, duration

        # Find all candidates needing lyrics
        candidates_needing_lyrics = [
            c for c in candidates_lyricsSearch
            if "lyrics" not in c and 
            all(key in c for key in ["title", "artist", "album", "duration"])
        ]

        # Fetch all missing lyrics concurrently
        async def fetch_all_lyrics():
            tasks = []
            for c in candidates_needing_lyrics:
                task = lrclib_fetch_lyrics(
                    c["title"], 
                    c["artist"],
                    c["album"],
                    c["duration"]
                )
                tasks.append(task)
            
            # Run all lyric fetches concurrently
            lyrics_list = await asyncio.gather(*tasks)
            
            # Assign lyrics back to candidates
            for candidate, lyrics in zip(candidates_needing_lyrics, lyrics_list):
                candidate["lyrics"] = lyrics
        
        # Execute the async fetch
        asyncio.run(fetch_all_lyrics())
        
        # sort by
        candidates_lyricsSearch = list(sorted(candidates_lyricsSearch, key=lambda x: (x['title'].lower(), x['artist'].lower())))
        # score candidates
        scored_candidates_lyricsSearch = score_candidates_by_lyrics_match(sample_lyrics, candidates_lyricsSearch)
        # boost candidates
        boosted_candidates_lyricsSearch = boost_confidence_by_occurrence(scored_candidates_lyricsSearch,['track_id'])

        # save candidates_lyricsSearch
        save_update({
            "candidates_lyricsSearch": scored_candidates_lyricsSearch
        },track_item,process_data,process_path)

        # compute all candidates scores
        log(f"Calculating Score Results...")
        
        candidates = [*boosted_candidates_fingerprint, *boosted_candidates_lyricsSearch]

        filtered_candidates = sorted([c for c in candidates if c.get('confidence',0) >= 50],key=lambda x: -x.get('confidence'))

        # boost by occurances
        boosted_candidates = boost_confidence_by_occurrence(filtered_candidates,['title'],10)
        save_update({
            "candidates":boosted_candidates
        },track_item,process_data,process_path)

        
        if not boosted_candidates:
            log(f"❌ {f.name} -> Failed to find Matches")
            save_update({
                "status":"failed",
                "status_message":"Failed to find Matches"
            },track_item,process_data,process_path)
            continue

        if boosted_candidates[0]["confidence"] > 98:
            # tag file is score is 99%
            log(f"✔️ {f.name} -> title:{boosted_candidates[0]['title']}")
            tag_file(f, boosted_candidates[0]["title"], boosted_candidates[0]["artist"])
            save_update({
                "title":boosted_candidates[0]["title"],
                "artist":boosted_candidates[0]["artist"],
                "status":"done"
            },track_item,process_data,process_path)
        else:
            log(f"📒 {f.name} -> Is Marked For Review, possible title:{boosted_candidates[0]['title']} with confidence of {boosted_candidates[0]['confidence']}")


if __name__ == "__main__":
    main()