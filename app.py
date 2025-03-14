from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import librosa
import numpy as np
import shutil
import uuid
import tempfile
import logging
import traceback
from audioread.exceptions import NoBackendError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global storage for tracks and session data
sessions = {}

def detect_bpm(y_harmonic, sr):
    if len(y_harmonic) == 0:
        logger.warning("Harmonic signal is empty, check audio quality.")
        return 0.0  

    onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr)

    # Detectăm BPM cu două metode și alegem cea mai apropiată de medie
    bpm_librosa = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    bpm_librosa_2 = librosa.beat.tempo(y=y_harmonic, sr=sr)

    # Alegem BPM-ul cel mai probabil folosind un filtru inteligent
    bpm_values = [bpm_librosa[0], bpm_librosa_2[0]]
    bpm_final = min(bpm_values, key=lambda x: abs(x - 125))  # Ne asigurăm că nu e dublat/înjumătățit

    logger.debug(f"Final BPM Calculation: {bpm_final}")
    return round(float(bpm_final), 2)

def analyze_audio(file_path):
    """Analizează un fișier audio pentru a extrage BPM, key (Camelot notation), și energy."""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=44100, mono=True)

        # --- PRE-PROCESARE AUDIO ---
        y_normalized = librosa.util.normalize(y)  # Normalizare volum
        if y_normalized is None:
            raise ValueError("y_normalized is not defined")
        y_harmonic, _ = librosa.effects.hpss(y_normalized)  # HPSS separă armonicul și percutantul
        logger.debug("Audio normalization and HPSS completed")

        # --- BPM DETECTION MAI PRECIS ---
        bpm = detect_bpm(y_harmonic, sr)
        logger.debug(f"Detected BPM for {file_path}: {bpm}")

        # --- DETECȚIE KEY (TONALITATE) CU LIBROSA ---
        def detect_key(y, sr):
            try:
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                if chroma.shape[1] == 0:
                    raise ValueError("Chroma feature extraction failed")

                key_data = np.mean(chroma, axis=1)
                key_index = int(np.argmax(key_data))

                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                camelot_map = {
                    "C": "8B", "C#": "3B", "D": "10B", "D#": "5B", "E": "12B", "F": "7B",
                    "F#": "2B", "G": "9B", "G#": "4B", "A": "11B", "A#": "6B", "B": "1B"
                }

                detected_key = key_names[key_index]
                camelot_key = camelot_map.get(detected_key, "Unknown")

                # Verificare: Dacă Key-ul apare la prea multe piese, poate fi greșit
                if camelot_key == "8B" and np.mean(key_data) < 0.1:
                    logger.warning(f"Potentially incorrect Key detected: {camelot_key}, adjusting...")
                    return "Unknown"

                return camelot_key

            except Exception as e:
                logger.error(f"Key detection error: {e}")
                return "Unknown"

        camelot_key = detect_key(y_harmonic, sr)
        logger.debug(f"Detected Key for {file_path} using librosa: {camelot_key}")

        # --- ENERGY CALCULATION ---
        energy = float(np.mean(librosa.feature.rms(y=y_normalized))) * 10
        energy_variance = float(np.var(librosa.feature.rms(y=y_normalized))) * 5
        final_energy = round(energy + energy_variance, 3)
        logger.debug(f"Energy calculated: {final_energy}")

        logger.debug(f"File: {file_path}, BPM: {bpm}, Key: {camelot_key}, Energy: {final_energy:.3f}")

        return {
            "bpm": float(bpm),
            "key": str(camelot_key),
            "energy": float(final_energy)
        }

    except Exception as e:
        logger.error(f"General error: {e}")
        logger.error(traceback.format_exc())  # Afișează stack trace-ul complet pentru debugging
        return {"error": f"Failed to process {os.path.basename(file_path)}: {str(e)}"}

def compatibility_score(t1, t2):
    """Calculates compatibility score between two tracks."""
    try:
        if not t1 or not t2:
            return float("inf")
        bpm_diff = abs(t1["bpm"] - t2["bpm"]) * 0.5
        key_diff = 0 if t1["key"] == t2["key"] else 10
        energy_diff = abs(t1["energy"] - t2["energy"]) * 20
        return bpm_diff + key_diff + energy_diff
    except Exception as e:
        logger.error(f"Error in compatibility_score: {str(e)}")
        return float("inf")

@app.route('/')
def index():
    """Renders the main page with a unique session ID."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"tracks": [], "temp_dir": tempfile.mkdtemp()}
    return render_template('index.html', session_id=session_id)

@app.route('/upload/<session_id>', methods=['POST'])
def upload_folder(session_id):
    """Handles folder upload for a specific session, skipping non-audio files."""
    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    session = sessions[session_id]
    session["tracks"].clear()

    if 'folder' not in request.files:
        return jsonify({"error": "No folder uploaded"}), 400

    uploaded_files = request.files.getlist('folder')
    valid_extensions = ('.mp3', '.wav', '.flac', '.aiff')
    supported_count = 0

    for file in uploaded_files:
        if not file.filename:
            logger.debug("Skipping file with empty filename")
            continue
        normalized_filename = file.filename.lower()
        if normalized_filename.startswith('._'):
            logger.debug(f"Skipping macOS metadata file: {file.filename}")
            continue
        if not normalized_filename.endswith(valid_extensions):
            logger.debug(f"Skipping file with invalid extension: {file.filename}")
            continue

        # **Elimină prefixul "Testt_" din numele fișierului**
        filename = file.filename.replace("Testt_", "")  # Folosește doar numele fișierului original
        safe_filename = filename.replace('/', '_').replace('\\', '_')  # Asigură-te că e sigur pentru sistem
        file_path = os.path.join(session["temp_dir"], safe_filename)
        file.save(file_path)
        session["tracks"].append((safe_filename, None))
        supported_count += 1
        logger.debug(f"Accepted file: {safe_filename}")

    if supported_count == 0:
        return jsonify({"error": "No supported audio files (.mp3, .wav, .flac, .aiff) found"}), 400

    return jsonify({"message": f"Uploaded {supported_count} valid audio files", "tracks": [t[0] for t in session["tracks"]]})

@app.route('/analyze/<session_id>', methods=['GET'])
def analyze_tracks(session_id):
    """Analyzes tracks for a specific session with progress updates."""
    if session_id not in sessions:
        return jsonify({"error": "Invalid session"}), 400

    session = sessions[session_id]
    tracks = session["tracks"]
    if not tracks:
        return jsonify({"error": "No tracks uploaded"}), 400
    
    results = []
    for idx, (file, _) in enumerate(tracks):
        file_path = os.path.join(session["temp_dir"], file)
        analysis = analyze_audio(file_path)
        if "error" in analysis:
            tracks[idx] = (file, None)
            results.append({"file": file, "status": "error", "message": analysis["error"]})
        else:
            tracks[idx] = (file, analysis)
            results.append({
                "file": file,
                "status": "success",
                "bpm": float(analysis["bpm"]),
                "key": str(analysis["key"]),
                "energy": float(analysis["energy"]),
            })
    
    session["tracks"] = tracks
    return jsonify({"results": results})

@app.route('/export/<session_id>', methods=['POST'])
def export_playlist(session_id):
    """Exports sorted playlist as a ZIP file for download."""
    if session_id not in sessions:
        logger.error(f"Session {session_id} not found")
        return jsonify({"error": "Invalid session"}), 400

    session = sessions[session_id]
    tracks = session["tracks"]
    if not tracks or any(t[1] is None for t in tracks):
        logger.error("Tracks not analyzed or empty")
        return jsonify({"error": "Analyze tracks first"}), 400
    
    try:
        # Sort tracks by compatibility
        sorted_tracks = [tracks.pop(0)]
        explanations = [f"Playlist starts with track: {sorted_tracks[0][0]}"]
        
        while tracks:
            last_track = sorted_tracks[-1][1]
            tracks.sort(key=lambda t: compatibility_score(last_track, t[1]))
            next_track = tracks.pop(0)
            sorted_tracks.append(next_track)
            explanations.append(
                f"{next_track[0]} selected after {sorted_tracks[-2][0]} "
                f"(BPM diff: {abs(last_track['bpm'] - next_track[1]['bpm']):.2f}, "
                f"Key {'match' if last_track['key'] == next_track[1]['key'] else 'mismatch'}, "
                f"Energy diff: {abs(last_track['energy'] - next_track[1]['energy']):.3f})"
            )
        
        # Create a temporary folder for export
        export_folder_name = f"SortedPlaylist_{uuid.uuid4().hex}"
        export_folder = os.path.join(tempfile.gettempdir(), export_folder_name)
        os.makedirs(export_folder, exist_ok=True)
        logger.debug(f"Created export folder: {export_folder}")
        
        # Copy files to the export folder with proper naming
        for idx, (file, _) in enumerate(sorted_tracks, start=1):
            src = os.path.join(session["temp_dir"], file)
            # Remove folder prefix from the filename (e.g., "Testt_" in "Testt_Gabi.B...")
            base_name = file.split('_', 1)[-1] if '_' in file else file
            safe_file = base_name.replace(' ', '_').replace('[', '').replace(']', '')
            dst = os.path.join(export_folder, f"{str(idx).zfill(2)}.{safe_file}")
            
            if not os.path.exists(src):
                logger.error(f"Source file {src} does not exist")
                return jsonify({"error": f"Source file {file} not found"}), 500
            
            logger.debug(f"Copying {src} to {dst}")
            shutil.copy2(src, dst)
        
        # Write explanations
        explanation_file = os.path.join(export_folder, "playlist_explanation.txt")
        with open(explanation_file, "w") as f:
            f.write("\n".join(explanations))
        logger.debug(f"Wrote explanations to {explanation_file}")
        
        # Create a ZIP file
        zip_path = os.path.join(tempfile.gettempdir(), export_folder_name)
        shutil.make_archive(zip_path, 'zip', export_folder)
        logger.debug(f"Created ZIP file: {zip_path}.zip")
        
        # Clean up the temporary export folder (but keep the ZIP)
        shutil.rmtree(export_folder, ignore_errors=True)
        
        return jsonify({"message": "Playlist exported as ZIP", "zip_file": f"{export_folder_name}.zip"})
    
    except Exception as e:
        logger.error(f"Error during export: {str(e)}", exc_info=True)
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/download/<zip_file>')
def download_zip(zip_file):
    """Sends the ZIP file for download."""
    try:
        zip_path = os.path.join(tempfile.gettempdir(), zip_file)
        if not os.path.exists(zip_path):
            logger.error(f"ZIP file {zip_path} not found")
            return jsonify({"error": "ZIP file not found"}), 404
        
        logger.debug(f"Sending ZIP file: {zip_path}")
        response = send_from_directory(tempfile.gettempdir(), zip_file, as_attachment=True)
        
        # Clean up the ZIP file after sending
        os.remove(zip_path)
        logger.debug(f"Cleaned up ZIP file: {zip_path}")
        
        return response
    except Exception as e:
        logger.error(f"Error during download: {str(e)}", exc_info=True)
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup(session_id):
    """Cleans up session data."""
    try:
        if session_id in sessions:
            session = sessions[session_id]
            shutil.rmtree(session["temp_dir"], ignore_errors=True)
            del sessions[session_id]
            logger.debug(f"Cleaned up session {session_id}")
        return jsonify({"message": "Session cleaned up"})
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)