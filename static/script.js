function logMessage(message) {
    const logList = document.getElementById('logList');
    const li = document.createElement('li');
    li.textContent = message;
    logList.appendChild(li);
}

function uploadFolder() {
    if (!sessionId) {
        logMessage("Error: Session ID is not defined. Please reload the page.");
        return;
    }

    const folderInput = document.getElementById('folderInput');
    const files = folderInput.files;
    if (!files.length) {
        alert("Please select a folder with audio files!");
        return;
    }

    const formData = new FormData();
    for (let file of files) {
        formData.append('folder', file);
    }

    fetch(`/upload/${sessionId}`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logMessage(data.error);
        } else {
            logMessage(data.message);
            data.tracks.forEach(track => logMessage(`Found: ${track}`));
        }
    })
    .catch(error => logMessage(`Upload error: ${error}`));
}

function analyzeTracks() {
    if (!sessionId) {
        logMessage("Error: Session ID is not defined. Please reload the page.");
        return;
    }

    document.getElementById('progress').innerHTML = '<div class="progress-bar" id="progressBar"></div>';
    const progressBar = document.getElementById('progressBar');

    fetch(`/analyze/${sessionId}`)
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logMessage(data.error);
            return;
        }

        let completed = 0;
        data.results.forEach(result => {
            completed++;
            const percentage = (completed / data.results.length) * 100;
            progressBar.style.width = `${percentage}%`;

            if (result.status === 'success') {
                logMessage(`Analyzed: ${result.file} → BPM: ${result.bpm}, Key: ${result.key}, Energy: ${result.energy}`);
            } else {
                logMessage(result.message);
            }
        });
        logMessage("Analysis complete!");
    })
    .catch(error => logMessage(`Analysis error: ${error}`));
}

function exportPlaylist() {
    if (!sessionId) {
        logMessage("Error: Session ID is not defined. Please reload the page.");
        return;
    }

    fetch(`/export/${sessionId}`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            logMessage(data.error);
            return;
        }

        logMessage(data.message);
        data.playlist.forEach((track, idx) => {
            logMessage(`${String(idx + 1).padStart(2, '0')}. ${track.file} – BPM: ${track.bpm}, Key: ${track.key}, Energy: ${track.energy}`);
        });

        const downloadLink = document.createElement('a');
        downloadLink.href = `/download/${data.folder}`;
        downloadLink.download = `${data.folder}.zip`;
        downloadLink.click();
    })
    .catch(error => logMessage(`Export error: ${error}`));
}