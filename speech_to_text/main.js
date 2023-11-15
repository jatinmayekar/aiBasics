let mp3Encoder;
let mp3Buffers = [];
const sampleRate = 16000; // Set the sample rate for your use case

const engine = {
    onmessage: function(e) {
        if (!mp3Encoder) {
            mp3Encoder = new lamejs.Mp3Encoder(1, sampleRate, 128); // Mono channel, 128kbps
        }

        if (e.data.command === 'process') {
            const inputData = e.data.inputFrame;
            const mp3Buffer = mp3Encoder.encodeBuffer(inputData);
            if (mp3Buffer.length > 0) {
                mp3Buffers.push(new Int8Array(mp3Buffer));
            }
        }
    }
}

document.getElementById('start-button').addEventListener('click', async () => {
    WebVoiceProcessor.subscribe(engine);
});

document.getElementById('stop-button').addEventListener('click', () => {
    WebVoiceProcessor.unsubscribe(engine);

    const finalMp3Buffer = mp3Encoder.flush();
    if (finalMp3Buffer.length > 0) {
        mp3Buffers.push(new Int8Array(finalMp3Buffer));
    }

    const mp3Blob = new Blob(mp3Buffers, { type: 'audio/mp3' });
    const mp3Url = URL.createObjectURL(mp3Blob);

    const downloadLink = document.createElement('a');
    downloadLink.href = mp3Url;
    downloadLink.download = 'recording.mp3';
    downloadLink.textContent = 'Download MP3';
    document.body.appendChild(downloadLink);
});
