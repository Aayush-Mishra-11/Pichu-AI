// ClickSpark Effect
// This file is kept minimal - the main application logic is in index.html inline script

(function() {
    // Wait for DOM and audio URL from template
    window.addEventListener('DOMContentLoaded', function() {
        // The sparkAudioUrl is defined in the template before this script loads
        if (typeof sparkAudioUrl === 'undefined') {
            window.sparkAudioUrl = "/static/audio/spark.mp3";
        }
    });
})();
