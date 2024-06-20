<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Player</title>
    <link rel="stylesheet" href="styles.css">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .video-container {
            max-width: 800px;
            margin: auto;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
        }

        video {
            width: 100%;
            display: block;
        }

        /* Optional: Customize video player controls */
        video::-webkit-media-controls {
            background-color: rgba(0, 0, 0, 0.7);
        }

        video::-webkit-media-controls-play-button,
        video::-webkit-media-controls-start-playback-button {
            color: white;
        }

        video::-webkit-media-controls-current-time-display,
        video::-webkit-media-controls-time-remaining-display {
            color: white;
        }

        video::-webkit-media-controls-mute-button,
        video::-webkit-media-controls-fullscreen-button {
            color: white;
        }

    </style>
</head>

<body>
    <div class="video-container">
        <video controls>
            <source src="video.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
</body>

</html>
