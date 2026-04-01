const startBtn = document.getElementById("startChat");
const video = document.getElementById("camera");
const canvas = document.getElementById("snapshot");
const chatBox = document.getElementById("chatBox");

let cameraStarted = false;

// Start Chat Button
startBtn.addEventListener("click", async () => {
    if (cameraStarted) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        cameraStarted = true;

        await loadModels();
        startFaceDetection();

    } catch (error) {
        alert("Camera permission needed!");
        console.error(error);
    }
});

// Load face detection model
async function loadModels() {
   await faceapi.nets.tinyFaceDetector.loadFromUri("./models");
    console.log("Face detection model loaded");
}

function startFaceDetection() {
    setInterval(async () => {

        if (video.readyState !== 4) return;

        const detections = await faceapi.detectAllFaces(
            video,
            new faceapi.TinyFaceDetectorOptions()
        );

        const faceNow = detections.length > 0;

        // Face appeared
        if (faceNow && !lastFaceState) {
            faceStatus.innerText = "Face detected";

            if (!moodAlreadySent) {
                analyzeMoodFromCamera();
                moodAlreadySent = true;
            }
        }

        // Face disappeared
        if (!faceNow && lastFaceState) {
            faceStatus.innerText = "No face detected";
            moodAlreadySent = false;
        }

        lastFaceState = faceNow;

    }, 1500);
}

// Capture image from camera
async function captureImage() {
    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0);

    const imageBase64 = canvas.toDataURL("image/jpeg").split(",")[1];

    console.log("Image captured");

    try {
        const response = await fetch("http://127.0.0.1:5000/analyze-mood", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageBase64
            })
        });

        const data = await response.json();

        console.log("AI Reply:", data.reply);

        addMessage("bot", data.reply);

    } catch (error) {
        console.error("Error connecting to backend:", error);
    }
}

// Add message to chat
function addMessage(sender, text) {
    const msg = document.createElement("div");
    msg.classList.add(sender === "bot" ? "bot-message" : "user-message");
    msg.innerText = text;
    chatBox.appendChild(msg);
}