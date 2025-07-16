const express = require("express");
const path = require("path");
const fs = require("fs");
const multer = require("multer");

// face-api.js requirements
const canvas = require("canvas");
const faceapi = require("face-api.js");
require("@tensorflow/tfjs-node");

const app = express();
const PORT = process.env.PORT || 3000;

// --- Face-API Setup ---
// Monkey patch the environment to use Node.js canvas
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load models once on server startup
const modelsPath = path.join(__dirname, "weights");
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromDisk(modelsPath),
  faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath),
  faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath),
])
  .then(() => console.log("Face-API models loaded successfully!"))
  .catch((err) => console.error("Error loading models:", err));

// --- Multer Setup for File Uploads ---
// Configure storage to save files temporarily
const upload = multer({ dest: "uploads/" });

// --- Express Middleware ---
// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, "public")));

// --- Helper Functions ---
function formatName(fullName) {
  const parts = fullName.trim().split(/\s+/);
  if (parts.length > 1) {
    return `${parts[0]} ${parts[parts.length - 1][0]}.`;
  }
  return fullName;
}

// --- API Endpoint ---
app.post("/process-image", upload.single("imageFile"), async (req, res) => {
  const filePath = req.file.path;
  const fullName = req.body.fullName;

  if (!filePath || !fullName) {
    fs.unlinkSync(filePath); // Clean up uploaded file
    return res.status(400).json({ message: "Missing image file or name." });
  }

  try {
    // Load image from disk into a canvas.Image object
    const image = await canvas.loadImage(filePath);

    // Detect face
    const detection = await faceapi.detectSingleFace(
      image,
      new faceapi.TinyFaceDetectorOptions()
    );

    let cropBox;
    if (detection) {
      const { x, y, width, height } = detection.box;
      const size = Math.max(width, height);
      const padding = size * 0.4;
      const paddedSize = size + padding * 2;
      cropBox = {
        x: x + width / 2 - paddedSize / 2,
        y: y + height / 2 - paddedSize / 2,
        width: paddedSize,
        height: paddedSize,
      };
    } else {
      // Fallback to center crop if no face is detected
      const size = Math.min(image.width, image.height);
      cropBox = {
        x: (image.width - size) / 2,
        y: (image.height - size) / 2,
        width: size,
        height: size,
      };
    }

    // Create a new canvas to draw the final result
    const outputCanvas = canvas.createCanvas(512, 512);
    const ctx = outputCanvas.getContext("2d");

    // Draw the cropped image onto the new canvas
    ctx.drawImage(
      image,
      cropBox.x,
      cropBox.y,
      cropBox.width,
      cropBox.height,
      0,
      0,
      512,
      512
    );

    // --- Draw the Nametag ---
    const name = formatName(fullName);
    const fontSize = outputCanvas.width * 0.08;
    ctx.font = `600 ${fontSize}px sans-serif`;
    const textMargin = outputCanvas.width * 0.05;
    const textMetrics = ctx.measureText(name);
    const textHeight =
      textMetrics.actualBoundingBoxAscent +
      textMetrics.actualBoundingBoxDescent;

    const textX = textMargin;
    const textY = outputCanvas.height - textMargin;
    const rectPadding = fontSize * 0.2;

    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    ctx.fillRect(
      textX - rectPadding,
      textY - textHeight - rectPadding,
      textMetrics.width + rectPadding * 2,
      textHeight + rectPadding * 2
    );

    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(name, textX, textY);

    // Send the final image back to the client
    const buffer = outputCanvas.toBuffer("image/png");
    res.set("Content-Type", "image/png");
    res.send(buffer);
  } catch (error) {
    console.error("Processing error:", error);
    res.status(500).json({ message: "Failed to process image." });
  } finally {
    // Clean up the temporarily uploaded file
    fs.unlinkSync(filePath);
  }
});

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
