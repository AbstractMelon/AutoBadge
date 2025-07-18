const express = require("express");
const path = require("path");
const fs = require("fs");
const multer = require("multer");
const AdmZip = require("adm-zip");

// face-api.js requirements
const canvas = require("canvas");
const faceapi = require("face-api.js");
require("@tensorflow/tfjs-node");

const app = express();
const PORT = process.env.PORT || 3007;

// --- Face-API Setup ---
// Monkey patch the environment to use Node.js canvas
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load models once on server startup
const modelsPath = path.join(__dirname, "weights");
Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath),
  faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath),
])
  .then(() => console.log("Face-API models loaded successfully!"))
  .catch((err) => console.error("Error loading models:", err));

// --- Multer Setup for File Uploads ---
// Configure storage to save files temporarily
const upload = multer({ dest: "uploads/" });

// --- Express Middleware ---
// Serve static files from the root directory (where index.html is)
app.use(express.static(path.join(__dirname, "public")));

// --- Helper Functions ---
function formatName(fullName) {
  const parts = fullName.trim().split(/\s+/);
  if (parts.length > 1) {
    // Keep first name and initial of last name for brevity
    return `${parts[0]} ${parts[parts.length - 1][0]}.`;
  }
  return fullName;
}

/**
 * Processes a single image to add a nametag.
 * @param {string} imagePath - Path to the input image file.
 * @param {string} fullName - The full name to add to the nametag.
 * @returns {Promise<Buffer>} A promise that resolves with the processed image as a PNG buffer.
 */
async function processSingleImage(imagePath, fullName) {
  try {
    const image = await canvas.loadImage(imagePath);
    console.log(
      `Processing image: ${imagePath} (${image.width}x${image.height})`
    );

    const detection = await faceapi.detectSingleFace(
      image,
      new faceapi.SsdMobilenetv1Options()
    );

    let cropBox;
    const targetWidth = 512;
    const targetHeight = 682; // Approximately 3:4 aspect ratio

    if (detection) {
      const { x, y, width, height } = detection.box;
      const faceCenterX = x + width / 2;
      const faceCenterY = y + height / 2;
      const minFaceDimension = Math.min(width, height);
      const zoomFactor = 1.8;
      const cropWidth = minFaceDimension * zoomFactor;
      const cropHeight =
        minFaceDimension * (zoomFactor * (targetHeight / targetWidth));

      const aspectFitScale = Math.max(
        cropWidth / targetWidth,
        cropHeight / targetHeight
      );
      const finalCropWidth = targetWidth * aspectFitScale;
      const finalCropHeight = targetHeight * aspectFitScale;

      let cropX = faceCenterX - finalCropWidth / 2;
      let cropY = faceCenterY - finalCropHeight / 2;

      if (cropX < 0) cropX = 0;
      if (cropY < 0) cropY = 0;
      if (cropX + finalCropWidth > image.width)
        cropX = image.width - finalCropWidth;
      if (cropY + finalCropHeight > image.height)
        cropY = image.height - finalCropHeight;

      cropBox = {
        x: cropX,
        y: cropY,
        width: finalCropWidth,
        height: finalCropHeight,
      };
    } else {
      const imageAspectRatio = image.width / image.height;
      const targetAspectRatio = targetWidth / targetHeight;

      let sourceWidth = image.width;
      let sourceHeight = image.height;
      let sourceX = 0;
      let sourceY = 0;

      if (imageAspectRatio > targetAspectRatio) {
        sourceWidth = image.height * targetAspectRatio;
        sourceX = (image.width - sourceWidth) / 2;
      } else {
        sourceHeight = image.width / targetAspectRatio;
        sourceY = (image.height - sourceHeight) / 2;
      }

      cropBox = {
        x: sourceX,
        y: sourceY,
        width: sourceWidth,
        height: sourceHeight,
      };
    }

    const outputCanvas = canvas.createCanvas(targetWidth, targetHeight);
    const ctx = outputCanvas.getContext("2d");

    ctx.drawImage(
      image,
      cropBox.x,
      cropBox.y,
      cropBox.width,
      cropBox.height,
      0,
      0,
      targetWidth,
      targetHeight
    );

    // Draw the Nametag
    const name = formatName(fullName);
    const baseFontSize = outputCanvas.width * 0.09;
    ctx.font = `900 ${baseFontSize}px 'Arial', sans-serif`;
    ctx.textBaseline = "alphabetic";

    const textMetrics = ctx.measureText(name);
    const textHeight =
      textMetrics.actualBoundingBoxAscent +
      textMetrics.actualBoundingBoxDescent;

    const marginX = outputCanvas.width * 0.05;
    const marginY = outputCanvas.height * 0.04;

    const rectPadding = baseFontSize * 0.3;
    const rectWidth = textMetrics.width + rectPadding * 2;
    const rectHeight = textHeight + rectPadding * 2;

    const rectX = (outputCanvas.width - rectWidth) / 2;
    const rectY = outputCanvas.height - rectHeight - marginY;

    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(rectX, rectY, rectWidth, rectHeight);

    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(
      name,
      rectX + rectPadding,
      rectY + rectPadding + textMetrics.actualBoundingBoxAscent
    );

    return outputCanvas.toBuffer("image/png");
  } catch (error) {
    console.error(`Error processing image ${imagePath}:`, error);
    throw error; // Re-throw to be caught by the calling function
  }
}

// --- API Endpoint for Single Image ---
app.post("/process-image", upload.single("imageFile"), async (req, res) => {
  const filePath = req.file.path;
  const fullName = req.body.fullName;

  if (!filePath || !fullName) {
    if (filePath) fs.unlinkSync(filePath);
    return res.status(400).json({ message: "Missing image file or name." });
  }

  try {
    const buffer = await processSingleImage(filePath, fullName);
    res.set("Content-Type", "image/png");
    res.send(buffer);
  } catch (error) {
    console.error("Single image processing error:", error);
    res.status(500).json({ message: "Failed to process image." });
  } finally {
    if (filePath) fs.unlinkSync(filePath);
  }
});

// --- API Endpoint for Mass Upload (Zip) ---
app.post("/process-zip", upload.single("zipFile"), async (req, res) => {
  const zipFilePath = req.file.path;
  const tempDir = path.join(__dirname, "temp_unzipped_" + Date.now());
  const outputDir = path.join(__dirname, "temp_processed_" + Date.now());

  try {
    // 1. Unzip the uploaded file
    const zip = new AdmZip(zipFilePath);
    zip.extractAllTo(tempDir, true); // true for overwrite
    console.log(`Zip extracted to: ${tempDir}`);

    const files = fs.readdirSync(tempDir);
    const imageFiles = files.filter((file) => {
      const ext = path.extname(file).toLowerCase();
      return [".png", ".jpg", ".jpeg", ".webp"].includes(ext);
    });

    if (imageFiles.length === 0) {
      return res.status(400).json({
        message:
          "No supported image files (.png, .jpg, .jpeg, .webp) found in the zip.",
      });
    }

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir);
    }

    const processedImagePromises = imageFiles.map(async (fileName, index) => {
      const imagePath = path.join(tempDir, fileName);
      const nameWithoutExtension = fileName.replace(/\.[^/.]+$/, "");
      // Derive full name from filename, replacing underscores/hyphens with spaces
      const derivedFullName = decodeURIComponent(nameWithoutExtension)
        .replace(/[_\-]+/g, " ")
        .trim();

      console.log(
        `Processing ${index + 1} of ${imageFiles.length}: ${fileName}`
      );

      try {
        const processedBuffer = await processSingleImage(
          imagePath,
          derivedFullName
        );
        const outputFileName = `${derivedFullName.replace(/\s+/g, "_")}.png`;
        const outputPath = path.join(outputDir, outputFileName);
        fs.writeFileSync(outputPath, processedBuffer);
        console.log(`Processed and saved: ${outputFileName}`);
        return outputPath;
      } catch (error) {
        console.warn(
          `Skipping ${fileName} due to processing error: ${error.message}`
        );
        return null; // Return null for failed processing
      }
    });

    console.log("Waiting for all images to be processed...");

    const processedPaths = (await Promise.all(processedImagePromises)).filter(
      Boolean
    ); // Filter out nulls

    if (processedPaths.length === 0) {
      return res.status(500).json({
        message:
          "No images could be processed from the zip file. Check image formats or corruption.",
      });
    }

    // 3. Create a new zip file with processed images
    const outputZip = new AdmZip();
    processedPaths.forEach((filePath) => {
      const fileName = path.basename(filePath);
      outputZip.addLocalFile(filePath, "", fileName); // Add to root of zip
    });

    console.log("Creating final zip file...");

    const finalZipBuffer = outputZip.toBuffer();
    const finalZipFileName = `autobadges_${Date.now()}.zip`;

    res.set("Content-Type", "application/zip");
    res.set(
      "Content-Disposition",
      `attachment; filename="${finalZipFileName}"`
    );
    res.send(finalZipBuffer);
    console.log("Sent final processed zip to client.");
  } catch (error) {
    console.error("Mass upload processing error:", error);
    res.status(500).json({ message: "Failed to process zip file." });
  } finally {
    // Clean up temporary files and directories
    if (zipFilePath && fs.existsSync(zipFilePath)) {
      fs.unlinkSync(zipFilePath);
      console.log("Deleted temporary uploaded zip:", zipFilePath);
    }
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
      console.log("Deleted temporary unzipped directory:", tempDir);
    }
    if (fs.existsSync(outputDir)) {
      fs.rmSync(outputDir, { recursive: true, force: true });
      console.log("Deleted temporary processed directory:", outputDir);
    }
  }
});

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
