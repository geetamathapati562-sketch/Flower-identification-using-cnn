// main.js
const imageInput = document.getElementById("image-input");
const previewImg = document.getElementById("preview-img");
const predictBtn = document.getElementById("predict-btn");
const resultBox = document.getElementById("result");
const predLabel = document.getElementById("pred-label");
const predConfidence = document.getElementById("pred-confidence");

let selectedFile = null;

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.hidden = false;
  predictBtn.disabled = false;
  resultBox.hidden = true;
});

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.innerText = "Predicting...";

  const form = new FormData();
  form.append("image", selectedFile);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: form
    });
    const data = await res.json();
    if (data.success) {
      predLabel.innerText = data.prediction;
      predConfidence.innerText = "Confidence: " + (data.confidence * 100).toFixed(2) + "%";
      resultBox.hidden = false;
    } else {
      predLabel.innerText = "Error: " + (data.error || "Unknown");
      resultBox.hidden = false;
    }
  } catch (err) {
    predLabel.innerText = "Request failed: " + err;
    resultBox.hidden = false;
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerText = "Predict";
  }
});
