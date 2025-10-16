<!-- src/App.vue -->
<script setup>
import { ref, computed } from 'vue';
import axios from 'axios';

// --- Reactive State ---
// Backend API URL
const API_URL = 'http://127.0.0.1:8000';

// File holders
const imageFile = ref(null);
const jsonFile = ref(null);

// URLs for displaying images
const originalImageUrl = ref('');
const processedImageUrl = ref('');

// Data from backend
const chartId = ref('');
const evaluationResults = ref(null);

// UI State
const isLoadingUpload = ref(false);
const isLoadingProcess = ref(false);
const isLoadingEvaluate = ref(false);
const statusMessage = ref('');
const errorMessage = ref('');

// --- Computed Properties for Disabling Buttons ---
const isUploadDisabled = computed(() => !imageFile.value || !jsonFile.value || isLoadingUpload.value);
const isProcessDisabled = computed(() => !chartId.value || isLoadingProcess.value);
const isEvaluateDisabled = computed(() => !processedImageUrl.value || isLoadingEvaluate.value);


// --- Methods ---

// Handle file selection from input fields
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (file) {
    imageFile.value = file;
    originalImageUrl.value = URL.createObjectURL(file);
    clearMessages();
  }
}

function handleJsonUpload(event) {
  const file = event.target.files[0];
  if (file) {
    jsonFile.value = file;
    clearMessages();
  }
}

// Clear status/error messages
function clearMessages() {
  statusMessage.value = '';
  errorMessage.value = '';
}

// 1. Upload files to the backend
async function uploadFiles() {
  if (isUploadDisabled.value) return;
  
  clearMessages();
  isLoadingUpload.value = true;
  
  const formData = new FormData();
  formData.append('file', imageFile.value);
  formData.append('json_data', jsonFile.value);

  try {
    const response = await axios.post(`${API_URL}/api/upload/`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    chartId.value = response.data.chart_id;
    statusMessage.value = `上传成功！图表ID: ${chartId.value}`;
  } catch (error) {
    console.error("Upload error:", error);
    errorMessage.value = `上传失败: ${error.response?.data?.detail || error.message}`;
  } finally {
    isLoadingUpload.value = false;
  }
}

// 2. Process the uploaded chart
async function processImage() {
  if (isProcessDisabled.value) return;
  
  clearMessages();
  isLoadingProcess.value = true;
  
  try {
    // 修正: 不再使用 FormData，而是将 chart_id 作为 URL 查询参数传递
    // POST 请求的第二个参数是请求体(body)，这里我们没有 body，所以设为 null
    const response = await axios.post(
      `${API_URL}/api/process/`, 
      null, // 请求体为空
      { 
        params: { chart_id: chartId.value } // 将 chart_id 作为查询参数
      }
    );
    
    // 构造完整的图片 URL
    processedImageUrl.value = `${API_URL}${response.data.encrypted_image_url}`;
    statusMessage.value = '加密处理成功！';
  } catch (error) {
    console.error("Processing error:", error);
    errorMessage.value = `处理失败: ${error.response?.data?.detail || error.message}`;
  } finally {
    isLoadingProcess.value = false;
  }
}

// 3. 评估处理后的图表并获取结果 (已修正)
async function evaluateChart() {
  if (isEvaluateDisabled.value) return;

  clearMessages();
  isLoadingEvaluate.value = true;
  evaluationResults.value = null;

  try {
    // 修正: 同样，将 chart_id 作为 URL 查询参数传递
    const evalResponse = await axios.post(
      `${API_URL}/api/evaluate/`, 
      null, // 请求体为空
      { 
        params: { chart_id: chartId.value } // 将 chart_id 作为查询参数
      }
    );
    statusMessage.value = '评估请求成功，正在获取结果...';
    
    const resultsUrl = evalResponse.data.results_url;
    const resultsResponse = await axios.get(`${API_URL}${resultsUrl}`);
    
    evaluationResults.value = resultsResponse.data;
    statusMessage.value = '评估结果获取成功！';
  } catch (error) {
    console.error("Evaluation error:", error);
    errorMessage.value = `评估失败: ${error.response?.data?.detail || error.message}`;
  } finally {
    isLoadingEvaluate.value = false;
  }
}
</script>

<template>
  <div class="main-container">
    <h1>图表智能分析</h1>

    <div v-if="statusMessage" class="message status">{{ statusMessage }}</div>
    <div v-if="errorMessage" class="message error">{{ errorMessage }}</div>

    <div class="columns-wrapper">
      <!-- Left Column: Upload -->
      <div class="column">
        <h2>1. 上传文件</h2>
        <div class="input-group">
          <label for="image-upload">选择图片文件:</label>
          <input id="image-upload" type="file" @change="handleImageUpload" accept="image/png, image/jpeg" />
        </div>
        <div class="input-group">
          <label for="json-upload">选择JSON数据:</label>
          <input id="json-upload" type="file" @change="handleJsonUpload" accept="application/json" />
        </div>
        
        <button @click="uploadFiles" :disabled="isUploadDisabled" class="action-button">
          {{ isLoadingUpload ? '上传中...' : '上传' }}
        </button>

        <div v-if="originalImageUrl" class="image-preview">
          <p>原始图片预览:</p>
          <img :src="originalImageUrl" alt="Original Chart Preview" />
        </div>
      </div>

      <!-- Middle Column: Process -->
      <div class="column">
        <h2>2. 加密处理</h2>
        <button @click="processImage" :disabled="isProcessDisabled" class="action-button">
          {{ isLoadingProcess ? '处理中...' : '加密处理' }}
        </button>

        <div v-if="processedImageUrl" class="image-preview">
          <p>加密后图片:</p>
          <img :src="processedImageUrl" alt="Processed Chart" />
        </div>
        <div v-else class="placeholder">
          <p>处理后的图片将在此处显示</p>
        </div>
      </div>

      <!-- Right Column: Evaluate -->
      <div class="column">
        <h2>3. 评估预测</h2>
        <button @click="evaluateChart" :disabled="isEvaluateDisabled" class="action-button">
          {{ isLoadingEvaluate ? '预测中...' : '进行预测' }}
        </button>

        <div v-if="evaluationResults" class="results-preview">
          <p>预测结果:</p>
          <pre><code>{{ JSON.stringify(evaluationResults, null, 2) }}</code></pre>
        </div>
         <div v-else class="placeholder">
          <p>预测的JSON数据将在此处显示</p>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* To see the transparent background, the body needs a background */
/* You would typically set this in a global CSS file like src/assets/main.css */
:global(body) {
  background: #283048;  /* fallback for old browsers */
  background: -webkit-linear-gradient(to right, #859398, #283048);
  background: linear-gradient(to right, #859398, #283048);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #fff;
}

.main-container {
  max-width: 1400px;
  margin: 2rem auto;
  padding: 2rem;
  background-color: rgba(0, 0, 0, 0.2); /* Transparent Background */
  border-radius: 15px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
  text-align: center;
}

h1 {
  margin-bottom: 2rem;
  font-weight: 300;
  letter-spacing: 1px;
}

.columns-wrapper {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2rem;
}

.column {
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.5rem;
}

h2 {
  margin-top: 0;
  font-weight: 400;
  border-bottom: 1px solid rgba(76, 175, 80, 0.5);
  padding-bottom: 0.5rem;
  width: 100%;
}

.input-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  width: 100%;
  gap: 0.5rem;
}

input[type="file"] {
  width: 100%;
}

.action-button {
  background-color: #4CAF50; /* Green */
  color: white;
  padding: 12px 25px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.1s ease;
  width: 80%;
}

.action-button:hover:not(:disabled) {
  background-color: #45a049;
  transform: scale(1.02);
}

.action-button:disabled {
  background-color: #5a6e5a;
  cursor: not-allowed;
  opacity: 0.6;
}

.image-preview, .results-preview, .placeholder {
  width: 100%;
  margin-top: 1rem;
  min-height: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.placeholder {
  border: 2px dashed rgba(255, 255, 255, 0.3);
  border-radius: 5px;
  color: rgba(255, 255, 255, 0.5);
}

.image-preview img {
  max-width: 100%;
  height: auto;
  border-radius: 5px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.results-preview pre {
  background-color: rgba(0, 0, 0, 0.5);
  color: #f1f1f1;
  padding: 1rem;
  border-radius: 5px;
  text-align: left;
  width: 100%;
  box-sizing: border-box;
  overflow-x: auto;
  font-size: 0.85rem;
}

.message {
  padding: 1rem;
  margin-bottom: 1.5rem;
  border-radius: 5px;
  color: white;
}

.message.status {
  background-color: rgba(76, 175, 80, 0.8);
}

.message.error {
  background-color: rgba(244, 67, 54, 0.8);
}
</style>