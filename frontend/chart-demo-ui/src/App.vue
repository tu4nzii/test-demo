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
  <div class="app-container">
    <header class="app-header">
      <h1>图表智能分析</h1>
    </header>
    
    <main class="content-container">
      <div v-if="statusMessage" class="message status">{{ statusMessage }}</div>
      <div v-if="errorMessage" class="message error">{{ errorMessage }}</div>

      <!-- 三列并排布局 -->
      <div class="columns-wrapper">
        <!-- 左侧：上传区域 -->
        <div class="column upload-column">
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
            <h3>原始图片</h3>
            <div class="large-image-container">
              <img :src="originalImageUrl" alt="Original Chart Preview" />
            </div>
          </div>
        </div>

        <!-- 中间：处理区域 -->
        <div class="column process-column">
          <h2>2. 加密处理</h2>
          <button @click="processImage" :disabled="isProcessDisabled" class="action-button">
            {{ isLoadingProcess ? '处理中...' : '加密处理' }}
          </button>

          <div class="image-preview">
            <h3>加密后图片</h3>
            <div v-if="processedImageUrl" class="large-image-container">
              <img :src="processedImageUrl" alt="Processed Chart" />
            </div>
            <div v-else class="placeholder">
              <p>处理后的图片将在此处显示</p>
            </div>
          </div>
        </div>

        <!-- 右侧：评估区域 -->
        <div class="column evaluation-column">
          <h2>3. 评估预测</h2>
          <button @click="evaluateChart" :disabled="isEvaluateDisabled" class="action-button">
            {{ isLoadingEvaluate ? '预测中...' : '进行预测' }}
          </button>

          <div class="evaluation-preview">
            <h3>预测结果</h3>
            <div v-if="evaluationResults" class="table-container">
              <table class="results-table academic">
                <thead>
                  <tr>
                    <th>标签名</th>
                    <th>amplifier</th>
                    <th>feedback</th>
                    <th>grid</th>
                    <th>baseline</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(data, key) in evaluationResults.rose_000.data" :key="key">
                    <td>{{ key }}</td>
                    <td>{{ data.amplifier }}</td>
                    <td>{{ data.feedback[0] }}</td>
                    <td>{{ data.with_grid[0] }}</td>
                    <td>{{ data.origin }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div v-else class="placeholder">
              <p>预测结果将在此处显示</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<style scoped>
/* 学术风格的全局样式 */
:global(body) {
  background-color: #f8f9fa;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #000000;
  margin: 0;
  padding: 0;
  line-height: 1.6;
}

.app-container {
  width: 100%;
  min-height: 100vh;
}

.app-header {
  background-color: #ffffff;
  border-bottom: 1px solid #e0e0e0;
  padding: 1rem 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.app-header h1 {
  margin: 0;
  font-weight: 600;
  color: #1a1a1a;
  text-align: center;
  font-size: 1.8rem;
  letter-spacing: 0.3px;
}

.content-container {
  max-width: 1800px;
  margin: 0 auto;
  padding: 2rem;
  width: 100%;
  box-sizing: border-box;
}

.message {
  padding: 1rem;
  margin-bottom: 1.5rem;
  border-radius: 4px;
  color: #ffffff;
  text-align: center;
  font-weight: 500;
}

.message.status {
  background-color: #4CAF50;
}

.message.error {
  background-color: #f44336;
}

/* 三列布局 */
.columns-wrapper {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1.5rem;
  height: calc(100vh - 200px);
  min-height: 700px;
}

/* 列样式 */
.column {
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  padding: 1.5rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  height: 100%;
}

.column h2 {
  margin-top: 0;
  margin-bottom: 1.5rem;
  font-weight: 600;
  color: #1a1a1a;
  border-bottom: 2px solid #0066cc;
  padding-bottom: 0.5rem;
  font-size: 1.3rem;
}

.column h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-weight: 500;
  color: #333333;
  font-size: 1.1rem;
}

/* 输入组样式 */
.input-group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  width: 100%;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.input-group label {
  font-weight: 500;
  color: #333333;
  font-size: 1rem;
}

input[type="file"] {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #d0d0d0;
  border-radius: 4px;
  background-color: #f9f9f9;
  font-size: 0.95rem;
  transition: border-color 0.3s ease;
}

input[type="file"]:focus {
  outline: none;
  border-color: #0066cc;
  box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.1);
}

/* 按钮样式 */
.action-button {
  background-color: #0066cc;
  color: white;
  padding: 12px 25px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: background-color 0.3s ease;
  width: 100%;
  margin-bottom: 1.5rem;
}

.action-button:hover:not(:disabled) {
  background-color: #0052a3;
}

.action-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.6;
}

/* 图片预览区域 */
.image-preview {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.large-image-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  padding: 1rem;
  min-height: 400px;
}

.large-image-container img {
  max-width: 100%;
  max-height: 100%;
  height: auto;
  object-fit: contain;
  border: 1px solid #d0d0d0;
  background-color: #ffffff;
}

/* 评估结果区域 */
.evaluation-preview {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.table-container {
  flex: 1;
  overflow-y: auto;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
}

/* 学术风格表格 */
.results-table.academic {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  background-color: #ffffff;
}

.results-table.academic th {
  background-color: #f0f0f0;
  border: 1px solid #d0d0d0;
  padding: 10px 8px;
  text-align: center;
  font-weight: 600;
  color: #000000;
  font-size: 0.85rem;
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 10;
}

.results-table.academic td {
  border: 1px solid #d0d0d0;
  padding: 10px 8px;
  text-align: center;
  color: #000000;
  font-size: 0.85rem;
}

.results-table.academic tr:nth-child(even) {
  background-color: #f9f9f9;
}

.results-table.academic tr:hover {
  background-color: #f5f5f5;
}

/* 占位符样式 */
.placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed #cccccc;
  border-radius: 4px;
  color: #666666;
  background-color: #f9f9f9;
  min-height: 400px;
}

/* 响应式设计 */
@media (max-width: 1400px) {
  .columns-wrapper {
    grid-template-columns: 1fr;
    height: auto;
    gap: 2rem;
  }
  
  .column {
    height: auto;
    min-height: 500px;
  }
  
  .large-image-container,
  .placeholder {
    min-height: 350px;
  }
}

@media (max-width: 768px) {
  .content-container {
    padding: 1rem;
  }
  
  .app-header h1 {
    font-size: 1.5rem;
  }
  
  .column h2 {
    font-size: 1.2rem;
  }
}
</style>