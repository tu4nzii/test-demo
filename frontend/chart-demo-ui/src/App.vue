<!-- src/App.vue -->
<script setup>
import { ref, computed } from 'vue';
import axios from 'axios';

// --- Reactive State ---
// Backend API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// File holders
const imageFile = ref(null);

// URLs for displaying images
const originalImageUrl = ref('');
const processedImageUrl = ref('');

// Data from backend
const chartId = ref('');
const chartType = ref('');
const confidence = ref('');
const evaluationResults = ref(null);

// UI State
const isLoadingUpload = ref(false);
const isLoadingProcess = ref(false);
const isLoadingEvaluate = ref(false);
const statusMessage = ref('');
const errorMessage = ref('');
const evaluationView = ref('visual');
const isDetailsFullscreen = ref(false);
const fileVersion = ref(0);
const pointPredictionChartTypes = new Set(['scatter', 'bubble']);
const circularPredictionChartTypes = new Set(['pie', 'donut']);
const polarPredictionChartTypes = new Set(['radar', 'rose']);

// --- Computed Properties for Disabling Buttons ---
const isUploadDisabled = computed(() => !imageFile.value || isLoadingUpload.value);
const isProcessDisabled = computed(() => !chartId.value || isLoadingProcess.value);
const isEvaluateDisabled = computed(() => !processedImageUrl.value || isLoadingEvaluate.value);
const evaluationSummary = computed(() => {
  if (!evaluationResults.value) return [];
  const result = evaluationResults.value;
  const summary = result.summary || {};
  const quality = result.quality || {};

  return [
    ['success', result.success],
    ['mode', result.mode],
    ['chart_id', result.chart_id],
    ['chart_type', result.chart_type],
    ['object_count', summary.object_count],
    ['chart_runs', summary.chart_runs],
    ['total_items', summary.total_items],
    ['matched_items', summary.matched_items],
    ['coverage', summary.coverage],
    ['avg_mae', summary.avg_mae],
    ['avg_relative_error', summary.avg_relative_error],
    ['x_ticks_count', quality.x_ticks_count],
    ['y_ticks_count', quality.y_ticks_count],
    ['r_ticks_count', quality.r_ticks_count],
    ['theta_ticks_count', quality.theta_ticks_count],
    ['colors_count', quality.colors_count],
    ['has_basic_grid', quality.has_basic_grid],
    ['has_encrypted_grid', quality.has_encrypted_grid],
  ].filter(([, value]) => value !== undefined && value !== null);
});
const extractedPredictions = computed(() => {
  if (!evaluationResults.value || !Array.isArray(evaluationResults.value.predictions)) {
    return [];
  }
  const predictions = evaluationResults.value.predictions;
  const processedJson = evaluationResults.value.processed_json || evaluationResults.value.source_payload || {};
  const chartTypeValue = getActiveChartType();
  if (pointPredictionChartTypes.has(chartTypeValue)) return predictions;

  const shouldUseXTicks = ['v_bar', 'v_stacked_bar', 'line'].includes(chartTypeValue);
  const primaryTicks = shouldUseXTicks ? processedJson.x_ticks : processedJson.y_ticks;
  const fallbackTicks = shouldUseXTicks ? processedJson.y_ticks : processedJson.x_ticks;
  const tickValues = Array.isArray(primaryTicks) && primaryTicks.length
    ? primaryTicks
    : (Array.isArray(fallbackTicks) ? fallbackTicks : []);
  if (!tickValues.length) return predictions;

  const tickOrder = new Map(tickValues.map((tick, index) => [String(tick), index]));
  return predictions
    .map((item, index) => ({ item, index }))
    .sort((left, right) => {
      const leftOrder = predictionTickOrder(left.item, tickOrder);
      const rightOrder = predictionTickOrder(right.item, tickOrder);
      if (leftOrder !== rightOrder) return leftOrder - rightOrder;
      return left.index - right.index;
    })
    .map(({ item }) => item);
});
const visibleSeriesNames = computed(() => {
  const names = new Set(
    extractedPredictions.value
      .map((item) => String(item.series_name || '').trim())
      .filter(Boolean)
  );
  return names.size > 1;
});
const maxPredictionAbsValue = computed(() => {
  const values = extractedPredictions.value
    .map((item) => predictionNumericValue(item))
    .filter((value) => Number.isFinite(value));
  return values.length ? Math.max(...values.map((value) => Math.abs(value)), 1) : 1;
});
const visualPredictions = computed(() => {
  return extractedPredictions.value.map((item) => {
    const value = predictionNumericValue(item);
    const safeValue = Number.isFinite(value) ? value : 0;
    return {
      ...item,
      numericValue: safeValue,
      displayValue: predictionDisplayValue(item),
      barWidth: `${Math.max(2, Math.min(100, (Math.abs(safeValue) / maxPredictionAbsValue.value) * 100))}%`,
      showSeriesName: visibleSeriesNames.value,
    };
  });
});
const isPointPredictionChart = computed(() => pointPredictionChartTypes.has(getActiveChartType()));
const isCircularPredictionChart = computed(() => circularPredictionChartTypes.has(getActiveChartType()));
const isPolarPredictionChart = computed(() => polarPredictionChartTypes.has(getActiveChartType()));
const pointVisualPredictions = computed(() => {
  return extractedPredictions.value.map((item, index) => {
    const xValue = item?.x ?? item?.value?.x;
    const yValue = item?.y ?? item?.value?.y;
    return {
      id: item?.id ?? `${index}`,
      name: item?.label || item?.id || item?.series_name || `Point ${index + 1}`,
      x: formatPredictionCoordinate(xValue),
      y: formatPredictionCoordinate(yValue),
    };
  });
});
const circularVisualPredictions = computed(() => {
  return extractedPredictions.value.map((item, index) => ({
    id: item?.id ?? `${index}`,
    label: item?.label || item?.id || `Segment ${index + 1}`,
    percentage: formatPercentage(item?.percentage),
    startAngle: formatPredictionCoordinate(item?.start_angle),
    endAngle: formatPredictionCoordinate(item?.end_angle),
  }));
});
const polarVisualPredictions = computed(() => {
  return extractedPredictions.value.map((item, index) => ({
    id: item?.id ?? `${index}`,
    object: item?.id || item?.label || `Object ${index + 1}`,
    series: item?.series_name || '',
    axis: item?.theta_label || item?.label || item?.axis || '-',
    r: formatPredictionCoordinate(item?.r ?? item?.value),
    showSeriesName: visibleSeriesNames.value,
  }));
});
const evaluationJson = computed(() => {
  if (!evaluationResults.value) return '';
  const processedJson = evaluationResults.value.processed_json || evaluationResults.value.source_payload;
  return JSON.stringify(processedJson || evaluationResults.value, null, 2);
});


// --- Methods ---

// Handle file selection from input fields
function handleImageUpload(event) {
  const file = event.target.files[0];
  if (file) {
    fileVersion.value += 1;
    imageFile.value = file;
    if (originalImageUrl.value) {
      URL.revokeObjectURL(originalImageUrl.value);
    }
    originalImageUrl.value = URL.createObjectURL(file);
    resetProcessedResults(true);
    clearMessages();
  }
}

function resetProcessedResults(clearChartInfo = false) {
  processedImageUrl.value = '';
  evaluationResults.value = null;
  evaluationView.value = 'visual';
  isDetailsFullscreen.value = false;
  isLoadingProcess.value = false;
  isLoadingEvaluate.value = false;
  if (clearChartInfo) {
    chartId.value = '';
    chartType.value = '';
    confidence.value = '';
  }
}

function getActiveChartType() {
  return String(evaluationResults.value?.chart_type || chartType.value || '').toLowerCase();
}

function predictionTickOrder(item, tickOrder) {
  const candidates = [
    item?.label,
    typeof item?.id === 'string' && item.id.includes(',') ? item.id.split(',').pop().trim() : item?.id,
  ];
  for (const candidate of candidates) {
    const key = String(candidate ?? '');
    if (tickOrder.has(key)) return tickOrder.get(key);
  }
  return Number.MAX_SAFE_INTEGER;
}

function formatPredictionCoordinate(value) {
  if (value === undefined || value === null || value === '') return '-';
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return value;
  return Number.isInteger(numberValue) ? String(numberValue) : Number(numberValue.toFixed(6)).toString();
}

function formatPercentage(value) {
  if (value === undefined || value === null || value === '') return '-';
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return value;
  return `${Number(numberValue.toFixed(4)).toString()}%`;
}

function predictionNumericValue(item) {
  if (item && Number.isFinite(Number(item.value))) return Number(item.value);
  if (item && Number.isFinite(Number(item.y))) return Number(item.y);
  if (item?.value && typeof item.value === 'object' && Number.isFinite(Number(item.value.y))) {
    return Number(item.value.y);
  }
  return 0;
}

function predictionDisplayValue(item) {
  if (item?.axis === 'theta' && item?.percentage !== undefined && item?.percentage !== null) {
    const percentage = Number(item.percentage);
    return Number.isFinite(percentage) ? `${Number(percentage.toFixed(4))}%` : item.percentage;
  }
  const xValue = item?.x ?? item?.value?.x;
  const yValue = item?.y ?? item?.value?.y;
  if (xValue !== undefined && yValue !== undefined) {
    return `x: ${xValue}, y: ${yValue}`;
  }
  return item?.value;
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
  resetProcessedResults(true);
  isLoadingUpload.value = true;
  const requestFileVersion = fileVersion.value;
  
  const formData = new FormData();
  formData.append('file', imageFile.value);

  try {
    const response = await axios.post(`${API_URL}/api/upload/`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    if (requestFileVersion !== fileVersion.value) return;
    chartId.value = response.data.chart_id;
    chartType.value = response.data.chart_type;
    confidence.value = response.data.confidence;
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
  processedImageUrl.value = '';
  evaluationResults.value = null;
  evaluationView.value = 'visual';
  isDetailsFullscreen.value = false;
  isLoadingProcess.value = true;
  const requestChartId = chartId.value;
  
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
    if (requestChartId !== chartId.value) return;
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
  evaluationView.value = 'visual';
  isDetailsFullscreen.value = false;
  const requestChartId = chartId.value;

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
    if (requestChartId !== chartId.value) return;
    
    evaluationResults.value = resultsResponse.data;
    evaluationView.value = 'visual';
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
          
          <button @click="uploadFiles" :disabled="isUploadDisabled" class="action-button">
            {{ isLoadingUpload ? '上传中...' : '上传' }}
          </button>

          <div v-if="originalImageUrl" class="image-preview">
            <h3>原始图片</h3>
            <div class="large-image-container">
              <img :src="originalImageUrl" alt="Original Chart Preview" />
            </div>
          </div>
          
          <!-- 图表信息显示区域 -->
          <div v-if="chartId" class="chart-info">
            <h3>图表信息</h3>
            <div class="info-grid">
              <div class="info-item">
                <span class="info-label">图表ID:</span>
                <span class="info-value">{{ chartId }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">图表类型:</span>
                <span class="info-value">{{ chartType }}</span>
              </div>
              <div class="info-item">
                <span class="info-label">置信度:</span>
                <span class="info-value">{{ confidence }}</span>
              </div>
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
              <div class="view-toggle" role="group" aria-label="prediction result view">
                <button
                  type="button"
                  :class="{ active: evaluationView === 'visual' }"
                  @click="evaluationView = 'visual'"
                >
                  可视化展示
                </button>
                <button
                  type="button"
                  :class="{ active: evaluationView === 'details' }"
                  @click="evaluationView = 'details'"
                >
                  具体信息
                </button>
              </div>

              <div v-if="evaluationView === 'visual'" class="prediction-visual">
                <table v-if="isPointPredictionChart && pointVisualPredictions.length" class="results-table academic point-prediction-table">
                  <thead>
                    <tr>
                      <th>点名称</th>
                      <th>X</th>
                      <th>Y</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="item in pointVisualPredictions" :key="item.id">
                      <td>{{ item.name }}</td>
                      <td>{{ item.x }}</td>
                      <td>{{ item.y }}</td>
                    </tr>
                  </tbody>
                </table>
                <table v-else-if="isCircularPredictionChart && circularVisualPredictions.length" class="results-table academic point-prediction-table">
                  <thead>
                    <tr>
                      <th>标签名</th>
                      <th>占比</th>
                      <th>起始角度</th>
                      <th>结束角度</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="item in circularVisualPredictions" :key="item.id">
                      <td>{{ item.label }}</td>
                      <td>{{ item.percentage }}</td>
                      <td>{{ item.startAngle }}</td>
                      <td>{{ item.endAngle }}</td>
                    </tr>
                  </tbody>
                </table>
                <table v-else-if="isPolarPredictionChart && polarVisualPredictions.length" class="results-table academic point-prediction-table">
                  <thead>
                    <tr>
                      <th>Object</th>
                      <th v-if="visibleSeriesNames">Series</th>
                      <th>Axis</th>
                      <th>R</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="item in polarVisualPredictions" :key="item.id">
                      <td>{{ item.object }}</td>
                      <td v-if="visibleSeriesNames">{{ item.series }}</td>
                      <td>{{ item.axis }}</td>
                      <td>{{ item.r }}</td>
                    </tr>
                  </tbody>
                </table>
                <div v-else-if="visualPredictions.length" class="visual-list">
                  <div v-for="item in visualPredictions" :key="item.id" class="visual-row">
                    <div class="visual-label">
                      <span class="visual-object">{{ item.label || item.id }}</span>
                      <span v-if="item.showSeriesName" class="visual-series">{{ item.series_name }}</span>
                    </div>
                    <div class="visual-bar-track">
                      <div
                        class="visual-bar"
                        :class="{ negative: item.numericValue < 0 }"
                        :style="{ width: item.barWidth }"
                      ></div>
                    </div>
                    <div class="visual-value">{{ item.displayValue }}</div>
                  </div>
                </div>
                <div v-else class="placeholder compact">
                  <p>暂无可视化预测对象</p>
                </div>
              </div>

              <div v-else class="detail-view">
                <button
                  type="button"
                  class="fullscreen-button"
                  @click="isDetailsFullscreen = true"
                  aria-label="fullscreen details"
                >
                  全屏
                </button>
                <table class="results-table academic">
                  <thead>
                    <tr>
                      <th>标签名</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="([key, value]) in evaluationSummary" :key="key">
                      <td>{{ key }}</td>
                      <td>{{ value }}</td>
                    </tr>
                  </tbody>
                </table>
                <table v-if="extractedPredictions.length" class="results-table academic prediction-table">
                  <thead>
                    <tr>
                      <th>Object</th>
                      <th>Label</th>
                      <th>Value</th>
                      <th>Axis</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="item in extractedPredictions" :key="item.id">
                      <td>{{ item.id }}</td>
                      <td>{{ item.label }}</td>
                      <td>{{ predictionDisplayValue(item) }}</td>
                      <td>{{ item.axis }}</td>
                    </tr>
                  </tbody>
                </table>
                <pre class="json-preview">{{ evaluationJson }}</pre>
              </div>
            </div>
            <div v-else class="placeholder">
              <p>预测结果将在此处显示</p>
            </div>
          </div>
        </div>
      </div>
    </main>

    <div v-if="isDetailsFullscreen" class="details-modal">
      <div class="details-modal-header">
        <h3>具体信息</h3>
        <button type="button" class="fullscreen-button close" @click="isDetailsFullscreen = false">关闭</button>
      </div>
      <div class="details-modal-content">
        <table class="results-table academic">
          <thead>
            <tr>
              <th>标签名</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="([key, value]) in evaluationSummary" :key="key">
              <td>{{ key }}</td>
              <td>{{ value }}</td>
            </tr>
          </tbody>
        </table>
        <table v-if="extractedPredictions.length" class="results-table academic prediction-table">
          <thead>
            <tr>
              <th>Object</th>
              <th>Label</th>
              <th>Value</th>
              <th>Axis</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in extractedPredictions" :key="item.id">
              <td>{{ item.id }}</td>
              <td>{{ item.label }}</td>
              <td>{{ predictionDisplayValue(item) }}</td>
              <td>{{ item.axis }}</td>
            </tr>
          </tbody>
        </table>
        <pre class="json-preview fullscreen-json">{{ evaluationJson }}</pre>
      </div>
    </div>
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
  min-height: 0;
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

.view-toggle {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  margin: 0 0 1rem;
  border: 1px solid #c8d2dc;
  border-radius: 4px;
  overflow: hidden;
}

.view-toggle button {
  min-height: 40px;
  border: 0;
  background-color: #f5f7fa;
  color: #243447;
  cursor: pointer;
  font-weight: 600;
}

.view-toggle button + button {
  border-left: 1px solid #c8d2dc;
}

.view-toggle button.active {
  background-color: #0066cc;
  color: #ffffff;
}

.prediction-visual,
.detail-view {
  padding: 1rem;
}

.detail-view {
  position: relative;
  max-height: 100%;
  overflow: auto;
  padding-top: 3.25rem;
}

.visual-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.visual-row {
  display: grid;
  grid-template-columns: minmax(120px, 1.2fr) minmax(120px, 2fr) minmax(64px, auto);
  gap: 0.75rem;
  align-items: center;
}

.visual-label {
  min-width: 0;
}

.visual-object,
.visual-series {
  display: block;
  overflow-wrap: anywhere;
}

.visual-object {
  font-weight: 600;
  color: #111827;
}

.visual-series {
  color: #5d6b7a;
  font-size: 0.8rem;
}

.visual-bar-track {
  height: 18px;
  background-color: #eef2f6;
  border: 1px solid #d8e0e8;
  border-radius: 4px;
  overflow: hidden;
}

.visual-bar {
  height: 100%;
  background-color: #2f80ed;
}

.visual-bar.negative {
  background-color: #c2410c;
}

.visual-value {
  font-family: 'Courier New', Courier, monospace;
  font-weight: 700;
  text-align: right;
  color: #111827;
}

.prediction-table {
  margin-top: 1rem;
}

.point-prediction-table {
  margin: 0;
}

.point-prediction-table th,
.point-prediction-table td {
  text-align: left;
  white-space: nowrap;
}

.point-prediction-table td:first-child {
  white-space: normal;
  overflow-wrap: anywhere;
  font-weight: 600;
}

.point-prediction-table td:nth-child(2),
.point-prediction-table td:nth-child(3) {
  font-family: 'Courier New', Courier, monospace;
  font-weight: 700;
}

.placeholder.compact {
  min-height: 180px;
}

.fullscreen-button {
  border: 1px solid #0066cc;
  background-color: #ffffff;
  color: #0066cc;
  border-radius: 4px;
  padding: 0.45rem 0.8rem;
  cursor: pointer;
  font-weight: 600;
}

.detail-view > .fullscreen-button {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  z-index: 20;
}

.fullscreen-button:hover {
  background-color: #eef6ff;
}

.details-modal {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background-color: #ffffff;
  display: flex;
  flex-direction: column;
  padding: 1.5rem;
}

.details-modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #d0d0d0;
  padding-bottom: 1rem;
}

.details-modal-header h3 {
  margin: 0;
}

.details-modal-content {
  flex: 1;
  overflow: auto;
  padding-top: 1rem;
}

.json-preview {
  max-height: 260px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background-color: #111827;
  color: #f9fafb;
  padding: 1rem;
  margin: 1rem 0 0;
  border-radius: 4px;
  font-size: 0.8rem;
}

.fullscreen-json {
  max-height: none;
}
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

/* 图表信息区域样式 */
.chart-info {
  margin-top: 1.5rem;
  border-top: 1px solid #e0e0e0;
  padding-top: 1.5rem;
}

.chart-info h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-weight: 500;
  color: #333333;
  font-size: 1.1rem;
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.75rem;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem;
  background-color: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
}

.info-label {
  font-weight: 500;
  color: #333333;
}

.info-value {
  font-weight: 600;
  color: #0066cc;
  font-family: 'Courier New', Courier, monospace;
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
