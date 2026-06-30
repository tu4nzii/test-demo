<!-- src/App.vue -->
<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue';
import axios from 'axios';

// --- Reactive State ---
// Backend API URL
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const STATIC_PREVIEW_BASE = `${import.meta.env.BASE_URL || '/'}static-preview`;
const FORCE_STATIC_PREVIEW = import.meta.env.VITE_STATIC_PREVIEW_ONLY === 'true'
  || (!import.meta.env.VITE_API_URL && typeof window !== 'undefined' && window.location.hostname.endsWith('github.io'));

// File holders
const imageFile = ref(null);

// URLs for displaying images
const originalImageUrl = ref('');
const processedImageUrl = ref('');
const processedStandardImageUrl = ref('');
const processedColoredImageUrl = ref('');
const gridLinePreviewMode = ref('standard');

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
const messageCountdown = ref(0);
let messageCountdownTimer = null;
const evaluationView = ref('visual');
const isDetailsFullscreen = ref(false);
const fileVersion = ref(0);
const datasetSamples = ref([]);
const datasetSource = ref('realworld');
const datasetCategory = ref('');
const datasetCategories = ref([]);
const selectedDatasetSampleId = ref('');
const selectedStaticDatasetSample = ref(null);
const isLoadingDataset = ref(false);
const staticPreviewManifest = ref(null);
const staticPreviewMode = ref(FORCE_STATIC_PREVIEW);
const datasetSourceOptions = [
  { value: 'realworld', label: 'Final-RealDataset' },
  { value: 'synthetic', label: 'Sy.Dataset' },
];
const pointPredictionChartTypes = new Set(['scatter', 'bubble']);
const circularPredictionChartTypes = new Set(['pie', 'donut']);
const polarPredictionChartTypes = new Set(['radar', 'rose']);

function stopMessageCountdown() {
  if (messageCountdownTimer) {
    window.clearInterval(messageCountdownTimer);
    messageCountdownTimer = null;
  }
}

function startMessageCountdown() {
  stopMessageCountdown();
  if (!statusMessage.value && !errorMessage.value) {
    messageCountdown.value = 0;
    return;
  }
  messageCountdown.value = 3;
  messageCountdownTimer = window.setInterval(() => {
    messageCountdown.value -= 1;
    if (messageCountdown.value <= 0) {
      stopMessageCountdown();
      clearMessages();
    }
  }, 1000);
}

// --- Computed Properties for Disabling Buttons ---
const isUploadDisabled = computed(() => !imageFile.value || isLoadingUpload.value || staticPreviewMode.value);
const isProcessDisabled = computed(() => !chartId.value || isLoadingProcess.value);
const isEvaluateDisabled = computed(() => !processedImageUrl.value || isLoadingEvaluate.value);
const hasColoredGridPreview = computed(() => Boolean(processedColoredImageUrl.value));
const hasGridPreview = computed(() => Boolean(processedStandardImageUrl.value || processedColoredImageUrl.value));
const previewImageUrl = computed(() => {
  if (gridLinePreviewMode.value === 'original') return originalImageUrl.value;
  if (gridLinePreviewMode.value === 'color' && processedColoredImageUrl.value) return processedColoredImageUrl.value;
  return processedStandardImageUrl.value || processedImageUrl.value;
});
const previewImageTitle = computed(() => (gridLinePreviewMode.value === 'original' ? '原图' : '加密图片'));
const previewPlaceholderText = computed(() => (
  gridLinePreviewMode.value === 'original'
    ? '上传图片或选择数据集样例后显示'
    : '处理后的图片将在此处显示'
));
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

  const shouldUseXTicks = ['v_bar', 'line'].includes(chartTypeValue);
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
async function handleImageUpload(event) {
  const file = event.target.files[0];
  if (file) {
    fileVersion.value += 1;
    imageFile.value = file;
    selectedDatasetSampleId.value = '';
    revokeOriginalImageUrl();
    originalImageUrl.value = URL.createObjectURL(file);
    resetProcessedResults(true);
    clearMessages();
    await uploadFiles();
  }
}

function revokeOriginalImageUrl() {
  if (originalImageUrl.value && originalImageUrl.value.startsWith('blob:')) {
    URL.revokeObjectURL(originalImageUrl.value);
  }
}

function resetProcessedResults(clearChartInfo = false) {
  processedImageUrl.value = '';
  processedStandardImageUrl.value = '';
  processedColoredImageUrl.value = '';
  gridLinePreviewMode.value = 'standard';
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

function setGridLinePreviewMode(mode) {
  gridLinePreviewMode.value = mode;
  if (mode === 'color' && processedColoredImageUrl.value) {
    processedImageUrl.value = processedColoredImageUrl.value;
    return;
  }
  if (mode !== 'original') {
    processedImageUrl.value = processedStandardImageUrl.value;
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

function datasetCategoryDisplayLabel(category) {
  const value = String(category?.value || '').toLowerCase();
  if (value === 'v_bar') return 'v-bar';
  if (value === 'h_bar') return 'h-bar';
  return category?.label || '';
}

function joinUrl(base, path) {
  if (!path) return '';
  const value = String(path);
  if (/^https?:\/\//i.test(value) || value.startsWith('blob:') || value.startsWith('data:')) {
    return value;
  }
  return `${String(base).replace(/\/+$/, '')}/${value.replace(/^\/+/, '')}`;
}

function staticPreviewUrl(path) {
  return joinUrl(STATIC_PREVIEW_BASE, path);
}

function backendUrl(path) {
  return joinUrl(API_URL, path);
}

function sampleImageSrc(sample) {
  if (!sample) return '';
  return sample.static_preview ? staticPreviewUrl(sample.image_url) : backendUrl(sample.image_url);
}

async function loadStaticPreviewManifest() {
  if (staticPreviewManifest.value) return staticPreviewManifest.value;
  const response = await axios.get(staticPreviewUrl('manifest.json'));
  staticPreviewManifest.value = response.data || {};
  return staticPreviewManifest.value;
}

function currentStaticSource() {
  const sources = staticPreviewManifest.value?.sources || {};
  return sources[datasetSource.value] || null;
}

function applyStaticDatasetCategories() {
  const source = currentStaticSource();
  datasetCategories.value = Array.isArray(source?.categories) ? source.categories : [];
  if (!datasetCategory.value || !datasetCategories.value.some((item) => item.value === datasetCategory.value)) {
    datasetCategory.value = datasetCategories.value[0]?.value || '';
  }
}

function applyStaticDatasetSamples() {
  const source = currentStaticSource();
  const samples = Array.isArray(source?.samples) ? source.samples : [];
  datasetSamples.value = samples
    .filter((sample) => !datasetCategory.value || datasetCategory.value === 'all' || sample.category === datasetCategory.value)
    .slice(0, 36);
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
  messageCountdown.value = 0;
}

function applyProcessedImageUrls(data, staticPreview = false) {
  const resolveUrl = staticPreview ? staticPreviewUrl : backendUrl;
  processedStandardImageUrl.value = data.standard_grid_url || data.encrypted_image_url
    ? resolveUrl(data.standard_grid_url || data.encrypted_image_url)
    : '';
  processedColoredImageUrl.value = data.colored_grid_url ? resolveUrl(data.colored_grid_url) : '';
  if (processedStandardImageUrl.value) {
    setGridLinePreviewMode('standard');
  }
}

async function loadEvaluationResultsFromUrl(resultsUrl, requestChartId, staticPreview = false) {
  if (!resultsUrl) return false;
  const response = await axios.get(staticPreview ? staticPreviewUrl(resultsUrl) : backendUrl(resultsUrl));
  if (requestChartId && requestChartId !== chartId.value) return false;
  evaluationResults.value = response.data;
  evaluationView.value = 'visual';
  return true;
}

async function fetchDatasetSamples() {
  isLoadingDataset.value = true;
  try {
    if (staticPreviewMode.value) {
      await loadStaticPreviewManifest();
      applyStaticDatasetCategories();
      applyStaticDatasetSamples();
      return;
    }
    const response = await axios.get(`${API_URL}/api/dataset-preview/samples/`, {
      params: { source: datasetSource.value, category: datasetCategory.value, limit: 36 },
    });
    if (Array.isArray(response.data.categories) && response.data.categories.length) {
      datasetCategories.value = response.data.categories;
    }
    if (response.data.category) {
      datasetCategory.value = response.data.category;
    }
    datasetSamples.value = response.data.samples || [];
  } catch (error) {
    console.error('Dataset sample load error:', error);
    try {
      staticPreviewMode.value = true;
      await loadStaticPreviewManifest();
      applyStaticDatasetCategories();
      applyStaticDatasetSamples();
    } catch (staticError) {
      console.error('Static dataset sample load error:', staticError);
    }
  } finally {
    isLoadingDataset.value = false;
  }
}

async function fetchDatasetCategories() {
  isLoadingDataset.value = true;
  try {
    if (staticPreviewMode.value) {
      await loadStaticPreviewManifest();
      applyStaticDatasetCategories();
      datasetSamples.value = [];
      return;
    }
    const response = await axios.get(`${API_URL}/api/dataset-preview/categories/`, {
      params: { source: datasetSource.value },
    });
    datasetCategories.value = response.data.categories || [];
    datasetCategory.value = datasetCategories.value[0]?.value || '';
    datasetSamples.value = [];
  } catch (error) {
    console.error('Dataset category load error:', error);
    try {
      staticPreviewMode.value = true;
      await loadStaticPreviewManifest();
      applyStaticDatasetCategories();
      datasetSamples.value = [];
    } catch (staticError) {
      console.error('Static dataset category load error:', staticError);
      datasetCategories.value = [];
      datasetCategory.value = '';
    }
  } finally {
    isLoadingDataset.value = false;
  }
}

async function setDatasetSource(source) {
  if (datasetSource.value === source || isLoadingDataset.value) return;
  datasetSource.value = source;
  datasetCategory.value = '';
  datasetCategories.value = [];
  selectedDatasetSampleId.value = '';
  selectedStaticDatasetSample.value = null;
  datasetSamples.value = [];
  await fetchDatasetCategories();
  fetchDatasetSamples();
}

function setDatasetCategory(category) {
  if (datasetCategory.value === category || isLoadingDataset.value) return;
  datasetCategory.value = category;
  selectedDatasetSampleId.value = '';
  selectedStaticDatasetSample.value = null;
  datasetSamples.value = [];
  fetchDatasetSamples();
}

async function selectDatasetSample(sample) {
  if (!sample || isLoadingDataset.value) return;
  clearMessages();
  resetProcessedResults(true);
  imageFile.value = null;
  selectedDatasetSampleId.value = sample.sample_id;
  selectedStaticDatasetSample.value = sample.static_preview ? sample : null;
  isLoadingDataset.value = true;

  try {
    if (sample.static_preview) {
      chartId.value = sample.chart_id || `dataset_${sample.sample_id}`;
      chartType.value = sample.chart_type;
      confidence.value = sample.confidence ?? 1.0;
      revokeOriginalImageUrl();
      originalImageUrl.value = staticPreviewUrl(sample.image_url);
      applyProcessedImageUrls(sample, true);
      let hasEvaluationCache = false;
      if (sample.evaluated && sample.results_url) {
        hasEvaluationCache = await loadEvaluationResultsFromUrl(sample.results_url, chartId.value, true);
      }
      if (sample.cached && hasEvaluationCache) {
        statusMessage.value = 'Loaded static cached grid and prediction preview.';
      } else if (sample.cached) {
        statusMessage.value = 'Loaded static cached grid preview.';
      } else {
        statusMessage.value = 'Loaded static dataset sample. No cached grid is available.';
      }
      return;
    }
    const response = await axios.post(
      `${API_URL}/api/dataset-preview/select/`,
      null,
      { params: { sample_id: sample.sample_id } }
    );
    const data = response.data;
    chartId.value = data.chart_id;
    chartType.value = data.chart_type;
    confidence.value = data.confidence;
    revokeOriginalImageUrl();
    originalImageUrl.value = backendUrl(data.original_image_url || sample.image_url);
    applyProcessedImageUrls(data);
    let hasEvaluationCache = false;
    if (data.evaluated && data.results_url) {
      hasEvaluationCache = await loadEvaluationResultsFromUrl(data.results_url, data.chart_id);
    }
    if (data.cached && hasEvaluationCache) {
      statusMessage.value = '已载入数据集样例，并复用缓存的网格和评估预测结果。';
    } else if (data.cached) {
      statusMessage.value = '已载入数据集样例，并复用缓存网格结果。';
    } else {
      statusMessage.value = '已载入数据集样例，可直接进行加密处理。';
    }
  } catch (error) {
    console.error('Dataset sample select error:', error);
    errorMessage.value = `数据集样例载入失败: ${error.response?.data?.detail || error.message}`;
  } finally {
    isLoadingDataset.value = false;
  }
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

  if (selectedStaticDatasetSample.value) {
    clearMessages();
    applyProcessedImageUrls(selectedStaticDatasetSample.value, true);
    if (processedStandardImageUrl.value || processedColoredImageUrl.value) {
      statusMessage.value = 'Loaded cached static encryption preview.';
    } else {
      errorMessage.value = 'Static preview has no cached encryption result for this sample.';
    }
    return;
  }
  
  clearMessages();
  processedImageUrl.value = '';
  processedStandardImageUrl.value = '';
  processedColoredImageUrl.value = '';
  gridLinePreviewMode.value = 'standard';
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
    applyProcessedImageUrls(response.data);
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

  if (selectedStaticDatasetSample.value) {
    clearMessages();
    isLoadingEvaluate.value = true;
    evaluationResults.value = null;
    evaluationView.value = 'visual';
    try {
      if (!selectedStaticDatasetSample.value.results_url) {
        throw new Error('No cached evaluation result for this sample.');
      }
      await loadEvaluationResultsFromUrl(selectedStaticDatasetSample.value.results_url, chartId.value, true);
      statusMessage.value = 'Loaded cached static prediction preview.';
    } catch (error) {
      console.error("Static evaluation error:", error);
      errorMessage.value = `Static evaluation load failed: ${error.response?.data?.detail || error.message}`;
    } finally {
      isLoadingEvaluate.value = false;
    }
    return;
  }

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
    
    await loadEvaluationResultsFromUrl(evalResponse.data.results_url, requestChartId);
    statusMessage.value = '评估结果获取成功！';
  } catch (error) {
    console.error("Evaluation error:", error);
    errorMessage.value = `评估失败: ${error.response?.data?.detail || error.message}`;
  } finally {
    isLoadingEvaluate.value = false;
  }
}

onMounted(async () => {
  await fetchDatasetCategories();
  fetchDatasetSamples();
});

watch([statusMessage, errorMessage], startMessageCountdown);

onBeforeUnmount(() => {
  stopMessageCountdown();
});
</script>

<template>
  <div class="app-container">
    <Teleport to="body">
      <div v-if="statusMessage || errorMessage" class="floating-message-stack">
        <div v-if="statusMessage" class="message status">
          <span>{{ statusMessage }}</span>
          <span class="message-countdown">{{ messageCountdown }}s</span>
        </div>
        <div v-if="errorMessage" class="message error">
          <span>{{ errorMessage }}</span>
          <span class="message-countdown">{{ messageCountdown }}s</span>
        </div>
      </div>
    </Teleport>
    
    <main class="content-container">
      <!-- 三列并排布局 -->
      <div class="columns-wrapper">
        <!-- 左侧：输入与数据集区域 -->
        <div class="column upload-column">
          <div class="section-heading upload-heading">
            <div class="sidebar-brand">图表智能分析</div>
            <label
              class="clickable-heading"
              :class="{ disabled: isLoadingUpload || staticPreviewMode }"
              for="image-upload"
              role="button"
              tabindex="0"
            >
              1. 上传文件{{ isLoadingUpload ? '中...' : '' }}
            </label>
            <input
              id="image-upload"
              class="hidden-file-input"
              type="file"
              @change="handleImageUpload"
              accept="image/png, image/jpeg"
              :disabled="isLoadingUpload || staticPreviewMode"
            />
          </div>

          <div class="dataset-preview-panel" :class="{ compact: chartId }">
            <div class="dataset-preview-header">
              <h3>数据集快速预览</h3>
              <div class="dataset-source-toggle" role="group" aria-label="dataset source">
                <button
                  v-for="option in datasetSourceOptions"
                  :key="option.value"
                  type="button"
                  :class="{ active: datasetSource === option.value }"
                  :disabled="isLoadingDataset"
                  @click="setDatasetSource(option.value)"
                >
                  {{ option.label }}
                </button>
              </div>
              <button type="button" class="text-button" @click="fetchDatasetSamples" :disabled="isLoadingDataset">
                {{ isLoadingDataset ? '加载中' : '刷新' }}
              </button>
            </div>
            <div v-if="datasetCategories.length" class="dataset-category-list" role="group" aria-label="dataset category">
              <button
                v-for="category in datasetCategories"
                :key="category.value"
                type="button"
                :class="{ active: datasetCategory === category.value }"
                :disabled="isLoadingDataset"
                @click="setDatasetCategory(category.value)"
              >
                <span>{{ datasetCategoryDisplayLabel(category) }}</span>
                <small>{{ category.count }}</small>
              </button>
            </div>
            <div v-if="datasetSamples.length" class="dataset-sample-grid">
              <button
                v-for="sample in datasetSamples"
                :key="sample.sample_id"
                type="button"
                class="dataset-sample-button"
                :class="{ active: selectedDatasetSampleId === sample.sample_id }"
                :disabled="isLoadingDataset"
                @click="selectDatasetSample(sample)"
              >
                <img :src="sampleImageSrc(sample)" :alt="sample.name" />
              </button>
            </div>
            <div v-else class="dataset-empty">
              {{ isLoadingDataset ? '正在加载数据集样例' : '暂无可用样例' }}
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
          <div class="section-heading process-heading">
            <h2
              class="clickable-heading"
              :class="{ disabled: isProcessDisabled }"
              role="button"
              tabindex="0"
              @click="processImage"
              @keydown.enter.prevent="processImage"
              @keydown.space.prevent="processImage"
            >
              2. 加密处理{{ isLoadingProcess ? '中...' : '' }}
            </h2>
            <div class="heading-actions">
              <div v-if="originalImageUrl || hasGridPreview" class="grid-preview-toggle compact" role="group" aria-label="grid line preview mode">
                <button
                  type="button"
                  :class="{ active: gridLinePreviewMode === 'original' }"
                  :disabled="!originalImageUrl"
                  @click="setGridLinePreviewMode('original')"
                >
                  原图
                </button>
                <button
                  type="button"
                  :class="{ active: gridLinePreviewMode === 'standard' }"
                  :disabled="!processedStandardImageUrl"
                  @click="setGridLinePreviewMode('standard')"
                >
                  标准灰色
                </button>
                <button
                  type="button"
                  :class="{ active: gridLinePreviewMode === 'color' }"
                  :disabled="!hasColoredGridPreview"
                  @click="setGridLinePreviewMode('color')"
                >
                  彩色预览
                </button>
              </div>
            </div>
          </div>

          <div class="process-image-stack single">
            <div v-if="previewImageUrl" class="large-image-container compact-image-container">
              <h3 class="image-inside-title">{{ previewImageTitle }}</h3>
              <img :src="previewImageUrl" alt="Chart Preview" />
            </div>
            <div v-else class="placeholder compact-image-placeholder">
              <h3 class="image-inside-title">{{ previewImageTitle }}</h3>
              <p>{{ previewPlaceholderText }}</p>
            </div>
          </div>
          <div v-if="processedImageUrl && !hasColoredGridPreview" class="preview-hint">
            当前样例仅提供标准灰色网格预览。
          </div>
        </div>

        <!-- 右侧：评估区域 -->
        <div class="column evaluation-column">
          <div class="section-heading evaluation-heading">
            <h2
              class="clickable-heading"
              :class="{ disabled: isEvaluateDisabled }"
              role="button"
              tabindex="0"
              @click="evaluateChart"
              @keydown.enter.prevent="evaluateChart"
              @keydown.space.prevent="evaluateChart"
            >
              3. 评估预测{{ isLoadingEvaluate ? '中...' : '' }}
            </h2>
            <div v-if="evaluationResults" class="view-toggle compact" role="group" aria-label="prediction result view">
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
          </div>

          <div class="evaluation-preview">
            <div v-if="evaluationResults" class="table-container">
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
:global(*) {
  box-sizing: border-box;
}

:global(html),
:global(body) {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

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
  height: 100vh;
  overflow: hidden;
}

.content-container {
  max-width: 1800px;
  margin: 0 auto;
  padding: 1rem 1.25rem;
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
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

.floating-message-stack {
  position: fixed;
  top: max(0.35rem, env(safe-area-inset-top));
  left: 50%;
  z-index: 1200;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  width: min(720px, calc(100vw - 2rem));
  transform: translateX(-50%);
  pointer-events: none;
}

.floating-message-stack .message {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 1rem;
  margin: 0;
  padding: 0.55rem 0.85rem;
  border: 1px solid rgba(255, 255, 255, 0.45);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
  font-size: 0.9rem;
  text-align: center;
}

.floating-message-stack .message > span:first-child {
  grid-column: 2;
  justify-self: center;
}

.message-countdown {
  grid-column: 3;
  justify-self: end;
  flex: 0 0 auto;
  min-width: 2.4rem;
  border-radius: 999px;
  background-color: rgba(255, 255, 255, 0.24);
  padding: 0.1rem 0.45rem;
  text-align: center;
  font-weight: 700;
}

@media (max-width: 900px) {
  .floating-message-stack {
    position: fixed;
    top: max(0.5rem, env(safe-area-inset-top));
    width: min(92vw, 720px);
  }
}

/* 三列布局 */
.columns-wrapper {
  display: grid;
  grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.25fr) minmax(360px, 1fr);
  gap: 1.5rem;
  flex: 1;
  height: 100%;
  min-height: 0;
  overflow: hidden;
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
  min-height: 0;
  overflow: hidden;
}

.evaluation-column {
  min-height: 0;
}

.sidebar-brand {
  margin: 0;
  color: #1a1a1a;
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.2;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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

.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  border-bottom: 2px solid #0066cc;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  min-height: 44px;
  box-sizing: border-box;
}

.section-heading h2,
.section-heading label {
  border-bottom: 0;
  margin: 0;
  padding-bottom: 0;
}

.section-heading .clickable-heading {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  height: 36px;
  border: 0;
  box-sizing: border-box;
  cursor: pointer;
  border-radius: 4px;
  background-color: #0066cc;
  color: #ffffff;
  font-size: 0.96rem;
  font-weight: 700;
  line-height: 1.2;
  padding: 0 0.95rem;
  margin-left: 0;
  white-space: nowrap;
}

.section-heading .clickable-heading:hover:not(.disabled),
.section-heading .clickable-heading:focus-visible:not(.disabled) {
  background-color: #0052a3;
  color: #ffffff;
  outline: none;
}

.section-heading .clickable-heading.disabled {
  background-color: #d6dce3;
  color: #ffffff;
  cursor: not-allowed;
  opacity: 1;
}

.process-heading {
  align-items: center;
}

.evaluation-heading {
  align-items: center;
}

.heading-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 0.65rem;
  min-width: 0;
  flex: 1 1 auto;
}

.upload-heading {
  align-items: center;
}

.upload-heading .clickable-heading {
  flex: 0 0 auto;
}

.hidden-file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
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
.dataset-preview-panel {
  margin-bottom: 1rem;
  border: 1px solid #d9e1ea;
  border-radius: 6px;
  padding: 0.8rem;
  background-color: #fbfcfe;
}

.dataset-preview-panel.compact {
  margin-bottom: 0.75rem;
  padding: 0.65rem;
}

.dataset-preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

.dataset-preview-header h3 {
  flex: 0 0 auto;
  margin: 0;
  font-size: 0.95rem;
  white-space: nowrap;
}

.text-button {
  flex: 0 0 auto;
  border: 1px solid #b8c7d6;
  border-radius: 4px;
  background-color: #ffffff;
  color: #1f3a56;
  cursor: pointer;
  min-height: 30px;
  padding: 0 0.55rem;
  font-weight: 600;
  font-size: 0.82rem;
}

.text-button:disabled {
  cursor: not-allowed;
  color: #8a96a3;
}

.dataset-source-toggle {
  display: grid;
  grid-template-columns: 1fr 1fr;
  flex: 0 1 174px;
  min-width: 150px;
  border: 1px solid #c8d2dc;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0;
}

.dataset-source-toggle button {
  min-height: 30px;
  border: 0;
  background-color: #f5f7fa;
  color: #243447;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.7rem;
  padding: 0 0.25rem;
  white-space: nowrap;
}

.dataset-source-toggle button + button {
  border-left: 1px solid #c8d2dc;
}

.dataset-source-toggle button.active {
  background-color: #0066cc;
  color: #ffffff;
}

.dataset-source-toggle button:disabled {
  cursor: not-allowed;
  color: #8a96a3;
}

.dataset-category-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  max-height: 112px;
  overflow: auto;
  margin-bottom: 0.75rem;
  padding-right: 0.2rem;
}

.dataset-preview-panel.compact .dataset-category-list {
  gap: 0.35rem;
  max-height: 88px;
  margin-bottom: 0.65rem;
}

.dataset-category-list button {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  min-height: 30px;
  max-width: 100%;
  border: 1px solid #c8d2dc;
  border-radius: 4px;
  background-color: #ffffff;
  color: #243447;
  cursor: pointer;
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0 0.55rem;
}

.dataset-preview-panel.compact .dataset-category-list button {
  min-height: 26px;
  font-size: 0.74rem;
  padding: 0 0.45rem;
}

.dataset-category-list button.active {
  border-color: #0066cc;
  background-color: #eaf3ff;
  color: #004f9e;
}

.dataset-category-list button:disabled {
  cursor: not-allowed;
  color: #8a96a3;
}

.dataset-category-list small {
  color: #607086;
  font-size: 0.72rem;
  font-weight: 700;
}

.dataset-sample-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.6rem;
  max-height: 300px;
  overflow: auto;
  padding-right: 0.25rem;
}

.dataset-preview-panel.compact .dataset-sample-grid {
  gap: 0.55rem;
  max-height: 250px;
}

.dataset-sample-button {
  display: block;
  border: 1px solid #d4dde7;
  border-radius: 4px;
  background-color: #ffffff;
  color: #182635;
  cursor: pointer;
  padding: 0.3rem;
}

.dataset-preview-panel.compact .dataset-sample-button {
  padding: 0.28rem;
}

.dataset-sample-button.active {
  border-color: #0066cc;
  box-shadow: 0 0 0 2px rgba(0, 102, 204, 0.12);
}

.dataset-sample-button:disabled {
  cursor: wait;
}

.dataset-sample-button img {
  width: 100%;
  height: 96px;
  object-fit: contain;
  background-color: #ffffff;
  border: 1px solid #eef1f4;
}

.dataset-preview-panel.compact .dataset-sample-button img {
  height: 82px;
}

.dataset-empty {
  min-height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #516274;
  border: 1px dashed #cbd6e2;
  border-radius: 4px;
  font-size: 0.9rem;
}

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
  position: relative;
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

.process-image-stack {
  flex: 1;
  display: grid;
  grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
  gap: 0.7rem;
  min-height: 0;
}

.process-image-stack.single {
  display: flex;
  flex-direction: column;
}

.process-image-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.compact-image-container,
.compact-image-placeholder {
  min-height: 0;
}

.compact-image-container {
  padding: 1.6rem 0.5rem 0.5rem;
}

.image-inside-title {
  position: absolute;
  top: 0.35rem;
  left: 0.55rem;
  z-index: 2;
  margin: 0;
  padding: 0.1rem 0.45rem;
  border: 1px solid #d5dde7;
  border-radius: 4px;
  background-color: rgba(255, 255, 255, 0.92);
  color: #243447;
  font-size: 0.82rem;
  font-weight: 700;
  line-height: 1.35;
}

.preview-hint {
  margin-top: 0.75rem;
  color: #5b6878;
  font-size: 0.86rem;
}

/* 评估结果区域 */
.evaluation-preview {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

.evaluation-preview > .placeholder {
  min-height: 0;
}

.table-container {
  flex: 1;
  min-height: 0;
  max-height: 100%;
  overflow-y: auto;
  overflow-x: auto;
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

.view-toggle,
.grid-preview-toggle {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: 1fr;
  gap: 0;
  margin: 0 0 1rem;
  border: 1px solid #c8d2dc;
  border-radius: 4px;
  overflow: hidden;
}

.grid-preview-toggle {
  margin: 0 0 0.75rem;
}

.grid-preview-toggle.compact {
  width: 300px;
  min-width: 270px;
  height: 36px;
  margin: 0;
  flex: 0 0 auto;
}

.view-toggle.compact {
  width: 180px;
  height: 36px;
  margin: 0;
  flex: 0 0 auto;
}

.grid-preview-toggle.compact button {
  min-height: 34px;
  font-size: 0.86rem;
}

.view-toggle.compact button {
  min-height: 34px;
  font-size: 0.86rem;
}

.view-toggle button,
.grid-preview-toggle button {
  min-height: 40px;
  border: 0;
  background-color: #f5f7fa;
  color: #243447;
  cursor: pointer;
  font-weight: 600;
}

.view-toggle button + button,
.grid-preview-toggle button + button {
  border-left: 1px solid #c8d2dc;
}

.view-toggle button.active,
.grid-preview-toggle button.active {
  background-color: #0066cc;
  color: #ffffff;
}

.grid-preview-toggle button:disabled {
  cursor: not-allowed;
  color: #8a96a3;
}

.prediction-visual,
.detail-view {
  min-height: 0;
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
  position: relative;
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
  overflow: hidden;
}

/* 图表信息区域样式 */
.chart-info {
  margin-top: 0.75rem;
  border-top: 1px solid #e0e0e0;
  padding-top: 0.75rem;
}

.chart-info h3 {
  margin-top: 0;
  margin-bottom: 0.45rem;
  font-weight: 500;
  color: #333333;
  font-size: 0.95rem;
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.4rem;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  padding: 0.4rem 0.55rem;
  background-color: #f9f9f9;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  min-height: 28px;
}

.info-label {
  font-weight: 500;
  color: #333333;
  font-size: 0.82rem;
  white-space: nowrap;
}

.info-value {
  font-weight: 600;
  color: #0066cc;
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.8rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 响应式设计 */
@media (max-width: 900px) {
  :global(html),
  :global(body) {
    width: 100%;
    max-width: 100vw;
    height: auto;
    min-height: 100%;
    overflow-x: hidden;
    overflow-y: auto;
  }

  .app-container {
    width: 100%;
    max-width: 100vw;
    height: auto;
    min-height: 100vh;
    overflow-x: hidden;
    overflow-y: visible;
  }

  .content-container {
    width: 100%;
    max-width: 100vw;
    height: auto;
    min-height: 100vh;
    overflow-x: hidden;
    overflow-y: visible;
  }

  .columns-wrapper {
    grid-template-columns: 1fr;
    width: 100%;
    max-width: 100%;
    height: auto;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: visible;
    gap: 1rem;
  }
  
  .column {
    width: 100%;
    max-width: 100%;
    height: auto;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: visible;
  }
  
  .large-image-container,
  .placeholder {
    min-height: 260px;
  }

  .process-image-stack.single,
  .evaluation-preview,
  .table-container {
    min-height: 320px;
  }
}

@media (max-width: 768px) {
  .content-container {
    padding: 0.75rem;
  }
  
  .column h2 {
    font-size: 1.2rem;
  }

  .sidebar-brand {
    font-size: 1.05rem;
  }

  .column {
    padding: 1rem;
  }

  .section-heading {
    gap: 0.65rem;
    min-width: 0;
  }

  .upload-heading {
    flex-wrap: wrap;
  }

  .sidebar-brand {
    min-width: 0;
    flex: 1 1 auto;
  }

  .upload-heading .clickable-heading {
    flex: 0 0 auto;
  }

  .process-heading,
  .evaluation-heading {
    flex-wrap: wrap;
    align-items: flex-start;
  }

  .heading-actions {
    min-width: 0;
    flex: 1 1 100%;
    justify-content: flex-start;
  }

  .grid-preview-toggle.compact,
  .view-toggle.compact {
    width: 100%;
    min-width: 0;
  }

  .dataset-preview-header {
    min-width: 0;
    flex-wrap: wrap;
    justify-content: flex-start;
  }

  .dataset-preview-header h3 {
    flex: 1 1 auto;
    min-width: 0;
  }

  .dataset-source-toggle {
    flex: 1 1 100%;
    min-width: 0;
  }

  .dataset-sample-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    max-height: none;
  }

  .dataset-preview-panel.compact .dataset-sample-grid {
    max-height: none;
  }

  .dataset-sample-button img,
  .dataset-preview-panel.compact .dataset-sample-button img {
    height: 110px;
  }
}
</style>
