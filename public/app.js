document.addEventListener('DOMContentLoaded', () => {
  // 1. Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      navTabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });

  // 2. State & DOM references
  let selectedSliceId = null;
  let selectedFile = null;

  const sampleChipsContainer = document.getElementById('sample-chips-container');
  const dropzone = document.getElementById('mri-dropzone');
  const fileInput = document.getElementById('mri-file-input');
  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const dropzonePreview = document.getElementById('dropzone-preview');
  const previewImage = document.getElementById('preview-image');
  const btnClearPreview = document.getElementById('btn-clear-preview');
  const modelArchSelect = document.getElementById('model-arch-select');
  const btnRunAnalysis = document.getElementById('btn-run-analysis');
  const btnText = btnRunAnalysis.querySelector('.btn-text');
  const btnLoader = btnRunAnalysis.querySelector('.btn-loader');

  // Report elements
  const emptyState = document.getElementById('empty-state');
  const reportContent = document.getElementById('report-content');
  const reportTimestamp = document.getElementById('report-timestamp');
  const predictedClassName = document.getElementById('predicted-class-name');
  const predictedConfidenceVal = document.getElementById('predicted-confidence-val');
  const barGlioma = document.getElementById('bar-glioma');
  const probGlioma = document.getElementById('prob-glioma');
  const barMeningioma = document.getElementById('bar-meningioma');
  const probMeningioma = document.getElementById('prob-meningioma');
  const barPituitary = document.getElementById('bar-pituitary');
  const probPituitary = document.getElementById('prob-pituitary');
  const camRawImg = document.getElementById('cam-raw-img');
  const camOverlayImg = document.getElementById('cam-overlay-img');
  const camOpacitySlider = document.getElementById('cam-opacity-slider');
  const opacityVal = document.getElementById('opacity-val');
  const pathologyDescription = document.getElementById('pathology-description');

  // Metrics Table & Modal
  const metricsTbody = document.getElementById('metrics-tbody');
  const imageModal = document.getElementById('image-modal');
  const modalBackdrop = document.getElementById('modal-backdrop');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const modalImage = document.getElementById('modal-image');
  const modalCaption = document.getElementById('modal-caption');

  // 3. Load Curated Test Samples
  fetch('/api/samples')
    .then(r => r.json())
    .then(samples => {
      sampleChipsContainer.innerHTML = '';
      samples.forEach((s, idx) => {
        const chip = document.createElement('div');
        chip.className = 'sample-chip';
        chip.dataset.sliceId = s.slice_id;

        const pillClass = `pill-${s.class_name}`;
        chip.innerHTML = `
          <div class="chip-info">
            <span class="chip-title">${s.title}</span>
            <span class="chip-meta">Patient ID: ${s.pid} | Slice #${s.slice_id}</span>
          </div>
          <span class="chip-pill ${pillClass}">${s.class_name}</span>
        `;

        chip.addEventListener('click', () => {
          document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('selected'));
          chip.classList.add('selected');
          selectedSliceId = s.slice_id;
          selectedFile = null;

          // Update preview
          previewImage.src = `/slices/${s.slice_id}.png`;
          dropzonePrompt.classList.add('hidden');
          dropzonePreview.classList.remove('hidden');
        });

        sampleChipsContainer.appendChild(chip);

        // Pre-select first sample for immediate wow factor
        if (idx === 0) {
          chip.click();
        }
      });
    })
    .catch(err => {
      sampleChipsContainer.innerHTML = `<div class="sample-chip">Failed to load samples: ${err.message}</div>`;
    });

  // 4. File Upload Handling
  dropzone.addEventListener('click', (e) => {
    if (e.target !== btnClearPreview) {
      fileInput.click();
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    selectedFile = file;
    selectedSliceId = null;
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('selected'));

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      dropzonePrompt.classList.add('hidden');
      dropzonePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  btnClearPreview.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    selectedSliceId = null;
    fileInput.value = '';
    dropzonePrompt.classList.remove('hidden');
    dropzonePreview.classList.add('hidden');
    previewImage.src = '';
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('selected'));
  });

  // 5. Run Prediction & Generate Report
  btnRunAnalysis.addEventListener('click', () => {
    if (!selectedSliceId && !selectedFile) {
      alert('Please select a verified clinical test sample or upload an MRI slice first.');
      return;
    }

    const formData = new FormData();
    const modelArch = modelArchSelect.value;
    formData.append('model_arch', modelArch);

    if (selectedFile) {
      formData.append('mri_file', selectedFile);
    } else if (selectedSliceId) {
      formData.append('slice_id', selectedSliceId);
    }

    // UI Loading state
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');
    btnRunAnalysis.disabled = true;

    fetch('/api/predict', {
      method: 'POST',
      body: formData
    })
      .then(r => {
        if (!r.ok) throw new Error(`Inference returned HTTP status ${r.status}`);
        return r.json();
      })
      .then(res => {
        renderReport(res);
      })
      .catch(err => {
        alert('Analysis error: ' + err.message);
      })
      .finally(() => {
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
        btnRunAnalysis.disabled = false;
      });
  });

  function renderReport(data) {
    emptyState.classList.add('hidden');
    reportContent.classList.remove('hidden');

    const now = new Date();
    reportTimestamp.textContent = `Analyzed: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;

    // Diagnosis & Class Colors
    const predClass = data.predicted_class.toUpperCase();
    predictedClassName.textContent = predClass;

    const confPct = (data.confidence * 100).toFixed(1);
    predictedConfidenceVal.textContent = `${confPct}%`;

    if (data.predicted_class === 'glioma') {
      predictedClassName.style.color = 'var(--color-glioma)';
    } else if (data.predicted_class === 'meningioma') {
      predictedClassName.style.color = 'var(--color-meningioma)';
    } else {
      predictedClassName.style.color = 'var(--color-pituitary)';
    }

    // Probabilities
    const probs = data.probabilities;
    const gPct = (probs.glioma * 100).toFixed(1);
    const mPct = (probs.meningioma * 100).toFixed(1);
    const pPct = (probs.pituitary * 100).toFixed(1);

    probGlioma.textContent = `${gPct}%`;
    barGlioma.style.width = `${gPct}%`;

    probMeningioma.textContent = `${mPct}%`;
    barMeningioma.style.width = `${mPct}%`;

    probPituitary.textContent = `${pPct}%`;
    barPituitary.style.width = `${pPct}%`;

    // Grad-CAM Viewports
    camRawImg.src = data.source_image_url;
    camOverlayImg.src = data.cam_url;
    camOverlayImg.style.opacity = (camOpacitySlider.value / 100).toString();

    // Clinical Profile
    const profile = data.clinical_profile || {};
    pathologyDescription.innerHTML = `
      <strong>${profile.title || predClass}:</strong> ${profile.prevalence || ''}
      <br><br>
      <strong>Radiological Manifestation:</strong> ${profile.features || ''}
      <br><br>
      <strong>Clinical Management Protocol:</strong> ${profile.management || ''}
      <br><br>
      <span style="font-size: 0.72rem; color: var(--accent-cyan);">Validation Note: Evaluated on unseen patient cohort (Zero Patient Leakage protocol).</span>
    `;

    // Smooth scroll to results if on mobile
    if (window.innerWidth < 1024) {
      reportContent.scrollIntoView({ behavior: 'smooth' });
    }
  }

  // 6. Heatmap Opacity Slider
  camOpacitySlider.addEventListener('input', (e) => {
    const val = e.target.value;
    opacityVal.textContent = `${val}%`;
    if (camOverlayImg) {
      camOverlayImg.style.opacity = (val / 100).toString();
    }
  });

  // 7. Load Benchmark Metrics Table
  fetch('/api/metrics')
    .then(r => r.json())
    .then(records => {
      metricsTbody.innerHTML = '';
      records.forEach(row => {
        const tr = document.createElement('tr');
        const isResnet = row.model === 'RESNET50' && row.split_type === 'patient_disjoint';
        if (isResnet) tr.className = 'highlight-row';

        const fmtPct = (val) => val !== null && val !== undefined ? (val * 100).toFixed(2) + '%' : 'N/A';
        const fmtAuc = (val) => val !== null && val !== undefined ? (typeof val === 'number' ? val.toFixed(4) : val) : 'N/A';

        const splitLabel = row.split_type === 'patient_disjoint'
          ? '<span class="badge badge-success" style="font-size:0.68rem; padding: 0.2rem 0.5rem;">Patient-Disjoint</span>'
          : '<span class="badge badge-danger" style="font-size:0.68rem; padding: 0.2rem 0.5rem;">Random Leaky</span>';

        tr.innerHTML = `
          <td><strong>${row.model}</strong> ${isResnet ? '<span style="color:var(--accent-cyan); font-size:0.75rem;">★ SOTA</span>' : ''}</td>
          <td>${splitLabel}</td>
          <td>${fmtPct(row.accuracy)}</td>
          <td><strong>${fmtPct(row.macro_f1)}</strong></td>
          <td>${fmtPct(row.balanced_accuracy)}</td>
          <td>${fmtPct(row.meningioma_f1)}</td>
          <td>${fmtPct(row.glioma_f1)}</td>
          <td>${fmtPct(row.pituitary_f1)}</td>
          <td>${fmtAuc(row.macro_roc_auc)}</td>
        `;
        metricsTbody.appendChild(tr);
      });
    })
    .catch(err => {
      metricsTbody.innerHTML = `<tr><td colspan="9">Failed to load metrics: ${err.message}</td></tr>`;
    });

  // 8. Lightbox Zoom for Figures
  const zoomablePlots = document.querySelectorAll('.zoomable-plot');
  zoomablePlots.forEach(img => {
    img.addEventListener('click', () => {
      modalImage.src = img.src;
      modalCaption.textContent = img.alt || 'Visual Inspection';
      imageModal.classList.remove('hidden');
    });
  });

  const closeModal = () => imageModal.classList.add('hidden');
  modalCloseBtn.addEventListener('click', closeModal);
  modalBackdrop.addEventListener('click', closeModal);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !imageModal.classList.contains('hidden')) {
      closeModal();
    }
  });
});
