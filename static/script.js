document.addEventListener('DOMContentLoaded', () => {
    const modelSelect = document.getElementById('model-select');
    const form = document.getElementById('predict-form');
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('dropzone');
    const fileNameDisplay = document.getElementById('file-name');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    // Result elements
    const resultsCard = document.getElementById('results-card');
    const riskBanner = document.getElementById('risk-banner');
    const riskBadge = document.getElementById('risk-badge');
    const probValue = document.getElementById('prob-value');
    const confVal = document.getElementById('conf-val');
    const clipsVal = document.getElementById('clips-val');
    const stdVal = document.getElementById('std-val');
    const ensembleStdItem = document.getElementById('ensemble-std-item');
    const modelStdVal = document.getElementById('model-std-val');
    const abstainAlert = document.getElementById('abstain-alert');
    const abstainReason = document.getElementById('abstain-reason');
    const enableAbstainToggle = document.getElementById('enable-abstain');

    // Fetch available models
    fetch('/models')
        .then(res => res.json())
        .then(data => {
            modelSelect.innerHTML = '';
            
            // Group ensemble at top
            if (data.models.includes('ensemble')) {
                const opt = document.createElement('option');
                opt.value = 'ensemble';
                opt.textContent = 'Ensemble (Mean of all folds)';
                opt.style.fontWeight = 'bold';
                modelSelect.appendChild(opt);
            }

            data.models.forEach(model => {
                if (model !== 'ensemble') {
                    const opt = document.createElement('option');
                    opt.value = model;
                    opt.textContent = `Single Model: ${model.toUpperCase()}`;
                    modelSelect.appendChild(opt);
                }
            });
        });

    // File input handling
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
            dropzone.style.borderColor = 'var(--primary)';
        }
    });

    // Drag and drop effects
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
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            fileNameDisplay.textContent = fileInput.files[0].name;
            dropzone.style.borderColor = 'var(--primary)';
        }
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            alert('Please upload a .npy or video file.');
            return;
        }

        // UI Loading State
        submitBtn.disabled = true;
        btnText.textContent = 'Inferencing...';
        spinner.classList.remove('hidden');
        resultsCard.classList.add('hidden');
        abstainAlert.classList.add('hidden');

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('model_selection', modelSelect.value);

            const res = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                displayResults(data, modelSelect.value === 'ensemble');
            }
        } catch (err) {
            alert(`Request failed: ${err.message}`);
        } finally {
            // Reset UI
            submitBtn.disabled = false;
            btnText.textContent = 'Run Inference';
            spinner.classList.add('hidden');
        }
    });

    let motionChart = null;
    let lastResult = null;
    let lastFilename = "";

    function displayResults(data, isEnsemble) {
        let riskText = data.risk_level.replace('_', ' ');
        let bannerClass = data.risk_level.toLowerCase();
        
        const enableAbstain = enableAbstainToggle.checked;
        if (!enableAbstain) {
            if (data.final_prob >= 0.5) {
                riskText = "ASD";
                bannerClass = "high_risk";
            } else {
                riskText = "TD";
                bannerClass = "low_risk";
            }
        }
        
        riskBanner.className = 'result-header ' + bannerClass;
        riskBadge.textContent = riskText;
        lastResult = data;
        lastFilename = fileInput.files[0] ? fileInput.files[0].name : "unknown_video";
        
        document.getElementById('download-report').classList.remove('hidden');
        probValue.textContent = Number(data.final_prob).toFixed(4);
        
        confVal.textContent = Number(data.details.confidence).toFixed(4);
        clipsVal.textContent = data.details.n_clips || '--';
        stdVal.textContent = data.details.clip_std ? Number(data.details.clip_std).toFixed(4) : '--';
        
        if (isEnsemble && data.details.model_agreement_std !== undefined) {
            ensembleStdItem.classList.remove('hidden');
            modelStdVal.textContent = Number(data.details.model_agreement_std).toFixed(4);
        } else {
            ensembleStdItem.classList.add('hidden');
        }

        if (data.risk_level === 'ABSTAIN' && enableAbstain) {
            abstainAlert.classList.remove('hidden');
            abstainReason.textContent = data.details.abstain_reason || 'Unknown reason';
        } else {
            abstainAlert.classList.add('hidden');
        }

        // --- Temporal Chart Implementation ---
        renderTemporalChart(data);
        renderMotionBreakdown(data.details.temporal_data);

        resultsCard.classList.remove('hidden');
        resultsCard.style.opacity = '0';
        resultsCard.style.transform = 'translateY(10px)';
        setTimeout(() => {
            resultsCard.style.transition = 'all 0.4s ease';
            resultsCard.style.opacity = '1';
            resultsCard.style.transform = 'translateY(0)';
        }, 50);
    }

    function renderTemporalChart(data) {
        const frameEnergies = data.details.frame_energies;
        if (!frameEnergies || frameEnergies.length === 0) return;

        const ctx = document.getElementById('motion-chart').getContext('2d');
        const totalFrames = data.details.total_frames || frameEnergies[frameEnergies.length - 1].frame;
        
        // Show as percentage of video duration for accuracy
        const labels = frameEnergies.map(d => {
            const pct = ((d.frame / totalFrames) * 100).toFixed(0);
            return `${pct}%`;
        });
        const totalEnergy = frameEnergies.map(d => d.energy);
        
        // Extract per-group energy if available
        const hasGroups = frameEnergies[0].groups !== undefined;
        const datasets = [];
        
        if (hasGroups) {
            const armEnergy = frameEnergies.map(d => d.groups.arms);
            const headEnergy = frameEnergies.map(d => d.groups.head);
            const legEnergy = frameEnergies.map(d => d.groups.legs);
        
            datasets.push({
                label: 'Arms',
                data: armEnergy,
                borderColor: '#f97316',
                backgroundColor: 'rgba(249, 115, 22, 0.05)',
                borderWidth: 2,
                fill: false,
                tension: 0.3,
                pointRadius: 0
            });
            datasets.push({
                label: 'Head',
                data: headEnergy,
                borderColor: '#a78bfa',
                backgroundColor: 'rgba(167, 139, 250, 0.05)',
                borderWidth: 2,
                fill: false,
                tension: 0.3,
                pointRadius: 0
            });
            datasets.push({
                label: 'Legs',
                data: legEnergy,
                borderColor: '#34d399',
                backgroundColor: 'rgba(52, 211, 153, 0.05)',
                borderWidth: 2,
                fill: false,
                tension: 0.3,
                pointRadius: 0
            });
        }
        
        // Total energy as the main filled area
        datasets.unshift({
            label: 'Total Motion',
            data: totalEnergy,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.15)',
            borderWidth: 3,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4
        });

        if (motionChart) {
            motionChart.destroy();
        }

        motionChart = new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { 
                        display: true, 
                        position: 'top',
                        labels: { color: '#94a3b8', boxWidth: 12, padding: 15 }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleColor: '#94a3b8',
                        bodyColor: '#f8fafc',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Motion Energy', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        title: { display: true, text: 'Time', color: '#94a3b8' },
                        grid: { display: false },
                        ticks: { color: '#94a3b8', maxRotation: 0, maxTicksLimit: 12 }
                    }
                }
            }
        });
    }

    function renderMotionBreakdown(temporalData) {
        const container = document.getElementById('motion-breakdown');
        if (!temporalData || temporalData.length === 0) {
            container.innerHTML = '<p class="placeholder-text">No temporal data available.</p>';
            return;
        }

        // Get top 3 peaks (descending prob)
        const sorted = [...temporalData].sort((a, b) => b.prob - a.prob);
        const peaks = sorted.slice(0, 3);
        
        // Define "Rapid Motion" threshold for root-centered landmarks
        const RAPID_THRESHOLD = 0.08; 

        container.innerHTML = '';
        peaks.forEach((peak, i) => {
            const totalFrames = temporalData[temporalData.length - 1]?.end_frame || 1;
            const startPct = ((peak.start_frame / totalFrames) * 100).toFixed(0);
            const endPct = ((peak.end_frame / totalFrames) * 100).toFixed(0);
            
            const sigs = peak.signatures;
            const activeGroups = Object.entries(sigs).filter(([_, energy]) => energy > RAPID_THRESHOLD);
            
            // Only create card if there's actual significant motion or very high P(ASD)
            if (activeGroups.length === 0 && peak.prob < 0.6) return;

            const card = document.createElement('div');
            card.className = 'motion-card';
            card.style.animationDelay = `${i * 0.1}s`;

            let statsHtml = '';
            if (activeGroups.length > 0) {
                // Sort active groups by energy
                activeGroups.sort((a,b) => b[1] - a[1]);
                activeGroups.forEach(([group, energy]) => {
                    statsHtml += `<span class="stat-pill high-energy">
                        🚀 ${group.toUpperCase()}: ${energy.toFixed(3)}
                    </span>`;
                });
            } else {
                statsHtml = `<span class="stat-pill">Subtle / Systematic movement</span>`;
            }

            card.innerHTML = `
                <div class="motion-card-header">
                    <span class="motion-time">⏳ Analysis Tier ${i+1} (${startPct}% - ${endPct}%)</span>
                    <span class="motion-prob">P(ASD) ${peak.prob.toFixed(3)}</span>
                </div>
                <div class="motion-stats">
                    ${statsHtml}
                </div>
            `;
            container.appendChild(card);
        });

        if (container.innerHTML === '') {
            container.innerHTML = '<p class="placeholder-text">No significant rapid motions detected in high-risk zones.</p>';
        }
    }
    
    // window.generateReport uses the variables defined at the top

    window.generateReport = async function() {
        if (!lastResult) return;
        
        const btn = document.getElementById('download-report');
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-small"></span> Generating...';
        
        try {
            // 1. Capture chart as image
            const chartCanvas = document.getElementById('motion-chart');
            const chartImage = chartCanvas.toDataURL('image/png');
            
            // 2. Prepare payload
            const payload = {
                filename: lastFilename,
                risk_level: lastResult.risk_level,
                final_prob: lastResult.final_prob,
                confidence: lastResult.details.confidence || 0,
                model_agreement: lastResult.details.model_agreement_std || 0,
                chart_image: chartImage,
                n_clips: lastResult.details.n_clips || 0,
                rapid_motion_detected: document.querySelectorAll('.motion-card').length > 0
            };
            
            // 3. Request PDF
            const res = await fetch('/generate-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!res.ok) throw new Error('Failed to generate report');
            
            // 4. Download file
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ASD_Report_${new Date().toISOString().slice(0,10)}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
        } catch (err) {
            alert(`Report generation failed: ${err.message}`);
        } finally {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    };
});
