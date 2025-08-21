// Global variables
let currentStep = 1;
const totalSteps = 6;
let formData = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeForm();
    updateSliderValues();
    setupEventListeners();
    showStep(currentStep);
});

// Initialize form and set up initial state
function initializeForm() {
    // Initialize progress bar
    updateProgressBar();
    
    // Set up slider value displays
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        updateSliderDisplay(slider);
        slider.addEventListener('input', function() {
            updateSliderDisplay(this);
        });
    });
    
    // Set up form validation
    setupFormValidation();
}

// Set up event listeners
function setupEventListeners() {
    // Form input listeners for real-time validation
    const inputs = document.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('change', function() {
            validateStep(currentStep);
        });
        
        input.addEventListener('input', function() {
            if (this.type !== 'range') {
                validateStep(currentStep);
            }
        });
    });
    
    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && currentStep < totalSteps) {
            e.preventDefault();
            changeStep(1);
        } else if (e.key === 'ArrowLeft' && currentStep > 1) {
            e.preventDefault();
            changeStep(-1);
        } else if (e.key === 'ArrowRight' && currentStep < totalSteps) {
            e.preventDefault();
            changeStep(1);
        }
    });
}

// Update slider display values
function updateSliderValues() {
    const sliders = [
        { id: 'academicPressure', displayId: 'academicPressureValue' },
        { id: 'studySatisfaction', displayId: 'studySatisfactionValue' },
        { id: 'workPressure', displayId: 'workPressureValue' },
        { id: 'jobSatisfaction', displayId: 'jobSatisfactionValue' }
    ];
    
    sliders.forEach(slider => {
        const element = document.getElementById(slider.id);
        const display = document.getElementById(slider.displayId);
        
        if (element && display) {
            element.addEventListener('input', function() {
                display.textContent = this.value;
                
                // Update slider thumb position for the floating value
                const percent = ((this.value - this.min) / (this.max - this.min)) * 100;
                display.style.left = `${percent}%`;
            });
            
            // Initialize display
            display.textContent = element.value;
        }
    });
}

// Update individual slider display
function updateSliderDisplay(slider) {
    const displayId = slider.id + 'Value';
    const display = document.getElementById(displayId);
    
    if (display) {
        display.textContent = slider.value;
        
        // Calculate position for floating value
        const percent = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
        display.style.left = `${percent}%`;
    }
}

// Show specific step
function showStep(step) {
    // Hide all steps
    const steps = document.querySelectorAll('.step');
    steps.forEach(s => s.classList.remove('active'));
    
    // Show current step
    const currentStepElement = document.getElementById(`step${step}`);
    if (currentStepElement) {
        currentStepElement.classList.add('active');
    }
    
    // Update navigation buttons
    updateNavigationButtons();
    
    // Update progress bar
    updateProgressBar();
    
    // Focus on first input of current step
    setTimeout(() => {
        const firstInput = currentStepElement?.querySelector('input, select');
        if (firstInput && step !== totalSteps) {
            firstInput.focus();
        }
    }, 300);
}

// Update navigation buttons
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    const restartBtn = document.getElementById('restartBtn');
    
    // Previous button
    if (prevBtn) {
        prevBtn.style.display = currentStep === 1 ? 'none' : 'flex';
        prevBtn.disabled = currentStep === 1;
    }
    
    // Next button
    if (nextBtn) {
        nextBtn.style.display = currentStep === totalSteps ? 'none' : 'flex';
        nextBtn.disabled = currentStep === totalSteps || !validateStep(currentStep);
    }
    
    // Submit button
    if (submitBtn) {
        submitBtn.style.display = currentStep === (totalSteps - 1) ? 'flex' : 'none';
        submitBtn.disabled = !validateStep(currentStep);
    }
    
    // Restart button
    if (restartBtn) {
        restartBtn.style.display = currentStep === totalSteps ? 'flex' : 'none';
    }
}

// Update progress bar
function updateProgressBar() {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    if (progressFill) {
        const percentage = (currentStep / totalSteps) * 100;
        progressFill.style.width = `${percentage}%`;
    }
    
    if (progressText) {
        progressText.textContent = `Step ${currentStep} of ${totalSteps}`;
    }
}

// Change step function
function changeStep(direction) {
    const newStep = currentStep + direction;
    
    // Validate current step before moving forward
    if (direction > 0 && !validateStep(currentStep)) {
        showValidationErrors(currentStep);
        return;
    }
    
    // Check bounds
    if (newStep < 1 || newStep > totalSteps) {
        return;
    }
    
    // Save current step data
    if (direction > 0) {
        saveStepData(currentStep);
    }
    
    // Update current step
    currentStep = newStep;
    
    // Show new step
    showStep(currentStep);
    
    // Add smooth scrolling to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Validate step function
function validateStep(step) {
    const stepElement = document.getElementById(`step${step}`);
    if (!stepElement) return true;
    
    const requiredInputs = stepElement.querySelectorAll('input[required], select[required]');
    const radioGroups = {};
    
    let isValid = true;
    
    // Check required inputs
    requiredInputs.forEach(input => {
        if (input.type === 'radio') {
            // Handle radio groups
            const groupName = input.name;
            if (!radioGroups[groupName]) {
                radioGroups[groupName] = stepElement.querySelectorAll(`input[name="${groupName}"]`);
            }
        } else {
            // Handle other inputs
            if (!input.value.trim()) {
                isValid = false;
                input.classList.add('error');
            } else {
                input.classList.remove('error');
            }
        }
    });
    
    // Check radio groups
    Object.keys(radioGroups).forEach(groupName => {
        const radios = radioGroups[groupName];
        const isChecked = Array.from(radios).some(radio => radio.checked);
        
        if (!isChecked) {
            isValid = false;
            radios.forEach(radio => {
                radio.closest('.radio-option')?.classList.add('error');
            });
        } else {
            radios.forEach(radio => {
                radio.closest('.radio-option')?.classList.remove('error');
            });
        }
    });
    
    return isValid;
}

// Show validation errors
function showValidationErrors(step) {
    const stepElement = document.getElementById(`step${step}`);
    if (!stepElement) return;
    
    // Add shake animation to step
    stepElement.style.animation = 'shake 0.5s ease-in-out';
    setTimeout(() => {
        stepElement.style.animation = '';
    }, 500);
    
    // Focus on first invalid input
    const firstInvalid = stepElement.querySelector('.error input, .error select, input.error, select.error');
    if (firstInvalid) {
        firstInvalid.focus();
    }
}

// Save step data
function saveStepData(step) {
    const stepElement = document.getElementById(`step${step}`);
    if (!stepElement) return;
    
    const inputs = stepElement.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        if (input.type === 'radio') {
            if (input.checked) {
                formData[input.name] = input.value;
            }
        } else {
            formData[input.name] = input.value;
        }
    });
}

// Setup form validation styles
function setupFormValidation() {
    // Add CSS for validation errors
    const style = document.createElement('style');
    style.textContent = `
        .error {
            border-color: var(--danger) !important;
            box-shadow: 0 0 0 3px rgba(220, 53, 69, 0.1) !important;
        }
        
        .radio-option.error {
            border-color: var(--danger) !important;
            background-color: rgba(220, 53, 69, 0.05) !important;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
    `;
    document.head.appendChild(style);
}

// Submit form function
function submitForm() {
    // Validate current step
    if (!validateStep(currentStep)) {
        showValidationErrors(currentStep);
        return;
    }
    
    // Save current step data
    saveStepData(currentStep);
    
    // Show loading state
    const submitBtn = document.getElementById('submitBtn');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    submitBtn.disabled = true;
    
    // Simulate processing time
    setTimeout(() => {
        // Process the prediction
        const prediction = processAssessment(formData);
        
        // Move to results step
        currentStep = totalSteps;
        showResults(prediction);
        showStep(currentStep);
        
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }, 2000);
}

// Process assessment and generate prediction
function processAssessment(data) {
    // Simple rule-based assessment (you can replace this with actual ML model integration)
    let riskScore = 0;
    let riskFactors = [];
    
    // Academic pressure (weight: 0.15)
    const academicPressure = parseFloat(data.academicPressure) || 0;
    if (academicPressure >= 4) {
        riskScore += 0.15;
        riskFactors.push("High academic pressure");
    }
    
    // Study satisfaction (weight: 0.1, inverse relationship)
    const studySatisfaction = parseFloat(data.studySatisfaction) || 0;
    if (studySatisfaction <= 2) {
        riskScore += 0.1;
        riskFactors.push("Low study satisfaction");
    }
    
    // Work pressure (weight: 0.1)
    const workPressure = parseFloat(data.workPressure) || 0;
    if (workPressure >= 3) {
        riskScore += 0.1;
        riskFactors.push("High work pressure");
    }
    
    // Financial stress (weight: 0.15)
    const financialStress = parseInt(data.financialStress) || 0;
    if (financialStress >= 4) {
        riskScore += 0.15;
        riskFactors.push("High financial stress");
    }
    
    // Sleep duration (weight: 0.1)
    const sleepDuration = data.sleepDuration;
    if (sleepDuration === "Less than 5 hours") {
        riskScore += 0.1;
        riskFactors.push("Insufficient sleep");
    }
    
    // Dietary habits (weight: 0.05)
    const dietaryHabits = data.dietaryHabits;
    if (dietaryHabits === "Unhealthy") {
        riskScore += 0.05;
        riskFactors.push("Poor dietary habits");
    }
    
    // Suicidal thoughts (weight: 0.25)
    const suicidalThoughts = data.suicidalThoughts;
    if (suicidalThoughts === "Yes") {
        riskScore += 0.25;
        riskFactors.push("History of suicidal thoughts");
    }
    
    // Family history (weight: 0.1)
    const familyHistory = data.familyHistory;
    if (familyHistory === "Yes") {
        riskScore += 0.1;
        riskFactors.push("Family history of mental illness");
    }
    
    // Determine risk level
    let riskLevel, probability;
    if (riskScore >= 0.6) {
        riskLevel = "high";
        probability = Math.min(0.9, 0.6 + riskScore * 0.4);
    } else if (riskScore >= 0.3) {
        riskLevel = "moderate";
        probability = 0.3 + riskScore * 0.5;
    } else {
        riskLevel = "low";
        probability = riskScore * 0.5;
    }
    
    return {
        riskLevel,
        probability: Math.round(probability * 100),
        riskScore: Math.round(riskScore * 100),
        riskFactors,
        formData: data
    };
}

// Show results
function showResults(prediction) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    const resultHTML = `
        <div class="result-card ${prediction.riskLevel}-risk">
            <i class="fas ${getResultIcon(prediction.riskLevel)} result-icon"></i>
            <h3 class="result-title">${getResultTitle(prediction.riskLevel)}</h3>
            <p class="result-description">${getResultDescription(prediction.riskLevel, prediction.probability)}</p>
            
            ${prediction.riskFactors.length > 0 ? `
                <div class="result-recommendations">
                    <h4><i class="fas fa-exclamation-triangle"></i> Identified Risk Factors:</h4>
                    <ul>
                        ${prediction.riskFactors.map(factor => `<li><i class="fas fa-chevron-right"></i> ${factor}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            <div class="result-recommendations">
                <h4><i class="fas fa-lightbulb"></i> Recommendations:</h4>
                <ul>
                    ${getRecommendations(prediction.riskLevel).map(rec => `<li><i class="fas fa-check"></i> ${rec}</li>`).join('')}
                </ul>
            </div>
            
            ${prediction.riskLevel === 'high' ? `
                <div class="mental-health-note">
                    <i class="fas fa-phone"></i>
                    <div>
                        <p><strong>Immediate Support Available:</strong></p>
                        <p>If you're in crisis, please visit <strong><a href="https://blog.opencounseling.com/suicide-hotlines/">Find Help</a></strong> (Suicide & Crisis Lifeline) or reach out to a mental health professional immediately.</p>
                    </div>
                </div>
            ` : ''}
        </div>
    `;
    
    resultsContainer.innerHTML = resultHTML;
}

// Get result icon based on risk level
function getResultIcon(riskLevel) {
    switch (riskLevel) {
        case 'low': return 'fa-smile';
        case 'moderate': return 'fa-meh';
        case 'high': return 'fa-frown';
        default: return 'fa-question';
    }
}

// Get result title based on risk level
function getResultTitle(riskLevel) {
    switch (riskLevel) {
        case 'low': return 'Low Risk Detected';
        case 'moderate': return 'Moderate Risk Detected';
        case 'high': return 'High Risk Detected';
        default: return 'Assessment Complete';
    }
}

// Get result description
function getResultDescription(riskLevel, probability) {
    switch (riskLevel) {
        case 'low':
            return `Based on your responses, there's a ${probability}% indication of depression risk. You seem to be managing well overall, but it's always good to maintain healthy habits.`;
        case 'moderate':
            return `Based on your responses, there's a ${probability}% indication of depression risk. Some factors suggest you might benefit from additional support and stress management strategies.`;
        case 'high':
            return `Based on your responses, there's a ${probability}% indication of depression risk. We strongly recommend speaking with a mental health professional for proper evaluation and support.`;
        default:
            return 'Unable to determine risk level.';
    }
}

// Get recommendations based on risk level
function getRecommendations(riskLevel) {
    const baseRecommendations = [
        "Maintain a regular sleep schedule (7-9 hours per night)",
        "Engage in regular physical exercise",
        "Practice stress reduction techniques (meditation, deep breathing)",
        "Connect with supportive friends and family",
        "Consider talking to a counselor or therapist"
    ];
    
    const moderateRecommendations = [
        "Schedule an appointment with a mental health professional",
        "Join a support group or peer counseling program",
        "Develop a daily routine and stick to it",
        "Limit alcohol and avoid recreational drugs",
        "Practice mindfulness and relaxation techniques"
    ];
    
    const highRecommendations = [
        "Seek immediate professional help from a psychiatrist or psychologist",
        "Contact your doctor for a comprehensive mental health evaluation",
        "Inform a trusted friend or family member about how you're feeling",
        "Remove any means of self-harm from your environment",
        "Create a safety plan with professional guidance"
    ];
    
    switch (riskLevel) {
        case 'low': return baseRecommendations.slice(0, 3);
        case 'moderate': return [...baseRecommendations.slice(0, 2), ...moderateRecommendations.slice(0, 3)];
        case 'high': return highRecommendations;
        default: return baseRecommendations.slice(0, 3);
    }
}

// Restart assessment
function restartAssessment() {
    // Reset form data
    formData = {};
    currentStep = 1;
    
    // Reset form
    const form = document.getElementById('assessmentForm');
    form.reset();
    
    // Reset sliders to default values
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        slider.value = slider.defaultValue;
        updateSliderDisplay(slider);
    });
    
    // Remove validation errors
    const errorElements = document.querySelectorAll('.error');
    errorElements.forEach(element => {
        element.classList.remove('error');
    });
    
    // Show first step
    showStep(currentStep);
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Utility function to save assessment data (for future use)
function saveAssessmentData(data) {
    // This could be used to save data to localStorage or send to a server
    localStorage.setItem('lastAssessment', JSON.stringify({
        timestamp: new Date().toISOString(),
        data: data
    }));
}

// Load previous assessment (for future use)
function loadPreviousAssessment() {
    const saved = localStorage.getItem('lastAssessment');
    if (saved) {
        return JSON.parse(saved);
    }
    return null;
}
