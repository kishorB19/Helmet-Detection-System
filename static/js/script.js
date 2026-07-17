const testImages = [
    '4928.webp',
    'Banner-7.jpg',
    'dc-Cover-0oq65ge0qcpuo13i3mfvu34qm3-20160318035510.Medi.jpeg',
    'gettyimages-1284091102-612x612.jpg',
    'gettyimages-1464681993-612x612.jpg',
    'handsome-happy-young-man-bicycle-260nw-2270995623.webp',
    'how-to-become-a-good-bike-rider.avif',
    'images.jpeg',
    'indian-man-driving-bike-wear-helmet-highway-road-268126278.webp',
    'istockphoto-1375207592-612x612.jpg',
    'istockphoto-518659648-612x612.jpg',
    'man-driving-motorcycle-without-helmet-260nw-2195048139.webp',
    'phuket-thailand-03june-2024-traffic-260nw-2479379633.webp',
    'pros-cons-of-group-motorcycle-riding_d65422ca-99d7-4a67-a19b-639ecefa758f.webp'
];

const $ = id => document.getElementById(id);
const uploadArea = $('uploadArea');
const fileInput = $('fileInput');
const originalImage = $('originalImage');
const resultImage = $('resultImage');
const originalPlaceholder = $('originalPlaceholder');
const resultPlaceholder = $('resultPlaceholder');
const detectBtn = $('detectBtn');
const resetBtn = $('resetBtn');
const loading = $('loading');
const resultInfo = $('resultInfo');
const errorInfo = $('errorInfo');
const helmetStatusCard = $('helmetStatusCard');
const helmetStatusText = $('helmetStatusText');
const processingTime = $('processingTime');
const errorMessage = $('errorMessage');
const imageContainer = $('imageContainer');
const testImagesGrid = $('testImagesGrid');

let currentImage = null;

// Initialize test images gallery
testImages.forEach(img => {
    const div = document.createElement('div');
    div.className = 'test-image-card';
    div.innerHTML = `
        <img src="/test_images/${img}" alt="Test Image" loading="lazy">
        <div class="overlay"><span>Detect</span></div>
    `;
    div.onclick = () => loadTestImage(img);
    testImagesGrid.appendChild(div);
});

async function loadTestImage(imgName) {
    try {
        resetApplication();
        const response = await fetch(`/test_images/${imgName}`);
        const blob = await response.blob();
        const reader = new FileReader();
        reader.onload = e => {
            setBase64Image(e.target.result);
            handleDetection();
        };
        reader.readAsDataURL(blob);
    } catch (err) {
        showError("Failed to load test image.");
    }
}

uploadArea.onclick = () => fileInput.click();
fileInput.onchange = handleFileSelect;
detectBtn.onclick = handleDetection;
resetBtn.onclick = resetApplication;

uploadArea.ondragover = e => { e.preventDefault(); uploadArea.style.borderColor = '#38bdf8'; };
uploadArea.ondragleave = () => uploadArea.style.borderColor = '';
uploadArea.ondrop = e => {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    if (e.dataTransfer.files[0]) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
};

function handleFileSelect() {
    if (!fileInput.files[0]) return;
    const file = fileInput.files[0];
    if (!file.type.match('image.*')) return showError('Please select a valid image file');
    
    const reader = new FileReader();
    reader.onload = e => {
        setBase64Image(e.target.result);
        // Auto run detection on upload as well for better UX
        handleDetection();
    };
    reader.readAsDataURL(file);
}

function setBase64Image(base64Str) {
    currentImage = base64Str;
    imageContainer.style.display = 'grid';
    originalImage.src = currentImage;
    originalImage.style.display = 'block';
    originalPlaceholder.style.display = 'none';
    resultImage.style.display = 'none';
    resultPlaceholder.style.display = 'flex';
    resultInfo.style.display = 'none';
    errorInfo.style.display = 'none';
    detectBtn.disabled = false;
    resetBtn.style.display = 'inline-block';
    
    // Scroll to preview
    imageContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function handleDetection() {
    if (!currentImage) return;
    
    detectBtn.disabled = true;
    loading.style.display = 'block';
    errorInfo.style.display = 'none';
    resultInfo.style.display = 'none';
    resultImage.style.display = 'none';
    resultPlaceholder.style.display = 'flex';
    
    const startTime = performance.now();

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImage })
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Detection failed on server');
        }

        resultImage.src = data.result_image;
        resultImage.style.display = 'block';
        resultPlaceholder.style.display = 'none';

        const total = (typeof data.total_detections === 'number') ? data.total_detections : (data.wearing_helmet + data.not_wearing_helmet);
        
        helmetStatusCard.className = 'helmet-stat-card';
        if (data.not_wearing_helmet > 0) {
            helmetStatusText.textContent = 'No';
            helmetStatusCard.classList.add('not-wearing');
        } else if (data.wearing_helmet > 0) {
            helmetStatusText.textContent = 'Yes';
            helmetStatusCard.classList.add('wearing');
        } else if (total > 0) {
            helmetStatusText.textContent = 'Unclear';
            helmetStatusCard.classList.add('not-wearing');
        } else {
            helmetStatusText.textContent = 'No person detected';
            helmetStatusCard.classList.add('not-wearing');
        }

        processingTime.textContent = Math.round(performance.now() - startTime);
        resultInfo.style.display = 'block';
        
    } catch (error) {
        showError(error.message || 'An error occurred during detection');
    } finally {
        loading.style.display = 'none';
        detectBtn.disabled = false;
    }
}

function resetApplication() {
    fileInput.value = '';
    currentImage = null;
    imageContainer.style.display = 'none';
    originalImage.src = ''; 
    originalImage.style.display = 'none';
    originalPlaceholder.style.display = 'flex';
    resultImage.src = ''; 
    resultImage.style.display = 'none';
    resultPlaceholder.style.display = 'flex';
    resultInfo.style.display = 'none';
    errorInfo.style.display = 'none';
    loading.style.display = 'none';
    detectBtn.disabled = true;
    resetBtn.style.display = 'none';
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorInfo.style.display = 'block';
    resultInfo.style.display = 'none';
}
