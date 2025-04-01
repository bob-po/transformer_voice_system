// 录音相关变量
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let mediaStream;

// 音频播放相关变量
const audioPlayer = document.getElementById('audioPlayer');
const stopAudioBtn = document.getElementById('stopAudio');

// 语音识别相关元素
const startRecordingBtn = document.getElementById('startRecording');
const stopRecordingBtn = document.getElementById('stopRecording');
const audioFileInput = document.getElementById('audioFile');
const processAudioFileBtn = document.getElementById('processAudioFile');
const recognitionLangSelect = document.getElementById('recognitionLang');
const recognitionResult = document.getElementById('recognitionResult');

// 文本翻译相关元素
const srcLangSelect = document.getElementById('srcLang');
const tgtLangSelect = document.getElementById('tgtLang');
const srcText = document.getElementById('srcText');
const tgtText = document.getElementById('tgtText');
const translateBtn = document.getElementById('translateBtn');

// 语音合成相关元素
const synthesisLang = document.getElementById('synthesisLang');
const voiceStyle = document.getElementById('voiceStyle');
const speedControl = document.getElementById('speedControl');
const speedValue = document.getElementById('speedValue');
const synthesisText = document.getElementById('synthesisText');
const synthesisBtn = document.getElementById('synthesisBtn');
const referenceAudio = document.getElementById('referenceAudio');
const referenceFileName = document.getElementById('referenceFileName');

// 声纹识别相关元素
const speakerNameInput = document.getElementById('speakerName');
const registerSpeakerBtn = document.getElementById('registerSpeakerBtn');
const registerAudioInput = document.getElementById('registerAudioInput');
const registerAudioContainer = document.getElementById('registerAudioContainer');
const registerAudioName = document.getElementById('registerAudioName');
const identifyAudioInput = document.getElementById('identifyAudioInput');
const identifyAudioContainer = document.getElementById('identifyAudioContainer');
const identifyAudioName = document.getElementById('identifyAudioName');
const identifySpeakerBtn = document.getElementById('identifySpeakerBtn');
const identificationResult = document.getElementById('identificationResult');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceScore = document.getElementById('confidenceScore');

// 录音功能
async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(mediaStream, {
            mimeType: 'audio/webm'
        });
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            try {
                // 显示处理中的状态
                recognitionResult.textContent = "正在处理录音...";
                
                // 将录音数据转换为WAV格式
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                const arrayBuffer = await audioBlob.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                
                // 创建WAV文件
                const wavBuffer = audioBufferToWav(audioBuffer);
                const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
                
                // 创建一个临时的文件对象并设置文件名
                const wavFile = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
                
                // 创建FormData对象并添加文件
                const formData = new FormData();
                formData.append('audio', wavFile);
                formData.append('language', recognitionLangSelect.value);
                
                // 发送到服务器
                const response = await fetch('/recognize_speech', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // 显示成功消息
                showRecognitionSuccess();
                
                recognitionResult.textContent = data.text;
                srcText.value = data.text;  // 自动填充到翻译输入框
                
                // 清理资源
                audioContext.close();
            } catch (error) {
                console.error('Error processing audio:', error);
                recognitionResult.textContent = '处理音频时出错：' + error.message;
                alert('处理音频时出错：' + error.message);
            }
        };

        mediaRecorder.start();
        isRecording = true;
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('无法访问麦克风，请确保已授予权限。');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
        
        // 停止所有音轨
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
        }
    }
}

// 将AudioBuffer转换为WAV格式
function audioBufferToWav(buffer) {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2;
    const buffer32 = new Float32Array(buffer.length * numOfChan);
    const view = new DataView(new ArrayBuffer(44 + length));
    let offset = 0;
    const writeString = (str) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    };

    // WAV文件头
    writeString('RIFF');
    offset += 4;
    view.setUint32(offset, 36 + length, true);
    offset += 4;
    writeString('WAVE');
    offset += 4;
    writeString('fmt ');
    offset += 4;
    view.setUint32(offset, 16, true);
    offset += 4;
    view.setUint16(offset, 1, true);
    offset += 2;
    view.setUint16(offset, numOfChan, true);
    offset += 2;
    view.setUint32(offset, buffer.sampleRate, true);
    offset += 4;
    view.setUint32(offset, buffer.sampleRate * 2 * numOfChan, true);
    offset += 4;
    view.setUint16(offset, numOfChan * 2, true);
    offset += 2;
    view.setUint16(offset, 16, true);
    offset += 2;
    writeString('data');
    offset += 4;
    view.setUint32(offset, length, true);
    offset += 4;

    // 写入音频数据
    const channelData = [];
    for (let i = 0; i < numOfChan; i++) {
        channelData[i] = buffer.getChannelData(i);
    }

    let index = 0;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numOfChan; channel++) {
            const sample = channelData[channel][i];
            buffer32[index++] = sample;
        }
    }

    // 转换为16位PCM
    for (let i = 0; i < buffer32.length; i++) {
        const s = Math.max(-1, Math.min(1, buffer32[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return view.buffer;
}

// 显示上传成功效果
function showUploadSuccess(container, fileName, fileNameElement) {
    const successElement = container.querySelector('.upload-success');
    
    // 显示文件名
    fileNameElement.textContent = fileName;
    
    // 显示成功提示
    successElement.classList.remove('hidden');
    
    // 3秒后隐藏成功提示
    setTimeout(() => {
        successElement.classList.add('hidden');
    }, 3000);
}

// 更新文件上传处理
audioFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        try {
            // 显示处理中的状态
            recognitionResult.textContent = "正在处理音频...";
            
            const formData = new FormData();
            formData.append('audio', file);
            formData.append('language', recognitionLangSelect.value);

            // 显示上传成功效果
            showUploadSuccess(document.querySelector('label[for="audioFile"]'), file.name, document.getElementById('selectedFileName'));
            
            const response = await fetch('/recognize_speech', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            // 显示识别成功提示
            showRecognitionSuccess();
            
            recognitionResult.textContent = data.text;
            srcText.value = data.text;  // 自动填充到翻译输入框
        } catch (error) {
            console.error('Error processing audio file:', error);
            recognitionResult.textContent = '处理音频时出错：' + error.message;
            alert('处理音频文件时出错：' + error.message);
        }
    }
});

// 显示通用成功提示框
function showSuccessNotification(message) {
    const successElement = document.createElement('div');
    successElement.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-fade-out';
    successElement.textContent = message;
    document.body.appendChild(successElement);
    
    // 3秒后移除提示
    setTimeout(() => {
        successElement.remove();
    }, 3000);
}

// 修改文本翻译函数
async function translateText() {
    const text = srcText.value.trim();
    if (!text) {
        alert('请输入要翻译的文本');
        return;
    }

    try {
        // 显示翻译中状态
        tgtText.textContent = "正在翻译...";
        
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                source_language: srcLangSelect.value,
                target_language: tgtLangSelect.value
            })
        });

        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        tgtText.textContent = data.translated_text;
        synthesisText.value = data.translated_text;  // 自动填充到语音合成输入框
        
        // 显示翻译成功提示
        showSuccessNotification('文本翻译成功！');
        
    } catch (error) {
        console.error('Error translating text:', error);
        tgtText.textContent = '翻译出错：' + error.message;
        alert('翻译文本时出错：' + error.message);
    }
}

// 显示音色上传成功效果
function showVoiceUploadSuccess(fileName) {
    const successElement = document.querySelector('.voice-upload-success');
    
    // 显示文件名
    referenceFileName.textContent = fileName;
    
    // 显示成功提示
    successElement.classList.remove('hidden');
    
    // 3秒后隐藏成功提示
    setTimeout(() => {
        successElement.classList.add('hidden');
    }, 3000);
}

// 处理参考音色文件上传
referenceAudio.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        showVoiceUploadSuccess(file.name);
    }
});

// 更新语速显示
speedControl.addEventListener('input', (event) => {
    speedValue.textContent = event.target.value;
});

// 修改语音合成函数
async function synthesizeVoice() {
    const text = synthesisText.value.trim();
    if (!text) {
        alert('请输入要合成的文本');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('language', synthesisLang.value);
        formData.append('style', voiceStyle.value);
        formData.append('speed', speedControl.value);

        // 如果有上传参考音色，添加到请求中
        if (referenceAudio.files[0]) {
            formData.append('reference_audio', referenceAudio.files[0]);
        }

        synthesisBtn.disabled = true;
        synthesisBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>生成中...';

        const response = await fetch('/synthesize_voice', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || '语音合成失败');
        }

        // 将base64音频数据转换为Blob
        const audioData = atob(data.audio_data);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const uint8Array = new Uint8Array(arrayBuffer);
        for (let i = 0; i < audioData.length; i++) {
            uint8Array[i] = audioData.charCodeAt(i);
        }
        const audioBlob = new Blob([arrayBuffer], { type: `audio/${data.format}` });
        
        // 创建音频URL并播放
        const audioUrl = URL.createObjectURL(audioBlob);
        audioPlayer.src = audioUrl;
        audioPlayer.play();
        stopAudioBtn.disabled = false;

        // 显示合成成功提示
        showSuccessNotification('语音合成成功！');

    } catch (error) {
        console.error('Error synthesizing voice:', error);
        alert('语音合成时出错：' + error.message);
    } finally {
        synthesisBtn.disabled = false;
        synthesisBtn.innerHTML = '<i class="fas fa-volume-up mr-2"></i>合成语音';
    }
}

// 停止音频播放
function stopAudio() {
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    stopAudioBtn.disabled = true;
}

// 事件监听器
startRecordingBtn.addEventListener('click', startRecording);
stopRecordingBtn.addEventListener('click', stopRecording);
translateBtn.addEventListener('click', translateText);
synthesisBtn.addEventListener('click', synthesizeVoice);
stopAudioBtn.addEventListener('click', stopAudio);

// 音频播放结束时禁用停止按钮
audioPlayer.addEventListener('ended', () => {
    stopAudioBtn.disabled = true;
});

// 拖拽上传
const dropZone = document.querySelector('label[for="audioFile"]');

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-blue-500');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-blue-500');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500');
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'audio/wav') {
        audioFileInput.files = e.dataTransfer.files;
        const event = new Event('change');
        audioFileInput.dispatchEvent(event);
    } else {
        alert('请上传WAV格式的音频文件');
    }
});

// 确保所有DOM元素都已加载
document.addEventListener('DOMContentLoaded', () => {
    // 重新绑定事件监听器
    document.getElementById('translateBtn').addEventListener('click', translateText);
    document.getElementById('synthesisBtn').addEventListener('click', synthesizeVoice);
    document.getElementById('registerSpeakerBtn').addEventListener('click', registerSpeaker);
    document.getElementById('identifySpeakerBtn').addEventListener('click', identifySpeaker);
});

// 修改显示识别成功效果的函数，使用通用提示框
function showRecognitionSuccess() {
    showSuccessNotification('语音识别成功！');
}

// 添加必要的CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    .animate-fade-out {
        animation: fadeOut 0.5s ease-out 2.5s forwards;
    }
`;
document.head.appendChild(style);

// 为上传区域添加点击事件
registerAudioContainer.addEventListener('click', () => {
    registerAudioInput.click();
});

identifyAudioContainer.addEventListener('click', () => {
    identifyAudioInput.click();
});

// 处理注册音频文件上传
registerAudioInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        showUploadSuccess(registerAudioContainer, file.name, registerAudioName);
    }
});

// 处理识别音频文件上传
identifyAudioInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        showUploadSuccess(identifyAudioContainer, file.name, identifyAudioName);
    }
});

// 添加拖拽上传功能
function setupDragAndDrop(container, input) {
    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        container.classList.add('border-blue-500');
    });

    container.addEventListener('dragleave', () => {
        container.classList.remove('border-blue-500');
    });

    container.addEventListener('drop', (e) => {
        e.preventDefault();
        container.classList.remove('border-blue-500');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('audio/')) {
            input.files = e.dataTransfer.files;
            const event = new Event('change');
            input.dispatchEvent(event);
        } else {
            alert('请上传音频文件');
        }
    });
}

// 设置拖拽上传
setupDragAndDrop(registerAudioContainer, registerAudioInput);
setupDragAndDrop(identifyAudioContainer, identifyAudioInput);

// 验证声纹
async function verifySpeaker() {
    if (!referenceAudioInput.files[0] || !testAudioInput.files[0]) {
        alert('请上传参考音频和待验证音频');
        return;
    }

    try {
        verifySpeakerBtn.disabled = true;
        verifySpeakerBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>验证中...';
        verificationResult.textContent = '正在验证...';

        const formData = new FormData();
        formData.append('reference_audio', referenceAudioInput.files[0]);
        formData.append('test_audio', testAudioInput.files[0]);

        const response = await fetch('/verify_speaker', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || '声纹验证失败');
        }

        // 更新相似度条
        const similarityPercentage = (data.similarity_score * 100).toFixed(1);
        similarityBar.style.width = `${similarityPercentage}%`;
        similarityScore.textContent = `相似度: ${similarityPercentage}%`;

        // 更新验证结果
        if (data.is_same_speaker) {
            verificationResult.innerHTML = '<span class="text-green-600"><i class="fas fa-check-circle mr-2"></i>是同一个人</span>';
        } else {
            verificationResult.innerHTML = '<span class="text-red-600"><i class="fas fa-times-circle mr-2"></i>不是同一个人</span>';
        }

        // 显示验证成功提示
        showSuccessNotification('声纹验证完成！');

    } catch (error) {
        console.error('Error verifying speaker:', error);
        verificationResult.innerHTML = `<span class="text-red-600"><i class="fas fa-exclamation-circle mr-2"></i>验证失败：${error.message}</span>`;
    } finally {
        verifySpeakerBtn.disabled = false;
        verifySpeakerBtn.innerHTML = '<i class="fas fa-fingerprint mr-2"></i>验证声纹';
    }
}

// 添加声纹验证按钮事件监听器
verifySpeakerBtn.addEventListener('click', verifySpeaker);

// 注册声纹
async function registerSpeaker() {
    const speakerName = speakerNameInput.value.trim();
    if (!speakerName) {
        alert('请输入说话人姓名');
        return;
    }

    if (!registerAudioInput.files[0]) {
        alert('请上传注册音频');
        return;
    }

    try {
        registerSpeakerBtn.disabled = true;
        registerSpeakerBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>注册中...';

        const formData = new FormData();
        formData.append('speaker_name', speakerName);
        formData.append('audio', registerAudioInput.files[0]);

        const response = await fetch('/register_speaker', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || '声纹注册失败');
        }

        // 显示注册成功提示
        showSuccessNotification('声纹注册成功！');
        
        // 清空输入
        speakerNameInput.value = '';
        registerAudioInput.value = '';
        registerAudioName.textContent = '';

    } catch (error) {
        console.error('Error registering speaker:', error);
        alert('声纹注册失败：' + error.message);
    } finally {
        registerSpeakerBtn.disabled = false;
        registerSpeakerBtn.innerHTML = '<i class="fas fa-user-plus mr-2"></i>注册声纹';
    }
}

// 识别声纹
async function identifySpeaker() {
    if (!identifyAudioInput.files[0]) {
        alert('请上传待识别音频');
        return;
    }

    try {
        identifySpeakerBtn.disabled = true;
        identifySpeakerBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>识别中...';
        identificationResult.textContent = '正在识别...';

        const formData = new FormData();
        formData.append('audio', identifyAudioInput.files[0]);

        const response = await fetch('/identify_speaker', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || '声纹识别失败');
        }

        // 更新置信度条
        const confidencePercentage = (data.confidence * 100).toFixed(1);
        confidenceBar.style.width = `${confidencePercentage}%`;
        confidenceScore.textContent = `置信度: ${confidencePercentage}%`;

        // 更新识别结果
        if (data.speaker_name) {
            identificationResult.innerHTML = `<span class="text-green-600"><i class="fas fa-check-circle mr-2"></i>识别为: ${data.speaker_name}</span>`;
        } else {
            identificationResult.innerHTML = '<span class="text-red-600"><i class="fas fa-times-circle mr-2"></i>未识别出说话人</span>';
        }

        // 显示识别成功提示
        showSuccessNotification('声纹识别完成！');

    } catch (error) {
        console.error('Error identifying speaker:', error);
        identificationResult.innerHTML = `<span class="text-red-600"><i class="fas fa-exclamation-circle mr-2"></i>识别失败：${error.message}</span>`;
    } finally {
        identifySpeakerBtn.disabled = false;
        identifySpeakerBtn.innerHTML = '<i class="fas fa-fingerprint mr-2"></i>识别声纹';
    }
} 