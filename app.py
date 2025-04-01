from flask import Flask, request, jsonify, render_template, send_file
import torch
from transformers import MarianMTModel, MarianTokenizer
import os
import tempfile
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave
import json
import soundfile as sf
import numpy as np
import sys
import librosa
import time
import base64
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# 将 OpenVoice 添加到 Python 路径
# openvoice_dir = os.path.join(os.path.dirname(__file__), 'models/openvoice')
# sys.path.append(openvoice_dir)

from openvoice.api import BaseSpeakerTTS
from openvoice.api import ToneColorConverter
from openvoice import se_extractor
import warnings
warnings.simplefilter("ignore")
app = Flask(__name__)

# 配置日志级别
SetLogLevel(-1)

# 初始化支持的语言
LANGUAGES = {
    'en': '英语',
    'fr': '法语',
    'zh': '中文'
}

# 模型路径配置
VOSK_MODEL_PATHS = {
    'en': 'models/vosk-model/vosk-model-small-en-us',
    'fr': 'models/vosk-model/vosk-model-small-fr',
    'zh': 'models/vosk-model/vosk-model-small-cn'
}
TRANSLATION_MODEL_PATH = 'models/translation'
SPEAKER_RECOGNITION_MODEL_PATH = 'models/speaker_recognition'

# OpenVoice模型路径配置
OPENVOICE_BASE_DIR = os.path.normpath('models/openvoice/checkpoints')
OPENVOICE_v2_DIR = os.path.normpath('models/openvoice/checkpoints_v2')
OPENVOICE_MODELS = {
    'en': {
        'config': os.path.normpath(os.path.join(OPENVOICE_BASE_DIR, 'base_speakers', 'EN', 'config.json')),
        'checkpoint': os.path.normpath(os.path.join(OPENVOICE_BASE_DIR, 'base_speakers', 'EN', 'checkpoint.pth')),
        'default_se': os.path.normpath(os.path.join(OPENVOICE_BASE_DIR, 'base_speakers', 'EN', 'en_default_se.pth')),
        'style_se': os.path.normpath(os.path.join(OPENVOICE_BASE_DIR, 'base_speakers', 'EN', 'en_style_se.pth'))
    },
    'zh': {
        'config': os.path.normpath(os.path.join(OPENVOICE_BASE_DIR, 'base_speakers', 'ZH', 'config.json')),
        'checkpoint': os.path.normpath(os.path.join(OPENVOICE_BASE_DIR, 'base_speakers', 'ZH', 'checkpoint.pth')),
        'default_se': os.path.normpath(os.path.join(OPENVOICE_v2_DIR, 'base_speakers', 'ses', 'zh.pth')),
        'style_se': os.path.normpath(os.path.join(OPENVOICE_v2_DIR, 'base_speakers', 'ses', 'zh.pth'))
    }
}

# 音色转换模型路径
CONVERTER_CONFIG = os.path.normpath(os.path.join(OPENVOICE_v2_DIR, 'converter', 'config.json'))
CONVERTER_CHECKPOINT = os.path.normpath(os.path.join(OPENVOICE_v2_DIR, 'converter', 'checkpoint.pth'))

print(f"音色转换模型配置文件路径: {CONVERTER_CONFIG}")
print(f"音色转换模型检查点路径: {CONVERTER_CHECKPOINT}")

# 确保converter目录存在
converter_dir = os.path.normpath(os.path.join(OPENVOICE_v2_DIR, 'converter'))
if not os.path.exists(converter_dir):
    os.makedirs(converter_dir)
    print(f"创建音色转换模型目录: {converter_dir}")

# 检查音色转换模型文件是否存在
if not os.path.exists(CONVERTER_CONFIG):
    print(f"警告: 音色转换模型配置文件不存在: {CONVERTER_CONFIG}")
if not os.path.exists(CONVERTER_CHECKPOINT):
    print(f"警告: 音色转换模型检查点文件不存在: {CONVERTER_CHECKPOINT}")

# 声纹识别模型路径
SPEAKER_RECOGNITION_DIR = os.path.normpath('models/speaker_recognition')
SPEAKER_RECOGNITION_HYPERPARAMS = os.path.normpath(os.path.join(SPEAKER_RECOGNITION_DIR, 'hyperparams.yaml'))

# 初始化翻译模型
translation_models = {}
translation_tokenizers = {}

def load_translation_model(source_lang, target_lang):
    if source_lang == target_lang:
        return None, None
    
    model_path = f'{TRANSLATION_MODEL_PATH}/{source_lang}-{target_lang}'
    model_key = f'{source_lang}-{target_lang}'
    
    if model_key not in translation_models:
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_path)
            model = MarianMTModel.from_pretrained(model_path)
            translation_models[model_key] = model
            translation_tokenizers[model_key] = tokenizer
        except Exception as e:
            print(f"加载模型 {model_path} 时出错: {str(e)}")
            return None, None
    
    return translation_models[model_key], translation_tokenizers[model_key]

# 初始化语音识别模型
vosk_models = {}
for lang, path in VOSK_MODEL_PATHS.items():
    try:
        vosk_models[lang] = Model(path)
        print(f"已加载 {LANGUAGES[lang]} 的语音识别模型")
    except Exception as e:
        print(f"加载 {LANGUAGES[lang]} 的语音识别模型时出错: {str(e)}")

# Initialize OpenVoice model
try:
    # 初始化语言模型字典
    base_speaker_tts_models = {}
    
    # 为每种语言加载模型
    for lang in ['en', 'zh']:
        try:
            model = BaseSpeakerTTS(
                config_path=OPENVOICE_MODELS[lang]['config'],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            model.load_ckpt(OPENVOICE_MODELS[lang]['checkpoint'])
            base_speaker_tts_models[lang] = model
            print(f"已加载 {LANGUAGES[lang]} 的语音合成模型")
        except Exception as e:
            print(f"加载 {LANGUAGES[lang]} 的语音合成模型时出错: {str(e)}")
    
    # 音色转换模型
    tone_color_converter = ToneColorConverter(CONVERTER_CONFIG, device="cuda" if torch.cuda.is_available() else "cpu") # 此处需要自动下载
    tone_color_converter.load_ckpt(CONVERTER_CHECKPOINT)
    
    print("OpenVoice模型加载成功")
except Exception as e:
    print(f"警告: 无法初始化OpenVoice模型: {str(e)}")
    base_speaker_tts_models = {}
    tone_color_converter = None

# 初始化声纹识别模型
try:
    print(SPEAKER_RECOGNITION_HYPERPARAMS)
    speaker_recognition_model = EncoderClassifier.from_hparams( # 此处我看过源码，是直接从本地加载的，但是还是需要远程加载一下
        source=SPEAKER_RECOGNITION_DIR,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    print("声纹识别模型加载成功")
except Exception as e:
    print(f"警告: 无法加载声纹识别模型: {str(e)}")
    speaker_recognition_model = None

# 声纹特征存储
SPEAKER_EMBEDDINGS = {}

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get('text')
        source_lang = data.get('source_language', 'auto')
        target_lang = data.get('target_language')

        if not text or not target_lang:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Load appropriate translation model
        model, tokenizer = load_translation_model(source_lang, target_lang)
        if not model or not tokenizer:
            return jsonify({'error': 'Translation model not available'}), 400

        # Perform translation
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return jsonify({
            'translated_text': translated_text,
            'source_language': source_lang,
            'target_language': target_lang
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recognize_speech', methods=['POST'])
def recognize_speech():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '未提供音频文件'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'error': '未选择文件'}), 400
            
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({'error': '仅支持WAV格式的音频文件'}), 400

        language = request.form.get('language', 'en')  # 默认使用英语

        # 检查是否有对应语言的模型
        if language not in vosk_models:
            return jsonify({'error': f'未找到 {LANGUAGES[language]} 的语音识别模型'}), 400

        # 创建临时目录（如果不存在）
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # 使用唯一文件名保存上传的文件
        temp_filename = os.path.join(temp_dir, f'upload_{int(time.time())}_{os.urandom(4).hex()}.wav')
        converted_filename = os.path.join(temp_dir, f'converted_{int(time.time())}_{os.urandom(4).hex()}.wav')
        
        wf = None
        try:
            audio_file.save(temp_filename)

            # 使用 librosa 加载并转换音频格式
            try:
                # 加载音频文件
                audio_data, sample_rate = librosa.load(temp_filename, sr=None, mono=True)
                
                # 转换为16位PCM格式
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # 保存为正确格式的WAV文件
                with wave.open(converted_filename, 'wb') as wf:
                    wf.setnchannels(1)  # 单声道
                    wf.setsampwidth(2)  # 16位
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data.tobytes())
                
            except Exception as e:
                return jsonify({'error': f'音频格式转换失败: {str(e)}'}), 400

            # 使用转换后的文件进行语音识别
            wf = wave.open(converted_filename, "rb")
            rec = KaldiRecognizer(vosk_models[language], wf.getframerate())
            rec.SetWords(True)

            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text += result.get('text', '') + " "

            # 获取最终结果
            final_result = json.loads(rec.FinalResult())
            text += final_result.get('text', '')

            return jsonify({
                'success': True,
                'text': text.strip(),
                'language': language,
                'filename': audio_file.filename
            })

        finally:
            # 确保关闭文件句柄
            if wf:
                wf.close()
                
            # 清理临时文件
            try:
                # 等待一小段时间确保文件句柄被完全释放
                time.sleep(0.1)
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                if os.path.exists(converted_filename):
                    os.remove(converted_filename)
            except Exception as e:
                print(f"清理临时文件时出错: {str(e)}")

    except Exception as e:
        return jsonify({
            'error': str(e),
            'details': '处理音频文件时发生错误'
        }), 500

@app.route('/synthesize_voice', methods=['POST'])
def synthesize_voice():
    try:
        if not base_speaker_tts_models or not tone_color_converter:
            return jsonify({'error': 'OpenVoice模型未初始化'}), 500

        if 'text' not in request.form:
            return jsonify({'error': '未提供要合成的文本'}), 400

        text = request.form['text']
        language = request.form.get('language', 'en')  # 默认使用英语
        
        # 检查是否支持该语言
        if language not in base_speaker_tts_models:
            return jsonify({'error': f'不支持 {LANGUAGES[language]} 的语音合成'}), 400
            
        style = request.form.get('style', 'default')  # 语音风格
        speed = float(request.form.get('speed', 1.0))  # 语速
        
        # 创建临时目录（如果不存在）
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成临时文件名
        temp_src = os.path.join(temp_dir, f'src_{int(time.time())}_{os.urandom(4).hex()}.wav')
        temp_output = os.path.join(temp_dir, f'output_{int(time.time())}_{os.urandom(4).hex()}.wav')
        
        try:
            # 将语言代码转换为OpenVoice支持的格式
            language_map = {
                'en': 'English',
                'zh': 'Chinese'
            }
            openvoice_lang = language_map.get(language.lower(), 'English')
            
            # 使用对应语言的模型生成音频
            current_model = base_speaker_tts_models[language]
            current_model.tts(
                text=text,
                output_path=temp_src,
                speaker=style,
                language=openvoice_lang,
                speed=speed
            )
            
            # 处理音色转换
            if 'reference_audio' in request.files:
                reference_file = request.files['reference_audio']
                reference_path = os.path.join(temp_dir, f'ref_{int(time.time())}_{os.urandom(4).hex()}.wav')
                try:
                    reference_file.save(reference_path)
                    # 提取目标音色特征
                    target_se, _ = se_extractor.get_se(reference_path, tone_color_converter, target_dir=temp_dir, vad=True)
                finally:
                    if os.path.exists(reference_path):
                        os.remove(reference_path)
            else:
                # 使用默认音色
                se_path = OPENVOICE_MODELS[language]['style_se'] if style != 'default' else OPENVOICE_MODELS[language]['default_se']
                target_se = torch.load(se_path).to(current_model.device)
            
            # 获取源音色特征
            source_se = torch.load(
                OPENVOICE_MODELS[language]['style_se'] if style != 'default' else OPENVOICE_MODELS[language]['default_se']
            ).to(current_model.device)
            
            # 进行音色转换
            tone_color_converter.convert(
                audio_src_path=temp_src,
                src_se=source_se,
                tgt_se=target_se,
                output_path=temp_output,
                message="Generated by OpenVoice"
            )
            
            # 读取音频文件并转换为base64
            with open(temp_output, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return jsonify({
                'success': True,
                'audio_data': audio_base64,
                'format': 'wav'
            })

        finally:
            # 清理临时文件
            for temp_file in [temp_src, temp_output]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"清理临时文件时出错: {str(e)}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify_speaker', methods=['POST'])
def verify_speaker():
    try:
        if not speaker_recognition_model:
            return jsonify({'error': '声纹识别模型未初始化'}), 500

        if 'reference_audio' not in request.files or 'test_audio' not in request.files:
            return jsonify({'error': '未提供音频文件'}), 400

        reference_audio = request.files['reference_audio']
        test_audio = request.files['test_audio']

        # 创建临时目录
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # 保存音频文件
        ref_path = os.path.join(temp_dir, f'ref_{int(time.time())}_{os.urandom(4).hex()}.wav')
        test_path = os.path.join(temp_dir, f'test_{int(time.time())}_{os.urandom(4).hex()}.wav')

        try:
            reference_audio.save(ref_path)
            test_audio.save(test_path)

            # 提取声纹特征
            ref_embedding = speaker_recognition_model.encode_batch(ref_path)
            test_embedding = speaker_recognition_model.encode_batch(test_path)

            # 计算相似度
            similarity = torch.nn.functional.cosine_similarity(ref_embedding, test_embedding)
            similarity_score = similarity.item()

            # 设置阈值（可以根据需要调整）
            threshold = 0.7
            is_same_speaker = similarity_score >= threshold

            return jsonify({
                'success': True,
                'similarity_score': float(similarity_score),
                'is_same_speaker': bool(is_same_speaker),
                'threshold': float(threshold)
            })

        finally:
            # 清理临时文件
            for temp_file in [ref_path, test_path]:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"清理临时文件时出错: {str(e)}")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/register_speaker', methods=['POST'])
def register_speaker():
    try:
        if not speaker_recognition_model:
            return jsonify({'error': '声纹识别模型未初始化'}), 500

        if 'speaker_name' not in request.form or 'audio' not in request.files:
            return jsonify({'error': '缺少必要参数'}), 400

        speaker_name = request.form['speaker_name']
        audio_file = request.files['audio']

        # 创建临时目录
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # 保存音频文件
        temp_path = os.path.join(temp_dir, f'register_{int(time.time())}_{os.urandom(4).hex()}.wav')
        try:
            # 保存上传的音频文件
            audio_file.save(temp_path)
            print(f"音频文件已保存到: {temp_path}")

            # 提取声纹特征
            try:
                print("开始提取声纹特征...")
                # 加载音频文件并转换为tensor
                waveform, sample_rate = torchaudio.load(temp_path)
                # 确保音频是单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # 确保采样率为16kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                print(f"音频数据形状: {waveform.shape}, 采样率: {sample_rate}")
                
                # 提取声纹特征
                embedding = speaker_recognition_model.encode_batch(waveform)
                print(f"声纹特征提取成功: {type(embedding)}, shape={embedding.shape if hasattr(embedding, 'shape') else 'no shape'}")
                
                # 将tensor转换为numpy数组并存储
                embedding_np = embedding.squeeze().detach().cpu().numpy()
                print(f"转换后的特征形状: {embedding_np.shape}")
                SPEAKER_EMBEDDINGS[speaker_name] = embedding_np
                print(f"声纹特征已存储: {speaker_name}")

                return jsonify({
                    'success': True,
                    'message': f'成功注册说话人: {speaker_name}'
                })
                
            except Exception as e:
                print(f"声纹特征提取失败: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return jsonify({'error': f'声纹特征提取失败: {str(e)}'}), 500

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print("临时文件已清理")
                except Exception as e:
                    print(f"清理临时文件时出错: {str(e)}")

    except Exception as e:
        print(f"声纹注册过程发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/identify_speaker', methods=['POST'])
def identify_speaker():
    try:
        if not speaker_recognition_model:
            return jsonify({'error': '声纹识别模型未初始化'}), 500

        if 'audio' not in request.files:
            return jsonify({'error': '未提供音频文件'}), 400

        audio_file = request.files['audio']

        # 创建临时目录
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # 保存音频文件
        temp_path = os.path.join(temp_dir, f'identify_{int(time.time())}_{os.urandom(4).hex()}.wav')
        try:
            audio_file.save(temp_path)

            # 提取声纹特征
            try:
                print("开始提取声纹特征...")
                # 加载音频文件并转换为tensor
                waveform, sample_rate = torchaudio.load(temp_path)
                # 确保音频是单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # 确保采样率为16kHz
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                print(f"音频数据形状: {waveform.shape}, 采样率: {sample_rate}")
                
                # 提取声纹特征
                test_embedding = speaker_recognition_model.encode_batch(waveform)
                print(f"声纹特征提取成功: {type(test_embedding)}, shape={test_embedding.shape if hasattr(test_embedding, 'shape') else 'no shape'}")
                
                # 将tensor转换为numpy数组
                test_embedding = test_embedding.squeeze().detach().cpu().numpy()
                print(f"转换后的特征形状: {test_embedding.shape}")

                # 如果没有注册的说话人
                if not SPEAKER_EMBEDDINGS:
                    return jsonify({
                        'success': True,
                        'speaker_name': None,
                        'confidence': 0.0,
                        'message': '未找到任何注册的说话人'
                    })

                # 计算与所有注册说话人的相似度
                max_similarity = -1
                identified_speaker = None

                for speaker_name, ref_embedding in SPEAKER_EMBEDDINGS.items():
                    # 计算余弦相似度
                    similarity = np.dot(test_embedding, ref_embedding) / (np.linalg.norm(test_embedding) * np.linalg.norm(ref_embedding))
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        identified_speaker = speaker_name

                # 设置阈值（可以根据需要调整）
                threshold = 0.7
                if max_similarity < threshold:
                    identified_speaker = None

                return jsonify({
                    'success': True,
                    'speaker_name': identified_speaker,
                    'confidence': float(max_similarity)
                })

            except Exception as e:
                print(f"声纹特征提取失败: {str(e)}")
                return jsonify({'error': str(e)}), 500

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False) 