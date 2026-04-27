import traceback
import whisperx

print('whisperx', whisperx.__version__)

device = 'cpu'
try:
    model = whisperx.load_model('base', device, compute_type='int8')
    print('loaded model')
    audio = whisperx.load_audio('data/raw_audio/test_dictation.wav')
    print('loaded audio', len(audio))
    result = model.transcribe(audio, batch_size=8, language='en')
    print('transcribed', result.keys())
except Exception:
    traceback.print_exc()
