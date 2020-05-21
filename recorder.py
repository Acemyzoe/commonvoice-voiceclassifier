import os
import wave
from pyaudio import PyAudio, paInt16
from voicemodel_gender import deploy_gender
from voicemodel_age import deploy_age
import librosa
import soundfile as sf

NUM_SAMPLES=2000
framerate=16000    #采样频率
channels=1    #声道数量（1 为单声道，2 为立体声）
sampwidth=2    #采样字节长度
TIME=10
filename = 'tmp.wav'

def save_wave_file(filename,data):
    '''save the data to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

#录音(10s左右)   
def my_record():
    pa=PyAudio()
    stream=pa.open(format = paInt16,channels=1,
                   rate=framerate,input=True,
                   frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0
    print('Start your record')
    while count<TIME*8:#控制录音时间
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count+=1
        print('.')
    print('Done!') 
    save_wave_file(filename,my_buf)
    stream.close()

#播放录音，并分辨性别年龄
chunk=1024
def play():
    wf=wave.open(filename,'rb')
    p=PyAudio()
    stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),
                  channels=wf.getnchannels(),
                  rate=wf.getframerate(),
                  output=True)
    data=wf.readframes(chunk)
    print('Playing your record')
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # gender classifier
    if deploy_gender(filename) == 1:
        print('You are man')
    else:
        print('You are woman')
        
    # age classifier
    age_list = ['teens','twenties','thirties','fourties']
    age = deploy_age(filename)
    print('Your age:',age_list[age])

def main():
    if os.path.exists('demo.mp3'):
        # wave不支持64位RIFF文件,故用librosa读取文件，再将其写回到临时的wav文件中
        x,_ = librosa.load('demo.mp3', sr=16000)
        sf.write('tmp.wav', x, 16000)  
        play()
    else:
        my_record()
        play()
     
if __name__ == '__main__':
    main()
