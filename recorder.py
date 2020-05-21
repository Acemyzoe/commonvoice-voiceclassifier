import wave
from pyaudio import PyAudio, paInt16
from voicemodel_gender import deploy

framerate=16000
NUM_SAMPLES=2000
channels=1
sampwidth=2
TIME=10
filename = '001.mp3'

def save_wave_file(filename,data):
    '''save the data to the wavfile'''
    wf=wave.open(filename,'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()
    
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
    if deploy(filename) == 1:
        print('You are man')
    else:
        print('You are woman')
        
if __name__ == '__main__':
    #my_record()
    play()
