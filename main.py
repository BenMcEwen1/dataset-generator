import bacpipe

def main():
    # to modify the audio data path for example, do
    bacpipe.config.audio_dir = r'test/'

    # to modify the models you want to run, do
    bacpipe.config.models = ['birdnet']
    # if you do not have the checkpoint yet, it will be automatically
    # downloaded and stored locally
    bacpipe.config.dashboard = False

    bacpipe.settings.device = 'cpu'
    # bacpipe uses multithreading which speeds up model inference if
    # run on a machine supporting cuda

    # then run with your settings.
    # By default the save logs is True
    bacpipe.play(save_logs=True)


if __name__ == "__main__":
    main()
