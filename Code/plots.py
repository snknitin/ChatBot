import pickle
import shared_util as su




if __name__ == '__main__':
    input_dir = "/mnt/nfs/scratch1/nsamala/dialogsystems/ChatBot-Text-Summarizer/datasets/Cornell/"
    model_dir = input_dir + 'models/'
    plot_path=model_dir+"plot_losses.pickle"
    plot_losses=su.loadpickle(plot_path)
    su.showPlot(plot_losses)