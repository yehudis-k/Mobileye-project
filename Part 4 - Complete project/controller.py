from tfl_manager import *

class Controller:
    def __init__(self, playlist):
        self.playlist = playlist
        self.frames = list(open(self.playlist, 'r', encoding='utf8'))
        pkl_path = self.frames[0][:-1]
        self.frames = self.frames[1:]
        with open(pkl_path, 'rb') as pklfile:
            self.pkl = pickle.load(pklfile, encoding='latin1')
        self.TFL_manager = TFL_manager(self.pkl['principle_point'], self.pkl['flx'])


    def get_ego(self, prev_id, curr_id):
        EM = np.eye(4)
        for i in range(prev_id, curr_id):
            EM = np.dot(self.pkl['egomotion_' + str(i) + '-' + str(i + 1)], EM)
        return EM


    def add_border(self, image):
        return ImageOps.expand(image, border=41, fill='black')


    def run(self):
        first_id = int(self.frames[0][-23:-17])
        for i, image_path in enumerate(self.frames):
            if i == 0:
                em = None
            else:
                em = self.get_ego(first_id+i-1, first_id+i)

            image = Image.open(image_path[:-1]) # to remove \n from path
            self.TFL_manager.manage(image, em)


def main(playlist):
    controller = Controller("playlists/"+playlist)
    controller.run()


#if __name__ == '__main__':
main("playlist_49.txt")